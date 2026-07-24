"""
Utilities for instrumenting a torch model.

Trace will hook one layer at a time.
TraceDict will hook multiple layers at once.
get_module and get_parameter resolve dotted names.
set_requires_grad recursively sets requires_grad in module parameters.
"""

import contextlib
import inspect
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Self

import torch


class Trace(contextlib.AbstractContextManager):
    """
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

        retain_input=True - also retains the input.
        retain_output=False - can disable retaining the output.
        edit_output=fn - calls the function to modify the output
            of the layer before passing it the rest of the model.
            fn can optionally accept (output, layer) arguments
            for the original output and the layer name.
        stop=True - throws a StopForward exception after the layer
            is run, which allows running just a portion of a model.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        layer: str | None = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Callable[..., Any] | None = None,
        stop: bool = False,
    ) -> None:
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        retainer = self
        self.layer = layer
        if layer is not None:
            module = get_module(module, layer)

        def retain_hook(
            m: torch.nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
            output,
        ) -> Any:
            """with_kwargs=True hook: PyTorch calls as (module, args, kwargs, output)."""
            if retain_input:
                # Try to get input from positional args first
                if len(inputs) > 0:
                    retainer.input = recursive_copy(
                        inputs[0] if len(inputs) == 1 else inputs,
                        clone=clone,
                        detach=detach,
                        retain_grad=False,
                    )
                # If inputs is empty, try to get hidden_states from kwargs (modern transformers)
                elif kwargs is not None and 'hidden_states' in kwargs:
                    retainer.input = recursive_copy(
                        kwargs['hidden_states'],
                        clone=clone,
                        detach=detach,
                        retain_grad=False,
                    )
                else:
                    retainer.input = None
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        # Try to use with_kwargs=True for PyTorch 2.0+ to capture kwargs like hidden_states
        try:
            # PyTorch 2.0+ supports with_kwargs parameter
            self.registered_hook = module.register_forward_hook(retain_hook, with_kwargs=True)
        except TypeError:
            # Fallback for older PyTorch versions — no kwargs available
            def legacy_hook(
                m: torch.nn.Module, inputs: tuple[Any, ...], output
            ) -> Any:
                return retain_hook(m, inputs, {}, output)
            self.registered_hook = module.register_forward_hook(legacy_hook)
        self.stop = stop

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback,
    ) -> bool | None:
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self) -> None:
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    To retain the output of multiple named layers during the computation
    of the given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the network will be stopped after the last layer
    listed (even if it would not have been the last to be executed).
    """

    def __init__(
        self,
        module: torch.nn.Module,
        layers: Iterable[str] | None = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Callable[..., Any] | None = None,
        stop: bool = False,
    ) -> None:
        self.stop = stop

        def flag_last_unseen(it: Iterable[str]) -> Iterator[tuple[bool, str]]:
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback,
    ) -> bool | None:
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self) -> None:
        for layer, trace in reversed(self.items()):
            trace.close()


class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """

    pass


def get_hidden_state(output) -> torch.Tensor:
    """
    Return the primary hidden-state tensor from a transformer layer output.
    Decoder layers may return the tensor directly or wrap it as the first
    item of a tuple/list.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("Cannot read hidden state from an empty layer output.")
        return output[0]
    raise TypeError(f"Unsupported layer output type: {type(output)}")


def replace_hidden_state(output, hidden_state: torch.Tensor) -> Any:
    """
    Replace the primary hidden-state tensor while preserving the original
    output container type.
    """
    if isinstance(output, torch.Tensor):
        return hidden_state
    if isinstance(output, list):
        updated = list(output)
        updated[0] = hidden_state
        return updated
    if isinstance(output, tuple):
        updated = list(output)
        updated[0] = hidden_state
        return type(output)(updated)
    raise TypeError(f"Unsupported layer output type: {type(output)}")


def recursive_copy(
    x,
    clone: bool | None = None,
    detach: bool | None = None,
    retain_grad: bool | None = None,
) -> Any:
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


def set_requires_grad(requires_grad: bool, *models) -> None:
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_parameter(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def invoke_with_optional_args(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Invokes a function with only the arguments that it
    is written to accept, giving priority to arguments
    that match by-name, using the following rules.
    (1) arguments with matching names are passed by name.
    (2) remaining non-name-matched args are passed by order.
    (3) extra caller arguments that the function cannot
        accept are not passed.
    (4) extra required function arguments that the caller
        cannot provide cause a TypeError to be raised.
    Ordinary python calling conventions are helpful for
    supporting a function that might be revised to accept
    extra arguments in a newer version, without requiring the
    caller to pass those new arguments.  This function helps
    support function callers that might be revised to supply
    extra arguments, without requiring the callee to accept
    those new arguments.
    """
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # Pass positional args that match name first, then by position.
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
    # Fill unmatched positional args with unmatched keyword args in order.
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
    # Pass remaining kw args if they can be accepted.
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    # Pass remaining positional args if they can be accepted.
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)
