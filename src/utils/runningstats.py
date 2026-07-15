"""
To use a runningstats object,

    1. Create the the desired stat object, e.g., `m = Mean()`
    2. Feed it batches via the add method, e.g., `m.add(batch)`
    3. Repeat step 2 any number of times.
    4. Read out the statistic of interest, e.g., `m.mean()`

Running-stat objects used by this project include:

    Mean - produces mean().
    NormMean - produces the average norm of input vectors.
    SecondMoment - moment() is the non-mean-centered covariance, E[x x^T].
    CombinedStat - aggregates any set of stats.

Add more running stats by subclassing the Stat class.

These statistics are vectorized along dim>=1, so stat.add()
should supply a two-dimensional input where the zeroth
dimension is the batch/sampling dimension and the first
dimension is the feature dimension.

The data type and device used matches the data passed to add();
for example, for higher-precision covariances, convert to double
before calling add().

It is common to want to compute and remember a statistic sampled
over a Dataset, computed in batches, possibly caching the computed
statistic in a file. The tally(stat, dataset, cache) handles
this pattern.  It takes a statistic, a dataset, and a cache filename
and sets up a data loader that can be run (or not, if cached) to
compute the statistic, adopting the convention that cached stats are
saved to and loaded from numpy npz files.
"""

import os
import random
import struct

import numpy
import torch
from torch.utils.data.sampler import Sampler


def tally(stat, dataset, cache=None, quiet=False, **kwargs):
    """
    To use tally, write code like the following.

        stat = Mean()
        ds = MyDataset()
        for batch in tally(stat, ds, cache='mymean.npz', batch_size=50):
           stat.add(batch)
        mean = stat.mean()

    The first argument should be the Stat being computed. After the
    loader is exhausted, tally will bring this stat to the cpu and
    cache it (if a cache is specified).

    The dataset can be a torch Dataset or a plain Tensor, or it can
    be a callable that returns one of those.

    Details on caching via the cache= argument:

        If the given filename cannot be loaded, tally will leave the
        statistic object empty and set up a DataLoader object so that
        the loop can be run.  After the last iteration of the loop, the
        completed statistic will be moved to the cpu device and also
        saved in the cache file.

        If the cached statistic can be loaded from the given file, tally
        will not set up the data loader and instead will return a fully
        loaded statistic object (on the cpu device) and an empty list as
        the loader.

        The `with cache_load_enabled(False):` context manager can
        be used to disable loading from the cache.

    If needed, a DataLoader will be created to wrap the dataset:

        Keyword arguments of tally are passed to the DataLoader,
        so batch_size, num_workers, pin_memory, etc. can be specified.

    Subsampling is supported via sample_size= and random_sample=:

        If sample_size=N is specified, rather than loading the whole
        dataset, only the first N items are sampled.  If additionally
        random_sample=S is specified, the pseudorandom seed S will be
        used to select a fixed psedorandom sample of size N to sample.
    """
    assert isinstance(stat, Stat)
    args = {}
    for k in ["sample_size"]:
        if k in kwargs:
            args[k] = kwargs[k]
    cached_state = load_cached_state(cache, args, quiet=quiet)
    if cached_state is not None:
        stat.load_state_dict(cached_state)

        def empty_loader():
            return
            yield

        return empty_loader()
    loader = make_loader(dataset, **kwargs)

    def wrapped_loader():
        yield from loader
        stat.to_(device="cpu")
        if cache is not None:
            save_cached_state(cache, stat, args)

    return wrapped_loader()


class cache_load_enabled:
    """
    When used as a context manager, cache_load_enabled(False) will prevent
    tally from loading cached statsitics, forcing them to be recomputed.
    """

    def __init__(self, enabled=True):
        self.prev = False
        self.enabled = enabled

    def __enter__(self):
        global global_load_cache_enabled
        self.prev = global_load_cache_enabled
        global_load_cache_enabled = self.enabled

    def __exit__(self, exc_type, exc_value, traceback):
        global global_load_cache_enabled
        global_load_cache_enabled = self.prev


class Stat:
    """
    Abstract base class for a running pytorch statistic.
    """

    def __init__(self, state):
        """
        By convention, all Stat subclasses can be initialized by passing
        state=; and then they will initialize by calling load_state_dict.
        """
        self.load_state_dict(resolve_state_dict(state))

    def add(self, x, *args, **kwargs):
        """
        Observes a batch of samples to be incorporated into the statistic.
        Dimension 0 should be the batch dimension, and dimension 1 should
        be the feature dimension of the pytorch tensor x.
        """
        pass

    def load_state_dict(self, d):
        """
        Loads this Stat from a dictionary of numpy arrays as saved
        by state_dict.
        """
        pass

    def state_dict(self):
        """
        Saves this Stat as a dictionary of numpy arrays that can be
        stored in an npz or reloaded later using load_state_dict.
        """
        return {}

    def save(self, filename):
        """
        Saves this stat as an npz file containing the state_dict.
        """
        save_cached_state(filename, self, {})

    def load(self, filename):
        """
        Loads this stat from an npz file containing a saved state_dict.
        """
        self.load_state_dict(load_cached_state(filename, {}, quiet=True, throw=True))

    def to_(self, device):
        """
        Moves this Stat to the given device.
        """
        pass

    def cpu_(self):
        """
        Moves this Stat to the cpu device.
        """
        self.to_("cpu")

    def cuda_(self):
        """
        Moves this Stat to the default cuda device.
        """
        self.to_("cuda")

    def _normalize_add_shape(self, x, attr="data_shape"):
        """
        Flattens input data to 2d.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if len(x.shape) < 1:
            x = x.view(-1)
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            data_shape = x.shape[1:]
            setattr(self, attr, data_shape)
        else:
            assert x.shape[1:] == data_shape
        return x.view(x.shape[0], int(numpy.prod(data_shape)))

    def _restore_result_shape(self, x, attr="data_shape"):
        """
        Restores output data to input data shape.
        """
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            return x
        return x.view(data_shape * len(x.shape))


class Mean(Stat):
    """
    Running mean.
    """

    def __init__(self, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self.batchcount = 0
        self._mean = None
        self.data_shape = None

    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        batch_count = a.shape[0]
        batch_mean = a.sum(0) / batch_count
        self.batchcount += 1
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = batch_mean
            return
        # Update a batch using Chan-style update for numerical stability.
        self.count += batch_count
        new_frac = float(batch_count) / self.count
        # Update the mean according to the batch deviation from the old mean.
        delta = batch_mean.sub_(self._mean).mul_(new_frac)
        self._mean.add_(delta)

    def size(self):
        return self.count

    def mean(self):
        return self._restore_result_shape(self._mean)

    def to_(self, device):
        if self._mean is not None:
            self._mean = self._mean.to(device)

    def load_state_dict(self, state):
        self.count = state["count"]
        self.batchcount = state["batchcount"]
        self._mean = torch.from_numpy(state["mean"])
        self.data_shape = (
            None if state["data_shape"] is None else tuple(state["data_shape"])
        )

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            data_shape=self.data_shape and tuple(self.data_shape),
            batchcount=self.batchcount,
            mean=self._mean.cpu().numpy(),
        )


class NormMean(Mean):
    """
    Running average of the norm of input vectors
    """

    def __init__(self, state=None):
        super().__init__(state)

    def add(self, a):
        super().add(a.norm(dim=-1))


class SecondMoment(Stat):
    """
    Running computation. Use this when the entire non-centered 2nd-moment
    'covariance-like' matrix is needed, and when the whole matrix fits
    in the GPU.
    """

    def __init__(self, split_batch=True, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self.mom2 = None
        self.split_batch = split_batch

    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        # Initial batch reveals the shape of the data.
        if self.count == 0:
            self.mom2 = a.new(a.shape[1], a.shape[1]).zero_()
        batch_count = a.shape[0]
        # Update the covariance using the batch deviation
        self.count += batch_count
        self.mom2 += a.t().mm(a)

    def to_(self, device):
        if self.mom2 is not None:
            self.mom2 = self.mom2.to(device)

    def moment(self):
        return self.mom2 / self.count

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            mom2=self.mom2.cpu().numpy(),
        )

    def load_state_dict(self, state):
        self.count = int(state["count"])
        self.mom2 = torch.from_numpy(state["mom2"])



class CombinedStat(Stat):
    """
    A Stat that bundles together multiple Stat objects.
    Convenient for loading and saving a state_dict made up of a
    hierarchy of stats, and for use with the tally() function.
    Example:

        cs = CombinedStat(m=Mean(), q=Quantile())
        for [b] in tally(cs, MyDataSet(), cache=fn, batch_size=100):
            cs.add(b)
        print(cs.m.mean())
        print(cs.q.median())
    """

    def __init__(self, state=None, **kwargs):
        self._objs = kwargs
        if state is not None:
            return super().__init__(state)

    def __getattr__(self, k):
        if k in self._objs:
            return self._objs[k]
        raise AttributeError()

    def add(self, d, *args, **kwargs):
        for obj in self._objs.values():
            obj.add(d, *args, **kwargs)

    def load_state_dict(self, state):
        for prefix, obj in self._objs.items():
            obj.load_state_dict(pull_key_prefix(prefix, state))

    def state_dict(self):
        result = {}
        for prefix, obj in self._objs.items():
            result.update(push_key_prefix(prefix, obj.state_dict()))
        return result

    def to_(self, device):
        """Switches internal storage to specified device."""
        for v in self._objs.values():
            v.to_(device)


def push_key_prefix(prefix, d):
    """
    Returns a dict with the same values as d, but where each key
    adds the prefix, followed by a dot.
    """
    return {prefix + "." + k: v for k, v in d.items()}


def pull_key_prefix(prefix, d):
    """
    Returns a filtered dict of all the items of d that start with
    the given key prefix, plus a dot, with that prefix removed.
    """
    pd = prefix + "."
    lpd = len(pd)
    return {k[lpd:]: v for k, v in d.items() if k.startswith(pd)}


# We wish to be able to save None (null) values in numpy npz files,
# yet do so without setting the unsecure 'allow_pickle' flag.  To do
# that, we will encode null as a special kind of IEEE 754 NaN value.
# Inspired by https://github.com/zuiderkwast/nanbox/blob/master/nanbox.h
# we follow the same Nanboxing scheme used in JavaScriptCore
# (search for JSCJSValue.h#L435), which encodes null values in NaN
# as the NaN value with hex pattern 0xfff8000000000002.

null_numpy_value = numpy.array(
    struct.unpack(">d", struct.pack(">Q", 0xFFF8000000000002))[0], dtype=numpy.float64
)


def is_null_numpy_value(v):
    """
    True if v is a 64-bit float numpy scalar NaN matching null_numpy_value.
    """
    return (
        isinstance(v, numpy.ndarray)
        and numpy.ndim(v) == 0
        and v.dtype == numpy.float64
        and numpy.isnan(v)
        and 0xFFF8000000000002 == struct.unpack(">Q", struct.pack(">d", v))[0]
    )


def box_numpy_null(d):
    """
    Replaces None with null_numpy_value, leaving non-None values unchanged.
    Recursively descends into a dictionary replacing None values.
    """
    try:
        return {k: box_numpy_null(v) for k, v in d.items()}
    except Exception:
        return null_numpy_value if d is None else d


def unbox_numpy_null(d):
    """
    Reverses box_numpy_null, replacing null_numpy_value with None.
    Recursively descends into a dictionary replacing None values.
    """
    try:
        return {k: unbox_numpy_null(v) for k, v in d.items()}
    except Exception:
        return None if is_null_numpy_value(d) else d


def resolve_state_dict(s):
    """
    Resolves a state, which can be a filename or a dict-like object.
    """
    if isinstance(s, str):
        return unbox_numpy_null(numpy.load(s))
    return s


global_load_cache_enabled = True


def load_cached_state(cachefile, args, quiet=False, throw=False):
    """
    Resolves a state, which can be a filename or a dict-like object.
    """
    if not global_load_cache_enabled or cachefile is None:
        return None
    try:
        if isinstance(cachefile, dict):
            dat = cachefile
            cachefile = "state"  # for printed messages
        else:
            dat = unbox_numpy_null(numpy.load(cachefile))
        for a, v in args.items():
            if a not in dat or dat[a] != v:
                if not quiet:...
                    # print("%s %s changed from %s to %s" % (cachefile, a, dat[a], v))
                return None
    except (FileNotFoundError, ValueError) as e:
        if throw:
            raise e
        return None
    else:
        if not quiet:...
            # print("Loading cached %s" % cachefile)
        return dat


def save_cached_state(cachefile, obj, args):
    """
    Saves the state_dict of the given object in a dict or npz file.
    """
    if cachefile is None:
        return
    dat = obj.state_dict()
    for a, v in args.items():
        if a in dat:
            assert dat[a] == v
        dat[a] = v
    if isinstance(cachefile, dict):
        cachefile.clear()
        cachefile.update(dat)
    else:
        os.makedirs(os.path.dirname(cachefile), exist_ok=True)
        numpy.savez(cachefile, **box_numpy_null(dat))


class FixedSubsetSampler(Sampler):
    """Represents a fixed sequence of data set indices.
    Subsets can be created by specifying a subset of output indexes.
    """

    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

class FixedRandomSubsetSampler(FixedSubsetSampler):
    """Samples a fixed number of samples from the dataset, deterministically.
    Arguments:
        data_source,
        sample_size,
        seed (optional)
    """

    def __init__(self, data_source, start=None, end=None, seed=1):
        rng = random.Random(seed)
        shuffled = list(range(len(data_source)))
        rng.shuffle(shuffled)
        super(FixedRandomSubsetSampler, self).__init__(shuffled[start:end])


def make_loader(
    dataset, sample_size=None, batch_size=1, sampler=None, random_sample=None, **kwargs
):
    """Utility for creating a dataloader on fixed sample subset."""
    import typing

    if isinstance(dataset, typing.Callable):
        # To support deferred dataset loading, support passing a factory
        # that creates the dataset when called.
        dataset = dataset()
    if isinstance(dataset, torch.Tensor):
        # The dataset can be a simple tensor.
        dataset = torch.utils.data.TensorDataset(dataset)
    if sample_size is not None:
        assert sampler is None, "sampler cannot be specified with sample_size"
        if sample_size > len(dataset):
            # print(
            #     "Warning: sample size %d > dataset size %d"
            #     % (sample_size, len(dataset))
            # )
            sample_size = len(dataset)
        if random_sample is None:
            sampler = FixedSubsetSampler(list(range(sample_size)))
        else:
            sampler = FixedRandomSubsetSampler(
                dataset, seed=random_sample, end=sample_size
            )
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, **kwargs
    )
