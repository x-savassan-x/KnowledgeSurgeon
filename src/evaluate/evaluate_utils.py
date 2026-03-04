import torch
import numpy as np
import typing
import openai
from openai import OpenAI
import time
import regex
import string


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def llm_judge(question, ground_truth, prediction, api_key):
    content_template = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT"].

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: Malia and Sasha Obama are the names of Barack Obama's children.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Malia and Sasha, Malia and Sasha, Malia and Sasha, Malia and Sasha (repeated answer)
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target or contain repeated answer.


Here is a sample. Simply reply with either CORRECT or INCORRECT.

```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

According to the gold target, please grade the predicted answer of this question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
    """.strip()

    content = content_template.format(
        question=question,
        target=ground_truth,
        predicted_answer=prediction,
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": content}
        ],
        temperature=0.0
    )
    llm_ans = completion.choices[0].message.content
    llm_score = 1.0 if llm_ans == "A" else 0.0
    time.sleep(1) # avoid high rate of request
    return llm_score

def test_prediction_acc_LLM_judge(model, tok, hparams, prompts, targets, device, locality=False):
    """Generation-based evaluation with optional LLM-as-a-Judge scoring."""
    all_score = []
    all_response = []
    if isinstance(prompts, str):
        prompts, targets = [prompts, ], [targets, ]
    for prompt in prompts:
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_tok = tok(
            text,
            padding = True,
            truncation = True,
            return_tensors="pt",
        ).to(f"cuda:{device}")
        gen_tokens = model.generate(
            input_ids=prompt_tok['input_ids'],
            attention_mask=prompt_tok['attention_mask'],
            max_new_tokens=512,
            stop_strings=[".", "\n", tok.eos_token],
            tokenizer=tok,
            pad_token_id=tok.eos_token_id,
            do_sample=False,
            use_cache=False,
        )
        # decoder-only: strip the prompt prefix from generated tokens
        trunc_gen_tokens = gen_tokens[0][prompt_tok['input_ids'].shape[1]:]
        gen_content = tok.decode(trunc_gen_tokens)
        suffixes_to_remove = [".", "\n", tok.eos_token]
        for suffix in suffixes_to_remove:
            if gen_content.endswith(suffix):
                gen_content = gen_content[:-len(suffix)]
        # LLM-as-a-Judge
        if hparams.evaluation_type == "generate-text":
            all_response.append(gen_content)
        elif hparams.evaluation_type == "LLM-judge" and hasattr(hparams, 'api_key') and hparams.api_key:
            LLM_Score = llm_judge(prompts, targets, gen_content, hparams.api_key)
            all_score.append(LLM_Score)
            all_response.append(gen_content)
        else:
            # the user did not provide api key, using exact match as an alternative
            EM_Score = float(exact_match_score(gen_content, targets))
            all_score.append(EM_Score)
            all_response.append(gen_content)
    
    if len(all_score) > 0:
        return all_score, all_response
    else:
        return all_response

def test_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False, vanilla_generation=False):
    """
    Core accuracy evaluator.
    
    vanilla_generation=True:  autoregressive decode, compare generated tokens to target.
    vanilla_generation=False: teacher-forcing, feed prompt+target, check argmax matches.
    """
    if vanilla_generation:
        if isinstance(prompts, str):
            prompts, targets = [prompts, ], [targets, ]
        results = []
        for prompt, target_new in zip(prompts, targets):
            target_new_tokens = tok.encode(target_new, add_special_tokens=False)
            prompt_tok = tok(
                prompt,
                return_tensors="pt",
            ).to(f"cuda:{device}")
            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=len(target_new_tokens),
                pad_token_id=tok.eos_token_id,
                do_sample=False,
                use_cache=False,
            )
            if locality:
                results.append(gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])
            else:
                results.append(np.mean(np.equal(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])))
        return results

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    if not locality and hasattr(hparams, 'use_chat_template') and hparams.use_chat_template:
        prompts = [[{"role":"user", "content":m}] for m in prompts]
        prompts=tok.apply_chat_template(prompts,
                                        add_generation_prompt=True,
                                        tokenize=False)
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    before_padding_side = tok.padding_side
    tok.padding_side = 'left'
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    )
    tok.padding_side = before_padding_side
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)
        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            return [np.mean(np.equal(answers, labels))]

def slice_list(matrix, start_indices, left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]
