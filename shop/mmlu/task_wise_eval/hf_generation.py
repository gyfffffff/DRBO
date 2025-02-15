import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import evaluate
from rouge_score import rouge_scorer
from mmlu.task_wise_eval.utils import *
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import json

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def get_score(
    i, args, generation, label, metric, scorer, eval_model, sacrebleu, indicator2dist
):
    if i % args.print_interval == 0:
        print("Sample %d answer" % i, generation)
        print("Sample %d ground truth" % i, label)
    if metric == "rougel":
        scores = scorer.score(generation, label)
        scores = scores["rougeL"].fmeasure
        indicator2dist[metric].append(scores)
        return scores
    elif metric in ["sent-transformer", "multilingual-sent-transformer"]:
        if isinstance(label, str):
            truth_embedding = eval_model.encode([label])[0]
            generation_embedding = eval_model.encode([generation])[0]
            current_score = (generation_embedding * truth_embedding).sum()
            current_score /= np.linalg.norm(
                generation_embedding, ord=2
            ) * np.linalg.norm(truth_embedding, ord=2)
            indicator2dist[metric].append(current_score)
            return current_score.item()
        else:
            scores = []
            generation_embedding = eval_model.encode([generation])[0]
            for label_item in label:
                truth_embedding = eval_model.encode([label_item])[0]
                score_ = (generation_embedding * truth_embedding).sum()
                score_ /= np.linalg.norm(generation_embedding, ord=2) * np.linalg.norm(
                    truth_embedding, ord=2
                )
                scores.append(score_)
            current_score = np.mean(scores)
            indicator2dist[metric].append(current_score)
            return current_score.item()
    elif metric == "bleu":
        # usage: sacrebleu.compute(predictions=xxx, references=yyy)
        # reference can be multiple lists of sentences
        # candidate is a list of sentences
        generation = generation.lstrip("\n").rstrip("\n").split("\n")[0]
        candidate = [generation]
        reference = [[label]]
        if "JP" not in args.test_subject:
            # japanese needs a different tokenizer
            tokenize = "13a"
        else:
            tokenize = "ja-mecab"
        current_score = (
            sacrebleu.compute(
                predictions=candidate,
                references=reference,
                lowercase=True,
                tokenize=tokenize,
            )["score"]
            / 100
        )
        indicator2dist[metric].append(current_score)
        return current_score
    else:
        raise NotImplementedError("metric not implemented")


def cal_score(args, tokenizer, model, indicator2dist):
    start_time = time.time()
    test_subject = args.test_subject
    model_name = args.model_name
    gen_result = []

    logger.info("Running %s on %s task." % (model_name, test_subject))
    scorer = None
    sacrebleu = None
    eval_model = None
    # metric
    if "extraction" in args.test_subject:
        metric = "rougel"
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        logger.info("Metric is ROUGE-L score")
    elif "translation" in args.test_subject:
        metric = "bleu"
        sacrebleu = evaluate.load("sacrebleu")
        logger.info("Metric is BLEU score")
    elif "multilingual" in args.test_subject:
        metric = "multilingual-sent-transformer"
        eval_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").cuda()
        logger.info("Metric is multilingual sentence transformer similarity")
    else:
        metric = "sent-transformer"
        eval_model = SentenceTransformer("all-MiniLM-L6-v2").cuda()
        logger.info("Metric is sentence transformer similarity.")

    filename = f"/home/moon_shop/mmlu_data/generation/{test_subject}_dataset.json"
    try:
        test_df = pd.read_json(filename, lines=True)
    except:
        raise FileNotFoundError(
            f"{filename} does not exist. Please modify the 'test_subject' argument. "
        )
    total_score = 0
    all_samples = test_df.shape[0]
    samples = []
    for i in range(all_samples):
        train_prompt = gen_system_prompt(args)
        test_prompt = format_example(test_df, i)
        prompt = train_prompt + test_prompt

        if i % args.print_interval == 0:
            print("Sample %d prompt" % i, prompt)
        label = test_df.iloc[i, -1]
        inputs = tokenizer(prompt)

        samples.append(
            {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "label": label,
            }
        )
    sample_count = 0
    for j in range(0, len(samples) // args.batch_size + 1):
        start_idx = j * args.batch_size
        end_idx = min((j + 1) * args.batch_size, len(samples))
        batch_samples = samples[start_idx:end_idx]
        if len(batch_samples) == 0:
            continue
        inputs_ids = [bs["input_ids"] for bs in batch_samples]
        attention_mask = [bs["attention_mask"] for bs in batch_samples]
        labels = [bs["label"] for bs in batch_samples]
        max_length = max([len(ii) for ii in inputs_ids])
        inputs_ids = [
            [tokenizer.eos_token_id] * max(0, max_length - len(ii)) + ii
            for ii in inputs_ids
        ]
        attention_mask = [
            [0] * max(0, max_length - len(ii)) + ii for ii in attention_mask
        ]
        inputs_ids = torch.LongTensor(inputs_ids).cuda()
        attention_mask = torch.LongTensor(attention_mask).cuda()
        if (
            "mistral" in args.model_name
            or "mixtral" in args.model_name
            or "zephyr" in args.model_name
            or "ecellm-m" == args.model_name
        ):
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_gen_len,
                pad_token_id=2,
                eos_token_id=tokenizer.eos_token_id,
            )
        elif "qwen" in args.model_name:
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_gen_len,
                pad_token_id=151643,
                temperature=0.0001,
                eos_token_id=tokenizer.eos_token_id,
            )
        elif "phi" in args.model_name or args.model_name == "ecellm-s":
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_gen_len,
                pad_token_id=50256,
                temperature=0.0001,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_gen_len,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        result = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for i, cur_res in enumerate(result):
            sample_count += 1
            cur_input_ids = inputs_ids[i]
            prompt = tokenizer.decode(
                cur_input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generation = cur_res[len(prompt) :]  # .split('\n')[0]
            cur_label = labels[i]

            current_score = get_score(
                i=sample_count,
                args=args,
                generation=generation,
                label=cur_label,
                metric=metric,
                scorer=scorer,
                eval_model=eval_model,
                sacrebleu=sacrebleu,
                indicator2dist=indicator2dist,
            )
            gen_result.append(
                {
                    "prompt": prompt,
                    "generation": generation,
                    "label": cur_label,
                    "metric": metric,
                    "score": current_score,
                }
            )
            total_score += current_score
            print("Sample %d score" % sample_count, current_score)
            print()

    logger.info(
        "The average score of %s on %s task is %.4f"
        % (model_name, test_subject, total_score / all_samples)
    )
    logger.info("Time Cost: %.4fs" % (time.time() - start_time))
    return total_score / all_samples, time.time() - start_time, gen_result


class Args:
    test_subject: str = ""
    model_name: str = ""
    use_task_specific_prompt: bool = True
    print_interval: int = 20
    max_gen_len: int = 128
    batch_size: int = 16


def main(model_name="/ssd1/models/Llama-3.2-3B-Instruct", model=None, tokenizer=None):
    args = Args()
    generate_dir = "/home/moon_shop/mmlu_data/generation"
    # model_name = "/ssd1/models/Meta-Llama-3-8B-Instruct"
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    args.model_name = model_name
    res = []
    indicator2dist = {
        "rougel": [],
        "sent-transformer": [],
        "multilingual-sent-transformer": [],
        "bleu": [],
    }
    all_gen_result = []
    for dataset_name in os.listdir(generate_dir):
        dataset_name = dataset_name.split("dataset")[0].strip("_")
        if "translation" not in dataset_name:
            continue

        args.test_subject = dataset_name
        average_score, cost_time, gen_result = cal_score(
            args, tokenizer, model, indicator2dist
        )
        res.append(
            {
                "dataset_name": dataset_name,
                "model": model_name,
                "average_score": str(average_score),
                "cost_time": str(cost_time),
            }
        )
        all_gen_result.extend(gen_result)
    with open(
        f"{os.path.split(model_name)[-1]}_generate_score.json", "w", encoding="utf8"
    ) as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    indicator2dist = {
        k: {"average": np.mean(np.array(v)).item(), "std": np.std(np.array(v)).item()}
        for k, v in indicator2dist.items()
    }
    with open(
        f"{os.path.split(model_name)[-1]}_generation_score_dist.json",
        "w",
        encoding="utf8",
    ) as f:
        json.dump(indicator2dist, f, ensure_ascii=False, indent=4)

    with open(
        f"{os.path.split(model_name)[-1]}_generation_result.json", "w", encoding="utf8"
    ) as f:
        json.dump(all_gen_result, f, ensure_ascii=False, indent=4)
    return res


if __name__ == "__main__":
    main()
