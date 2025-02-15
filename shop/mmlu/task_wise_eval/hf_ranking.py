import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import time
import random
import transformers
import numpy as np
from mmlu.task_wise_eval.utils import *
from loguru import logger


# This type of tasks ask an LLM to generate a re-ranked list of indices
# the evaluation metric is NDCG
# This applies to a range of re-ranking tasks
# such as recommendation


def ndcg(ranked_list, weight):
    idcg = 0
    dcg = 0
    for i in range(len(ranked_list)):
        position = i + 1
        if ranked_list[i] - 1 < len(weight):
            relevance = weight[ranked_list[i] - 1]
        else:
            relevance = 0
        dcg += (np.power(2, relevance) - 1) / np.log2(position + 1)
    weight.sort(reverse=True)
    for i in range(len(weight)):
        position = i + 1
        relevance = weight[i]
        idcg += (np.power(2, relevance) - 1) / np.log2(position + 1)

    return dcg / idcg


def is_permutation(arr):
    n = len(arr)
    expected_set = set(range(1, n + 1))
    return set(arr) == expected_set


def cal_score(args, tokenizer, model, indicator2dist):
    seed = args.seed
    if seed > 0:
        random.seed(seed)
        transformers.set_seed(seed)
    model_name = args.model_name
    test_subject = args.test_subject
    print_interval = args.print_interval

    start_time = time.time()
    logger.info("Running %s model on %s task" % (model_name, test_subject))

    ill_format = 0
    total_ndcg = 0
    filename = f"/home/moon_shop/mmlu_data/ranking/{test_subject}_dataset.json"
    try:
        test_df = pd.read_json(filename, lines=True)
    except:
        raise FileNotFoundError(
            f"{filename} does not exist. Please check the 'test_subject' argument."
        )
    all_samples = test_df.shape[0]
    samples = []
    for i in range(all_samples):
        train_prompt = gen_system_prompt(args)
        test_prompt = format_example(test_df, i)
        prompt = train_prompt + test_prompt
        if i % args.print_interval == 0:
            print("Sample %d prompt" % i, prompt)
        weight = test_df.iloc[i, -1]
        inputs = tokenizer(prompt)

        samples.append(
            {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "weight": weight,
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
        weights = [bs["weight"] for bs in batch_samples]
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
        with torch.no_grad():
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_gen_len,
                temperature=0.001,
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
            weight = weights[i]
            if sample_count % args.print_interval == 0:
                print("Sample %d generation" % sample_count, generation)
            answer = generation
            if "\n" in answer:
                answer = answer.lstrip("\n").rstrip("\n")
                answer = answer.split("\n")[0]
            ranked_list_str = answer.lstrip().rstrip().split(",")
            ranked_list = []
            for x in ranked_list_str:
                try:
                    ranked_list.append(int(x))
                except:
                    ill_format += 1
                    if sample_count % args.print_interval == 0:
                        print("Sample %d ill format" % sample_count)
                    continue

            if len(ranked_list) != len(weight):
                ill_format += 1
                if sample_count % args.print_interval == 0:
                    print("Sample %d ill format" % sample_count)

            if not is_permutation(ranked_list):
                ill_format += 1
                if sample_count % args.print_interval == 0:
                    print("Sample %d ill format" % sample_count)

            # start computing
            if sample_count % args.print_interval == 0:
                print("Sample %d weight" % sample_count, weight)
            ndcg_val = ndcg(ranked_list, weight)
            indicator2dist["ndcg"].append(ndcg_val)
            total_ndcg += ndcg_val
            if sample_count % args.print_interval == 0:
                print("Sample %d ranking" % sample_count, ranked_list)
                print("Sample %d ndcg" % sample_count, ndcg_val)
                print()

    logger.info("ranking task %s with %s:" % (test_subject, model_name))
    logger.info(
        "Average NDCG %.4f, %d ill-formatted generations"
        % (total_ndcg / all_samples, ill_format)
    )
    logger.info("Time cost %.4fs" % (time.time() - start_time))
    return total_ndcg / all_samples, ill_format, time.time() - start_time


class Args:
    test_subject: str = ""
    model_name: str = ""
    use_task_specific_prompt: bool = True
    print_interval: int = 20
    max_gen_len: int = 20
    seed: int = -1
    use_letter_choices: bool = True
    batch_size: int = 16


def main(model_name="/ssd1/models/Llama-3.1-8B-Instruct", model=None, tokenizer=None):
    args = Args()
    generate_dir = "/home/moon_shop/mmlu_data/ranking"
    # model_name = "/ssd1/models/Meta-Llama-3-8B-Instruct"
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    res = []
    indicator2dist = {
        "ndcg": [],
    }
    for dataset_name in os.listdir(generate_dir):
        dataset_name = dataset_name.split("dataset")[0].strip("_")
        args.test_subject = dataset_name
        score, ill_format, cost_time = cal_score(args, tokenizer, model, indicator2dist)
        res.append(
            {
                "dataset": dataset_name,
                "score": str(score),
                "ill_format": str(ill_format),
                "cost_time": str(cost_time),
                "model": model_name,
            }
        )
    indicator2dist = {
        k: {"average": np.mean(np.array(v)).item(), "std": np.std(np.array(v)).item()}
        for k, v in indicator2dist.items()
    }
    with open(
        f"{os.path.split(model_name)[-1]}_ndcg_score_dist.json", "w", encoding="utf8"
    ) as f:
        json.dump(indicator2dist, f, ensure_ascii=False, indent=4)
    with open(
        f"{os.path.split(model_name)[-1]}_ranking.json", "w", encoding="utf-8"
    ) as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    return res


if __name__ == "__main__":
    main()
