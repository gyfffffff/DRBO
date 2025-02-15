from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import time
import random
import transformers
from mmlu.task_wise_eval.utils import *
import os
import json
from loguru import logger
import numpy as np


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
    total_hit = 0

    filename = f"/home/moon_shop/mmlu_data/retrieval/{test_subject}_dataset.json"
    try:
        test_df = pd.read_json(filename, lines=True)
    except:
        raise FileNotFoundError(
            f"{filename} does not exist. Please check the 'test_subject' argument. "
        )
    all_samples = test_df.shape[0]
    samples = []
    for i in range(all_samples):
        system_prompt = gen_system_prompt(args)
        test_prompt = format_example(test_df, i)
        prompt = system_prompt + test_prompt
        if i % args.print_interval == 0:
            print("Sample %d prompt" % i, prompt)
        truth = test_df.iloc[i, -1]
        inputs = tokenizer(prompt)

        samples.append(
            {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "truth": truth,
            }
        )

    sample_count = 0
    for j in range(0, len(samples) // args.batch_size + 1):
        start_idx = j * args.batch_size
        end_idx = min((j + 1) * args.batch_size, len(samples))
        batch_samples = samples[start_idx:end_idx]
        inputs_ids = [bs["input_ids"] for bs in batch_samples]
        attention_mask = [bs["attention_mask"] for bs in batch_samples]
        truth_list = [bs["truth"] for bs in batch_samples]
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
            "mistral" in model_name
            or "mixtral" in model_name
            or model_name == "zephyr"
            or model_name == "ecellm-m"
        ):
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_gen_len,
                pad_token_id=2,
            )
        elif "qwen" in model_name:
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_gen_len,
                pad_token_id=151643,
                temperature=0.0001,
            )
        elif "phi" in args.model_name or args.model_name == "ecellm-s":
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_gen_len,
                pad_token_id=50256,
                temperature=0.0001,
            )
        else:
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_gen_len,
                pad_token_id=tokenizer.eos_token_id,
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
            truth = truth_list[i]

            if sample_count % args.print_interval == 0:
                print("Sample %d generation" % sample_count, generation)

            retrieved_list = generation
            retrieved_list = retrieved_list.lstrip().rstrip()
            if "\n" not in retrieved_list:
                retrieved_list = retrieved_list.split(",")
            else:
                retrieved_list = retrieved_list.split("\n")[0]
                retrieved_list = retrieved_list.split(",")
            retrieved_int = []
            for ret in retrieved_list:
                try:
                    ret_int = int(ret)
                    retrieved_int.append(ret_int)
                except:
                    logger.info(f"{ret} cannot be interpreted as int")
                    continue
            if len(retrieved_int) > 3:
                retrieved_int = retrieved_int[:3]

            if sample_count % args.print_interval == 0:
                print("Sample %d truth" % sample_count, truth)
            hit = len(set(truth).intersection(set(retrieved_int)))
            hit /= len(truth)
            indicator2dist["hit_rate@3"].append(hit)
            total_hit += hit

            if sample_count % args.print_interval == 0:
                print("Sample %d retrieval" % sample_count, retrieved_list)
                print("Sample %d hit" % sample_count, hit)
                print()

    logger.info("retrieval task %s with %s:" % (test_subject, model_name))
    logger.info(
        "Average hit rate %.4f, %d ill-formatted generations"
        % (total_hit / all_samples, ill_format)
    )
    logger.info("Time cost %.4fs" % (time.time() - start_time))
    return total_hit / all_samples, ill_format, time.time() - start_time


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
    generate_dir = "/home/moon_shop/mmlu_data/retrieval"
    # model_name = "/ssd1/models/Meta-Llama-3-8B-Instruct"
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    res = []
    indicator2dist = {
        "hit_rate@3": [],
    }
    for dataset_name in os.listdir(generate_dir):
        dataset_name = dataset_name.split("dataset")[0].strip("_")
        args.test_subject = dataset_name
        score, ill_format, cost_time = cal_score(args, tokenizer, model, indicator2dist)
        res.append(
            {
                "dataset": dataset_name,
                "score": str(score),
                "ill_format": ill_format,
                "cost_time": str(cost_time),
                "model": model_name,
            }
        )
    indicator2dist = {
        k: {"average": np.mean(np.array(v)).item(), "std": np.std(np.array(v)).item()}
        for k, v in indicator2dist.items()
    }
    with open(
        f"{os.path.split(model_name)[-1]}_hit_rate@3_score_dist.json",
        "w",
        encoding="utf8",
    ) as f:
        json.dump(indicator2dist, f, ensure_ascii=False, indent=4)
    with open(
        f"{os.path.split(model_name)[-1]}_retrieval.json", "w", encoding="utf-8"
    ) as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    return res


if __name__ == "__main__":
    main()
