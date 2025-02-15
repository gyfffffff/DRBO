import pandas as pd
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from mmlu.task_wise_eval.utils import *
from loguru import logger
import numpy as np


def cal_score(args, tokenizer, model, indicator2dist):
    start_time = time.time()
    test_subject = args.test_subject
    model_name = args.model_name
    logger.info("Running %s on %s task." % (model_name, test_subject))
    gen_result = []

    filename = f"/home/moon_shop/mmlu_data/named_entity_recognition/{test_subject}_dataset.json"
    try:
        test_df = pd.read_json(filename, lines=True)
    except:
        raise FileNotFoundError(
            f"{filename} does not exist. Please modify the 'test_subject' argument. "
        )
    all_samples = test_df.shape[0]

    true_positive = 0
    false_positive = 0
    false_negative = 0
    samples = []
    for i in range(all_samples):
        train_prompt = gen_system_prompt(args)
        test_prompt = format_example(test_df, i)
        prompt = train_prompt + test_prompt
        if i % args.print_interval == 0:
            print("Sample %d prompt" % i, prompt)
        label = test_df.iloc[i, -1]
        label_lower = []
        for l in label:
            label_lower.append(l.lower())
        inputs = tokenizer(prompt)

        samples.append(
            {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "label": label_lower,
            }
        )
    sample_count = 0
    for j in range(0, len(samples) // args.batch_size + 1):
        start_idx = j * args.batch_size
        end_idx = min((j + 1) * args.batch_size, len(samples))
        batch_samples = samples[start_idx:end_idx]
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
        if "mistral" not in model_name and "mixtral" not in model_name:
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_gen_len,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=2,
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
            answer = cur_res[len(prompt) :]  # .split('\n')[0]
            cur_label = labels[i]  # .split('\n')[0]
            answer = answer.lstrip("\n")
            if sample_count % args.print_interval == 0:
                print("Sample %d answer:" % sample_count, answer)
                print("Sample %d ground truth:" % sample_count, cur_label)
            answer = answer.split("\n")[0].lstrip(" ").rstrip(" ")
            answer = answer.split(",")
            answer = [a for a in answer if a != ""]
            answer_lower = []
            for a in answer:
                answer_lower.append(a.lower().lstrip(" ").rstrip(" "))

            if sample_count % args.print_interval == 0:
                print("Sample %d answer" % sample_count, answer_lower)

            tp = len(set(answer_lower).intersection(set(cur_label)))
            fp = len(answer_lower) - len(set(answer_lower).intersection(set(cur_label)))
            fn = len(cur_label) - len(set(answer_lower).intersection(set(cur_label)))
            p_ = tp / (tp + fp) if tp + fp != 0 else 0
            r_ = tp / (tp + fn) if tp + fn != 0 else 0
            f1_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ != 0 else 0
            indicator2dist["micro f1"].append(f1_)
            gen_result.append(
                {
                    "prompt": prompt,
                    "answer": answer_lower,
                    "label": cur_label,
                    "eval": {"tp": tp, "fp": fp, "fn": fn},
                }
            )
            true_positive += tp
            false_positive += fp
            false_negative += fn
            if sample_count % args.print_interval == 0:
                print(
                    f"Sample {sample_count}, TP {len(set(answer_lower).intersection(set(cur_label)))}, FP {len(answer_lower) - len(set(answer_lower).intersection(set(cur_label)))}, FN {len(cur_label) - len(set(answer_lower).intersection(set(cur_label)))}"
                )
                print()

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    logger.info("Time Cost: %.4fs" % (time.time() - start_time))

    logger.info(
        "The average F1 score of %s on %s task is %.4f, precision %.4f, recall %.4f"
        % (args.model_name, test_subject, f1, precision, recall)
    )
    return f1, precision, recall, time.time() - start_time, gen_result


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
    generate_dir = "/home/moon_shop/mmlu_data/named_entity_recognition"
    # model_name = "/ssd1/models/Meta-Llama-3-8B-Instruct"
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    res = []
    indicator2dist = {
        "micro f1": [],
    }
    all_gen_res = []
    for dataset_name in os.listdir(generate_dir):
        dataset_name = dataset_name.split("dataset")[0].strip("_")
        args.test_subject = dataset_name
        f1, precision, recall, cost_time, gen_result = cal_score(
            args, tokenizer, model, indicator2dist
        )
        res.append(
            {
                "dataset": dataset_name,
                "f1": str(f1),
                "precision": str(precision),
                "recall": str(recall),
                "cost_time": str(cost_time),
                "model": model_name,
            }
        )
        all_gen_res.extend(gen_result)
    indicator2dist = {
        k: {"average": np.mean(np.array(v)).item(), "std": np.std(np.array(v)).item()}
        for k, v in indicator2dist.items()
    }
    with open(
        f"{os.path.split(model_name)[-1]}_micro_f1_score_dist.json",
        "w",
        encoding="utf8",
    ) as f:
        json.dump(indicator2dist, f, ensure_ascii=False, indent=4)
    with open(f"{os.path.split(model_name)[-1]}_ner.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    with open(
        f"{os.path.split(model_name)[-1]}_ner_gen_result.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_gen_res, f, ensure_ascii=False, indent=4)
    return res


if __name__ == "__main__":
    main()
