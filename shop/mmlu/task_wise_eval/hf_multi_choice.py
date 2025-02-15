import pandas as pd
import time
import random
import transformers
from mmlu.task_wise_eval.utils import *
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import json
import numpy as np

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def cal_score(args, tokenizer, model, indicator2dist):
    seed = args.seed
    if seed != -1:
        random.seed(seed)
        transformers.set_seed(seed)

    model_name = args.model_name
    test_subject = args.test_subject
    print_interval = args.print_interval
    if not args.use_letter_choices:
        if "review_rating_prediction" not in test_subject:
            choices = ["0", "1", "2", "3"]
        else:
            choices = ["1", "2", "3", "4", "5"]
    else:
        choices = ["A", "B", "C", "D"]

    start_time = time.time()

    logger.info("Running %s model on %s task" % (model_name, test_subject))

    filename = f"/home/moon_shop/mmlu_data/multiple_choice/{test_subject}_dataset.csv"
    try:
        test_df = pd.read_csv(filename)
    except:
        raise FileNotFoundError(
            f"{filename} does not exist. Please check the 'test_subject' variable. "
        )
    correct = 0
    ill_format = 0
    all_samples = test_df.shape[0]

    samples = []
    for i in range(all_samples):
        few_shot_prompt = gen_system_prompt(args, is_multiple_choice=True)
        test_prompt = format_example(test_df, i, is_multi_choice=True, args=args)
        prompt = few_shot_prompt + test_prompt

        label = test_df.iloc[i, -1]
        if "review_rating_prediction" not in test_subject:
            label = choices[int(label)]

        if (
            "phi" in args.model_name or args.model_name == "ecellm-s"
        ) and not args.use_letter_choices:
            prompt += "\n(Please output a number only) Output: \n"
        if i % print_interval == 0:
            print("Sample %d" % i, prompt)
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
        inputs_ids = [bs["input_ids"] for bs in batch_samples]
        attention_mask = [bs["attention_mask"] for bs in batch_samples]
        labels = [bs["label"] for bs in batch_samples]
        if len(inputs_ids) == 0:
            continue
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
            or "zephyr" in args.model_name
            or "mixtral" in args.model_name
            or "ecellm-m" == args.model_name
        ):
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=4,
                temperature=0,
                pad_token_id=2,
            )
        elif "qwen" in args.model_name:
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=4,
                pad_token_id=151643,
                temperature=0.0001,
            )
        elif "llama" in args.model_name or args.model_name == "ecellm-l":
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=4,
                pad_token_id=151643,
                temperature=0.0001,
            )
        elif "phi" in args.model_name or args.model_name == "ecellm-s":
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=4,
                pad_token_id=50256,
                temperature=0.0001,
            )
        else:
            generate_ids = model.generate(
                inputs_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=4,
                temperature=0.0001,
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
            cur_label = labels[i]

            if not args.use_letter_choices:
                for k in answer:
                    if k.isnumeric():
                        answer = k
                        break
            else:
                for k in answer:
                    if k.isalnum():
                        answer = k
                        break
            if answer == str(cur_label):
                correct += 1
                indicator2dist["accuracy"].append(1)
            elif answer not in choices:
                print(f"ill format generation {answer}")
                ill_format += 1
                indicator2dist["accuracy"].append(0)
            else:
                indicator2dist["accuracy"].append(0)
            if sample_count % print_interval == 0:
                print(f"Sample {sample_count}, pred {answer}, label {cur_label}")
                print()

    logger.info(
        "%s model's accuracy on %s task is %.4f"
        % (model_name, test_subject, correct / all_samples)
    )
    logger.info(
        "There are %d ill-formatted examples out of %d" % (ill_format, all_samples)
    )
    logger.info("Time Cost: %.4fs" % (time.time() - start_time))
    return correct / all_samples, ill_format, all_samples, time.time() - start_time


class Args:
    test_subject: str = ""
    model_name: str = ""
    use_task_specific_prompt: bool = True
    print_interval: int = 20
    seed: int = -1
    use_letter_choices: bool = True
    batch_size: int = 16


def main(model_name="/ssd1/models/Llama-3.1-8B-Instruct", model=None, tokenizer=None):
    args = Args()
    generate_dir = "/home/moon_shop/mmlu_data/multiple_choice"
    # model_name = "/ssd1/models/Meta-Llama-3-8B-Instruct"
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    res = []
    indicator2dist = {
        "accuracy": [],
    }
    for dataset_name in os.listdir(generate_dir):
        dataset_name = dataset_name.split("dataset")[0].strip("_")
        args.test_subject = dataset_name
        score, ill_format, all_samples, cost_time = cal_score(
            args, tokenizer, model, indicator2dist
        )
        res.append(
            {
                "dataset": dataset_name,
                "score": str(score),
                "ill_format": ill_format,
                "all_samples": all_samples,
                "cost_time": str(cost_time),
                "model": model_name,
            }
        )

    indicator2dist = {
        k: {"average": np.mean(np.array(v)).item(), "std": np.std(np.array(v)).item()}
        for k, v in indicator2dist.items()
    }
    with open(
        f"{os.path.split(model_name)[-1]}_multi_choice_score_dist.json",
        "w",
        encoding="utf8",
    ) as f:
        json.dump(indicator2dist, f, ensure_ascii=False, indent=4)
    with open(
        f"{os.path.split(model_name)[-1]}_multi_choice.json", "w", encoding="utf-8"
    ) as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    return res


if __name__ == "__main__":
    main()
