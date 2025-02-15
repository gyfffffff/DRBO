from torch.utils.data import Dataset
import os
import pandas as pd
from mmlu.task_wise_eval.utils import gen_system_prompt, format_example
import torch


class MMLUDataset(Dataset):
    def __init__(self, args, tokenizer, samples_num=1000000):
        self.tokenizer = tokenizer
        self.args = args
        self.samples = []
        for task_dir in os.listdir(self.args.mmlu_data_dir):
            task_name = task_dir
            if task_name == "skills":
                continue
            task_dir = os.path.join(self.args.mmlu_data_dir, task_name)
            sample_count = 0
            for dataset_file in os.listdir(task_dir):
                dataset_file_path = os.path.join(task_dir, dataset_file)
                dataset_name = dataset_file.replace('_dataset.json', '').replace('_dataset.csv', '')
                args.test_subject = dataset_name
                if dataset_file.endswith(".csv"):
                    df = pd.read_csv(dataset_file_path)
                    if not args.use_letter_choices:
                        if 'review_rating_prediction' not in dataset_name:
                            choices = ['0', '1', '2', '3']
                        else:
                            choices = ['1', '2', '3', '4', '5']
                    else:
                        choices = ['A', 'B', 'C', 'D']
                    for i in range(df.shape[0]):
                        metric = 'accuracy'
                        few_shot_prompt = gen_system_prompt(args, is_multiple_choice=True)
                        test_prompt = format_example(df, i, is_multi_choice=True, args=args)
                        prompt = few_shot_prompt + test_prompt

                        label = df.iloc[i, -1]
                        if 'review_rating_prediction' not in dataset_name:
                            label = choices[int(label)]

                        target = label
                        sample = {
                            "metric": metric,
                            "task_name": task_name,
                            "dataset_name": dataset_name,
                            "prompt": prompt,
                            "target": target
                        }
                        self.samples.append(sample)
                        sample_count += 1
                        if sample_count >= samples_num:
                            break

                elif dataset_file.endswith(".json"):
                    df = pd.read_json(dataset_file_path, lines=True)
                    for i in range(df.shape[0]):
                        if task_name == "generation":
                            if 'extraction' in args.test_subject:
                                metric = 'rougel'
                            elif 'translation' in args.test_subject:
                                metric = 'bleu'
                            elif 'multilingual' in args.test_subject:
                                metric = 'multilingual-sent-transformer'
                            else:
                                metric = 'sent-transformer'
                            train_prompt = gen_system_prompt(args)
                            test_prompt = format_example(df, i)
                            prompt = train_prompt + test_prompt
                            target = df.iloc[i, -1]
                            sample = {
                                "metric": metric,
                                "task_name": task_name,
                                "dataset_name": dataset_name,
                                "prompt": prompt,
                                "target": target
                            }
                            self.samples.append(sample)
                        elif task_name == "retrieval":
                            metric = 'hit rate@3'
                            system_prompt = gen_system_prompt(args)
                            test_prompt = format_example(df, i)
                            prompt = system_prompt + test_prompt
                            target = df.iloc[i]['target_field']
                            sample = {
                                "metric": metric,
                                "task_name": task_name,
                                "dataset_name": dataset_name,
                                "prompt": prompt,
                                "target": target
                            }
                            self.samples.append(sample)
                        elif task_name == 'ranking':
                            metric = 'ndcg'
                            train_prompt = gen_system_prompt(args)
                            test_prompt = format_example(df, i)
                            prompt = train_prompt + test_prompt
                            target = df.iloc[i]['target_field']
                            sample = {
                                "metric": metric,
                                "task_name": task_name,
                                "dataset_name": dataset_name,
                                "prompt": prompt,
                                "target": target
                            }
                            self.samples.append(sample)
                        elif task_name == 'named_entity_recognition':
                            metric = 'micro f1'
                            train_prompt = gen_system_prompt(args)
                            test_prompt = format_example(df, i)
                            prompt = train_prompt + test_prompt

                            label = df.iloc[i, -1]
                            target = []
                            for l_ in label:
                                target.append(l_.lower())
                            sample = {
                                "metric": metric,
                                "task_name": task_name,
                                "dataset_name": dataset_name,
                                "prompt": prompt,
                                "target": target
                            }
                            self.samples.append(sample)
                        sample_count += 1
                        if sample_count >= samples_num:
                            break

        # self.samples = [sample for sample in self.samples if sample['metric'] == 'rougel']
        self.length = len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        prompt = sample['prompt']
        inputs = self.tokenizer(prompt)
        sample.update({
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask
        })
        return sample

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        metrics = []
        task_name = []
        dataset_name = []
        prompt = []
        target = []
        input_ids = []
        attention_mask = []
        for b_ in batch:
            metrics.append(b_['metric'])
            task_name.append(b_['task_name'])
            dataset_name.append(b_['dataset_name'])
            prompt.append(b_['prompt'])
            target.append(b_["target"])
            input_ids.append(b_['input_ids'])
            attention_mask.append(b_['attention_mask'])

        max_length = max([len(ii) for ii in input_ids])
        input_ids = [[self.tokenizer.eos_token_id]*max(0, max_length-len(ii)) + ii for ii in input_ids]
        attention_mask = [[0]*max(0, max_length-len(ii)) + ii for ii in attention_mask]
        return {
            "metrics": metrics,
            "task_name": task_name,
            "dataset_name": dataset_name,
            "prompt": prompt,
            "target": target,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }


def main():
    pass


if __name__ == "__main__":
    main()
