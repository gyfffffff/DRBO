from tqdm import tqdm
import numpy as np
import os
import torch
from local_evaluation import aggregate_scores
from shop.args import parse_args
from typing import Optional, List, Dict
from argparse import Namespace
from loguru import logger
from remax_rlhf.rlhf_engine import RLHFEngine
from remax_rlhf.remax_trainer import ReMaxTrainer
from transformers import AutoTokenizer
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
from mmlu.task_wise_eval.hf_ranking import is_permutation, ndcg
from rouge_score import rouge_scorer
import evaluate
from sentence_transformers import SentenceTransformer
import copy
from mmlu.mmlu_dataset import MMLUDataset


class LLM:
    def __init__(
            self,
            args: Optional[Namespace],
    ):
        self.args = args
        self.device = self.args.device
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, fast_tokenizer=True
        )
        self.args.end_of_conversation_token = self.tokenizer.eos_token

        # 初始化PPO优化器
        self.engine = self.init_remax_engine()
        self.trainers = self.init_trainer()
        self.model = self.engine.actor

        self.metric2func = {
            "rougel": rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True),
            "bleu": evaluate.load('sacrebleu'),
            'multilingual-sent-transformer': SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').cuda(),
            'sent-transformer': SentenceTransformer('all-MiniLM-L6-v2').cuda()
        }

    def init_remax_engine(self):
        llama3_2_1b_device_map = {
            'model.embed_tokens': 0,
            'lm_head': 0,
            'model.layers.0': 1,
            'model.layers.1': 1,
            'model.layers.2': 1,
            'model.layers.3': 1,
            'model.layers.4': 1,
            'model.layers.5': 1,
            'model.layers.6': 2,
            'model.layers.7': 2,
            'model.layers.8': 2,
            'model.layers.9': 2,
            'model.layers.10': 2,
            'model.layers.11': 3,
            'model.layers.12': 3,
            'model.layers.13': 3,
            'model.layers.14': 3,
            'model.layers.15': 3,
            'model.norm': 3,
            'model.rotary_emb': 3
        }

        qwen2_5_1b_device_map = {
            'model.embed_tokens': 0,
            'lm_head': 0,
            'model.layers.0': 1,
            'model.layers.1': 1,
            'model.layers.2': 1,
            'model.layers.3': 1,
            'model.layers.4': 1,
            'model.layers.5': 1,
            'model.layers.6': 1,
            'model.layers.7': 1,
            'model.layers.8': 1,
            'model.layers.9': 1,
            'model.layers.10': 2,
            'model.layers.11': 2,
            'model.layers.12': 2,
            'model.layers.13': 2,
            'model.layers.14': 2,
            'model.layers.15': 2,
            'model.layers.16': 2,
            'model.layers.17': 2,
            'model.layers.18': 2,
            'model.layers.19': 3,
            'model.layers.20': 3,
            'model.layers.21': 3,
            'model.layers.22': 3,
            'model.layers.23': 3,
            'model.layers.24': 3,
            'model.layers.25': 3,
            'model.layers.26': 3,
            'model.layers.27': 3,
            'model.norm': 3
        }

        engine = RLHFEngine(
            actor_model_name_or_path=self.args.model_name_or_path,
            reward_model_name_or_path=self.args.model_name_or_path,
            tokenizer=self.tokenizer,
            num_total_iters=6000,
            args=self.args,
            device_map="auto"
        )

        return engine

    def init_trainer(self):
        return ReMaxTrainer(self.engine, self.args)

    def preprocess_eval_data(
            self,
            data: Dict[str, List | str],
    ):
        new_data = [dict() for _ in list(data.values())[0]]
        for key, value in data.items():
            for i in range(len(value)):
                new_data[i][key] = value[i]
        normalized_data = copy.deepcopy(new_data)

        return normalized_data, new_data

    def get_reward(
            self,
            data: Dict[str, List | str],
    ):
        normalized_data, data = self.preprocess_eval_data(data)
        rewards = {}
        for i in range(len(data)):
            output = data[i]['output']
            label = data[i]['target']
            metric = data[i]['metrics']
            if metric == "accuracy":
                answer = None
                for k in output:
                    if k.isalnum():
                        answer = k
                        break
                reward = 1 if answer == label else 0
            elif metric == "micro f1":
                answer = output.lstrip('\n')
                answer = answer.split('\n')[0].lstrip(' ').rstrip(' ')
                answer = answer.split(',')
                answer = [a for a in answer if a != '']
                answer_lower = []
                for a in answer:
                    answer_lower.append(a.lower().lstrip(' ').rstrip(' '))
                cur_label = []
                for l in label:
                    cur_label.append(l.lower())

                tp = len(set(answer_lower).intersection(set(cur_label)))
                fp = len(answer_lower) - len(set(answer_lower).intersection(set(cur_label)))
                fn = len(cur_label) - len(set(answer_lower).intersection(set(cur_label)))
                reward = (tp, fp, fn)
            elif metric == "ndcg":
                reward = 0
                answer = output
                if '\n' in answer:
                    answer = answer.lstrip('\n').rstrip('\n')
                    answer = answer.split('\n')[0]
                ranked_list_str = answer.lstrip().rstrip().split(',')
                ranked_list = []

                for x in ranked_list_str:
                    try:
                        ranked_list.append(int(x))
                    except:
                        reward = reward

                if len(ranked_list) != len(label):
                    reward = reward
                elif not is_permutation(ranked_list):
                    reward = reward
                else:
                    reward = ndcg(ranked_list, label)
            elif metric == "hit rate@3":
                retrieved_list = output
                retrieved_list = retrieved_list.lstrip().rstrip()
                if '\n' not in retrieved_list:
                    retrieved_list = retrieved_list.split(',')
                else:
                    retrieved_list = retrieved_list.split('\n')[0]
                    retrieved_list = retrieved_list.split(',')
                retrieved_int = []
                for ret in retrieved_list:
                    try:
                        ret_int = int(ret)
                        retrieved_int.append(ret_int)
                    except:
                        continue
                if len(retrieved_int) > 3:
                    retrieved_int = retrieved_int[:3]

                hit = len(set(label).intersection(set(retrieved_int)))
                hit /= len(label)
                reward = hit
            else:
                if metric == 'rougel':
                    reward = self.metric2func[metric].score(output, label)['rougeL'].fmeasure
                elif metric in ['sent-transformer', 'multilingual-sent-transformer']:
                    if isinstance(label, str):
                        truth_embedding = self.metric2func[metric].encode([label])[0]
                        generation_embedding = self.metric2func[metric].encode([output])[0]
                        current_score = ((generation_embedding * truth_embedding).sum())
                        current_score /= (np.linalg.norm(generation_embedding, ord=2) * np.linalg.norm(truth_embedding, ord=2))
                    else:
                        scores = []
                        generation_embedding = self.metric2func[metric].encode([output])[0]
                        for label_item in label:
                            truth_embedding = self.metric2func[metric].encode([label_item])[0]
                            score_ = (generation_embedding * truth_embedding).sum()
                            score_ /= (np.linalg.norm(generation_embedding, ord=2) * np.linalg.norm(truth_embedding, ord=2))
                            scores.append(score_)
                        current_score = np.mean(scores)
                    reward = current_score
                elif metric == 'bleu':
                    # usage: sacrebleu.compute(predictions=xxx, references=yyy)
                    # reference can be multiple lists of sentences
                    # candidate is a list of sentences
                    generation = output.lstrip('\n').rstrip('\n').split('\n')[0]
                    candidate = [generation]
                    reference = [[label]]

                    if 'JP' not in data[i]['dataset_name']:
                        # japanese needs a different tokenizer
                        tokenize = '13a'
                    else:
                        tokenize = 'ja-mecab'
                    reward = \
                        self.metric2func[metric].compute(
                            predictions=candidate, references=reference, lowercase=True, tokenize=tokenize
                        )['score'] / 100
                else:
                    reward = 0
                    pass
            if metric not in rewards.keys():
                rewards[metric] = {
                    "task_name": data[i]['task_name'],
                    'task_type': data[i]['dataset_name'],
                    'metric': metric,
                    'sample_score': []
                }
            rewards[metric]['sample_score'].append(reward)

        rewards = aggregate_scores(rewards)
        rewards_list = []
        for _, row in rewards.iterrows():
            rewards_list.append({row['task_name']: row['overall_score']})
        return rewards_list, normalized_data

    # def get_reward(self):
    #     gen_res = hf_generation(model=self.engine.actor, tokenizer=self.engine.tokenizer)
    #     multi_choice_res = hf_multi_choice(model=self.engine.actor, tokenizer=self.engine.tokenizer)
    #     ranking_res = hf_ranking(model=self.engine.actor, tokenizer=self.engine.tokenizer)
    #     ner_res = hf_ner(model=self.engine.actor, tokenizer=self.engine.tokenizer)
    #     retrieval_res = hf_retrieval(model=self.engine.actor, tokenizer=self.engine.tokenizer)

    @torch.no_grad()
    def generate(
            self,
            prompt_ids: Tensor,
            attention_mask: Tensor,
            task_name: List[str],
            do_sample: bool = True
    ):
        self.model.eval()
        task_name2max_length = {
            "generation": 128,
            "multiple_choice": 4,
            "ranking": 20,
            "named_entity_recognition": 20,
            "retrieval": 20
        }
        max_tokens = max([task_name2max_length[tn] for tn in task_name])
        if do_sample:
            # TODO 要不要试试开beam search
            outputs = self.model.generate(
                input_ids=prompt_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                do_sample=do_sample,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        else:
            outputs = self.model.generate(
                input_ids=prompt_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                do_sample=do_sample,
                # top_p=self.args.top_p,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generations = []
        for i, output in enumerate(outputs):
            generation = self.tokenizer.decode(output[prompt_ids[i].shape[0]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generations.append(generation)
        self.model.train()
        return generations, outputs


class MEAL:
    def __init__(
            self,
            args: Optional[Namespace]
    ):
        self.args = args
        self.print_args()
        self.set_seed()

        self.llm = LLM(self.args)
        self.dataset = MMLUDataset(self.args, self.llm.tokenizer)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.dataset.collate_fn
        )

        # ['fluency', 'correctness', 'citation', 'rouge-l']
        self.indicator = [
            'micro f1',
            'hit rate@3',
            'accuracy',
            'ndcg',
            'sent-transformer',
            'bleu',
            'multilingual-sent-transformer',
            'rougel'
        ]
        llama3_2_3b_average_std = {
            'micro f1': {"average": 0.5761534046064433, "std": 0.46761944608969724},
            'hit rate@3': {"average": 0.5877243775332948, "std": 0.30184593800649673},
            'accuracy': {"average": 0.7264183068855139, "std": 0.44579675896881227},
            'ndcg': {"average": 0.5877243775332948, "std": 0.30184593800649673},
            'sent-transformer': {"average": 0.5200443267822266, "std": 0.18382175266742706},
            'bleu': {"average": 0.2297947167211648, "std": 0.16395431665250093},
            'multilingual-sent-transformer': {"average": 0.47872108221054077, "std": 0.16112563014030457},
            'rougel': {"average": 0.07895670208488156, "std": 0.1459379144167195},
        }
        llama3_2_1b_average_std = {
            'micro f1': {"average": 0.39717000310922956, "std": 0.4216519522495816},
            'hit rate@3': {"average": 0.15962169465354176, "std": 0.21236598130371825},
            'accuracy': {"average": 0.5512497446026017, "std": 0.4973665285060587},
            'ndcg': {"average": 0.5795919306692744, "std": 0.27809300986537844},
            'sent-transformer': {"average": 0.5005560517311096, "std": 0.18757575750350952},
            'bleu': {"average": 0.190903893450951, "std": 0.15292619626120557},
            'multilingual-sent-transformer': {"average": 0.43968796730041504, "std": 0.1926022320985794},
            'rougel': {"average": 0.13503407738595993, "std": 0.16806214267674677},
        }
        qwen_2_5_1b_average_std = {
            'micro f1': {"average": 0.11390423572744013, "std": 0.3054247302438866},
            'hit rate@3': {"average": 0.38641188959660294, "std": 0.27637349108443954},
            'accuracy': {"average": 0.5605121569161615, "std": 0.496324771561277},
            'ndcg': {"average": 0.7656525738816127, "std": 0.20725838334677982},
            'sent-transformer': {"average": 0.5113519430160522, "std": 0.1878712773323059},
            'bleu': {"average": 0.15693137404835045, "std": 0.15425797838753938},
            'multilingual-sent-transformer': {"average": 0.45012176036834717, "std": 0.1450491100549698},
            'rougel': {"average": 0.030311197219718554, "std": 0.026388036233325148},
        }
        self.indicator_dist = qwen_2_5_1b_average_std
        self.indicator_prob = [1 / len(self.indicator)] * len(self.indicator)
        self.indicator2prob = {indicator: 1 / len(self.indicator) for indicator in self.indicator}

    def print_args(self):
        for k in self.args.__dict__:
            logger.info(f"{k}: {self.args.__dict__[k]}")

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    @staticmethod
    def postprocess(
            output_array: Optional[List[str]]
    ):
        new_out_array = []
        for oa in output_array:
            if '\nsystem\n' in oa:
                new_out_array.append(oa[:oa.index('\nsystem\n')])
            elif "\nAssistant" in oa:
                new_out_array.append(oa[:oa.index("\nAssistant")])
            elif "Assistant:" in oa:
                new_out_array.append(oa[:oa.index("Assistant:")])
            else:
                new_out_array.append(oa)
        return new_out_array

    def update_prob(
            self,
            experience: Dict[str, float],
            total_sample: int,
            indic_sample_dict: Dict[str, float],
            pre_actor_rewards: Dict[str, float],
            step: int
    ):
        if self.args.simple_baseline1:
            prob_tmp = [0 for _ in self.indicator]
            for i, indic in enumerate(self.indicator):
                if indic not in experience:
                    continue
                prob_tmp[i] = 1/experience[indic] if experience[indic]!=0 else 1/1e-5
            z_list = prob_tmp
            self.indicator_prob = [_z / sum(z_list) if _z != 0 else self.indicator_prob[i] for i, _z in enumerate(z_list)]
            self.indicator2prob = {indicator: self.indicator_prob[i] if indicator in experience else 0 for i, indicator in enumerate(self.indicator)}
        else:
            prob_tmp = [0 for _ in self.indicator]
            prob_tmp_avg = [0 for _ in self.indicator]
            for key in experience.keys():
                indic_sample_dict[key] = indic_sample_dict.get(key, 0) + 1
            t = len(experience)
            for i, indic in enumerate(self.indicator):
                if indic not in experience.keys():
                    continue
                if self.args.delta:
                    prob_tmp_avg[i] = (experience[indic] - pre_actor_rewards[indic]) / t
                else:
                    prob_tmp_avg[i] = (experience[indic] + 1) / t
                # 公式二
                prob_tmp[i] = prob_tmp_avg[i] + np.sqrt((2 * np.log(total_sample)) / indic_sample_dict[indic]).item()
            # 公式三
            min__x = min(prob_tmp_avg)
            # 公式四

            if self.args.delta:
                z_list = [np.exp((_x - self.args.tau * min__x) / (0.01 * step)).item() if _x != 0 else 0 for _x in prob_tmp_avg]
            else:
                z_list = [1 / (_x - self.args.tau * min__x) if _x != 0 else 0 for _x in prob_tmp]
            # logger.info(f"z_list: {z_list}")
            self.indicator_prob = [_z / sum(z_list) if _z != 0 else self.indicator_prob[i] for i, _z in enumerate(z_list)]
            self.indicator2prob = {indicator: self.indicator_prob[i] for i, indicator in enumerate(self.indicator)}
        logger.info(f"indicator_prob: {self.indicator2prob}")

    def mab_epoch(
            self,
            update_round: int
    ):

        step = 0
        total_loss = 0
        experience = []
        total_actor_rewards = {}
        total_actor_count = {}
        normalized_data_list = []
        total_sample = 0
        rewards_count = 0
        actors_rewards2list = {}
        indic_sample_dict = dict()
        with tqdm(
                iterable=self.dataloader, desc=f"mab training: {update_round}", unit_scale=True
        ) as pbar:
            for batch in pbar:
                pre_actor_rewards = copy.deepcopy(total_actor_rewards)
                step += 1
                total_sample += len(batch)
                loss, exp, actors_rewards, normalized_data = self.mab_step(batch)

                normalized_data_list.extend(normalized_data)

                for actors_reward in actors_rewards:
                    for k, v in actors_reward.items():
                        if k not in actors_rewards2list.keys():
                            actors_rewards2list[k] = []
                        actors_rewards2list[k].append(v)
                        total_actor_rewards[k] = total_actor_rewards.get(k, 0) + (v- self.indicator_dist[k]['average']) / self.indicator_dist[k]['std']
                        total_actor_count[k] = total_actor_count.get(k, 0) + 1
                    rewards_count += 1
                experience.append(exp)
                loss.backward()
                self.llm.engine.optim.step()
                self.llm.engine.lr_scheduler.step()
                self.llm.engine.optim.zero_grad()
                total_loss += loss.item()
                if not self.args.weight_avg:
                    if step % self.args.mab_d == 0:
                        self.update_prob(
                            {k: v / rewards_count for k, v in total_actor_rewards.items()},
                            total_sample,
                            indic_sample_dict,
                            pre_actor_rewards,
                            step
                        )
                # if step % 100 == 0:
                #     logger.info("actor reward: {}".format({k: v / total_actor_count[k] for k, v in total_actor_rewards.items()}))
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.5f}",
                    }
                )
        actor_rewards = {k: v / total_actor_count[k] for k, v in total_actor_rewards.items()}
        # self.indicator_dist = {
        #     indicator: {"average": np.mean(rewards_list).item(), "std": np.std(rewards_list).item()}
        #     for indicator, rewards_list in actors_rewards2list.items()
        # }
        logger.info("actor reward: {}".format(actor_rewards))
        logger.info(f"indicator_prob: {self.indicator2prob}")
        return total_loss, step, actor_rewards

    def mab_train(self):
        for update_round in range(self.args.Epoch):
            logger.info("update_round: {}".format(update_round))
            total_loss, step, experience = self.mab_epoch(update_round)

            logger.info(f"loss: {total_loss / step}")
            # self.llm.engine.ref = self.llm.engine.actor
            self.save(update_round)

    def mab_step(self, batch):
        self.llm.model.train()
        output_array, outputs = self.llm.generate(
            batch['input_ids'],
            batch['attention_mask'],
            batch['task_name']
        )
        baseline_output_array, _ = self.llm.generate(
            batch['input_ids'],
            batch['attention_mask'],
            batch['task_name'],
            do_sample=False
        )

        output_array = self.postprocess(output_array)
        baseline_output_array = self.postprocess(baseline_output_array)

        reward_input = {'output': output_array}
        reward_input.update(batch)

        baseline_reward_input = {'output': baseline_output_array}
        baseline_reward_input.update(batch)
        # logger.info(output_array)
        # 获得输出
        rewards, normalized_data = self.llm.get_reward(reward_input)
        baseline_rewards, _ = self.llm.get_reward(baseline_reward_input)

        attention_mask = torch.ones_like(outputs, dtype=torch.long, device=self.llm.device)
        with torch.no_grad():
            output = self.llm.engine.actor(
                input_ids=outputs,
                attention_mask=attention_mask,
            )
            output_ref = self.llm.engine.ref(
                input_ids=outputs,
                attention_mask=attention_mask,
            )

        logits = output.logits
        logits_ref = output_ref.logits

        log_softmax_values = F.log_softmax(logits, dim=-1)
        softmax_probs = torch.exp(log_softmax_values)
        entropy = -torch.sum(softmax_probs * log_softmax_values, dim=-1)

        log_softmax_values_ref = F.log_softmax(logits_ref, dim=-1)
        full_kl = torch.sum(
            softmax_probs * (log_softmax_values - log_softmax_values_ref), dim=-1
        )

        logprobs = log_softmax_values.gather(
            dim=-1, index=outputs[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        ref_logprobs = log_softmax_values_ref.gather(
            dim=-1, index=outputs[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        rl_rewards = []
        rl_baseline_rewards = []
        for i in range(batch["input_ids"].shape[0]):
            reward_ = sum(
                [
                    sum(
                        [
                            (
                                    score - self.indicator_dist[indicator]['average']
                            ) * self.indicator2prob[indicator] / self.indicator_dist[indicator]['std']
                            for indicator, score in indicator_info.items()
                        ]
                    )
                    for indicator_info in rewards if batch['metrics'][i] in indicator_info.keys()
                ]
            )
            baseline_reward_ = sum(
                [
                    sum(
                        [
                            (
                                    score - self.indicator_dist[indicator]['average']
                            ) * self.indicator2prob[indicator] / self.indicator_dist[indicator]['std']
                            for indicator, score in indicator_info.items()
                        ]
                    )
                    for indicator_info in baseline_rewards if batch['metrics'][i] in indicator_info.keys()
                ]
            )

            rl_rewards.append(reward_)
            rl_baseline_rewards.append(baseline_reward_)
        rl_rewards = torch.tensor(
            rl_rewards,
            dtype=torch.float32,
            device=self.llm.device
        )
        rl_baseline_rewards = torch.tensor(
            rl_baseline_rewards,
            dtype=torch.float32,
            device=self.llm.device
        )
        exp = {
            "prompts": batch['input_ids'].to(self.llm.model.device),
            "logprobs": logprobs,
            "ref_logprobs": ref_logprobs,
            "rewards": rl_rewards,
            "baseline_rewards": rl_baseline_rewards,
            "full_kl": full_kl,
            "entropy": entropy,
            "input_ids": outputs,
            "attention_mask": attention_mask,
        }
        loss = self.llm.trainers.step(exp)
        return loss, exp, rewards, normalized_data

    def save(
            self,
            update_round: int
    ):
        # Save the result
        model_name = self.args.model_name_or_path
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        # 保存模型的参数
        save_dir = os.path.join(
            "/shared/ssd/models",
            f"{model_name}_weight_avg_{self.args.weight_avg}_delta_{self.args.delta}_epoch_{update_round}_simple_baseline1_{self.args.simple_baseline1}"
        )
        self.llm.model.save_pretrained(save_dir)
        self.llm.tokenizer.save_pretrained(save_dir)


# @logger.catch
def main():
    # Load config
    args = parse_args()
    logger.add('meal.log')
    meal = MEAL(args)
    meal.mab_train()


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
