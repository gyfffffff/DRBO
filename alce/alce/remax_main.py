from tqdm import tqdm
import numpy as np
import os
import torch
from typing import Optional, List, Dict, Any
from argparse import Namespace
from loguru import logger
from remax_rlhf.rlhf_engine import RLHFEngine
from remax_rlhf.remax_trainer import ReMaxTrainer
from transformers import AutoTokenizer
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
from rouge_score import rouge_scorer
import evaluate
from sentence_transformers import SentenceTransformer
import copy
from alce.alce_dataset import ALCEDataset
from alce.eval import compute_len, compute_str_em, compute_rouge, compute_mauve, compute_autoais, compute_claims, \
    compute_qampari_f1, load_autoais
import json
from alce.args import parse_args


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
        self.r_every_step = {
            "fluency": [],
            "correctness": [],
            "citation": [],
            "rouge-l": [],
        }

        average_std_file = f"test_result_{self.args.model_name_or_path.split('/')[-1]}_average_std.json"
        average_std = json.load(open(average_std_file, "r"))
        self.mean = [v['average'] for k, v in average_std.items()]
        self.std = [v['std'] for k, v in average_std.items()]
        

    def init_remax_engine(self):
        
        engine = RLHFEngine(
            actor_model_name_or_path=self.args.model_name_or_path,
            reward_model_name_or_path=self.args.model_name_or_path,
            tokenizer=self.tokenizer,
            num_total_iters=100,
            args=self.args,
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
    
    def get_asqa_reward(
            self,
            result: Dict[str, Any],
            normalized_data: List[Dict[str, List | str]],
            data: List[Dict[str, List | str]],
            normalized_data_all: List[Dict[str, List | str]]
    ):
        result['str_em'], result['str_hit'] = compute_str_em(normalized_data)
        result['rougeLsum'] = compute_rouge(normalized_data)
        result['mauve'] = compute_mauve(normalized_data_all)
        result["citation"] = compute_autoais(data, qampari=self.args.dataset_name == 'qampari', at_most_citations=3)
        if self.args.test:
            return {
                'fluency': result['mauve'],
                'correctness': result['str_em'],
                'citation': 2 * (result["citation"]['citation_prec'] * result["citation"]['citation_rec']) / (
                        result["citation"]['citation_prec'] + result["citation"]['citation_rec']) if (result["citation"]['citation_prec'] + result["citation"]['citation_rec']) > 0 else 0,
                'rouge-l': result['rougeLsum']
            }
        else:
            return {
                'fluency': (result['mauve'] - self.mean[0])/self.std[0],
                'correctness': (result['str_em']-self.mean[1])/self.std[1],
                'citation': ((2 * (result["citation"]['citation_prec'] * result["citation"]['citation_rec']) / (
                        result["citation"]['citation_prec'] + result["citation"]['citation_rec']))-self.mean[2])/self.std[2] if (result["citation"]['citation_prec'] + result["citation"]['citation_rec']) > 0 else 0,
                'rouge-l': (result['rougeLsum']-self.mean[3])/self.std[3]
            }
    
    def get_eli5_reward(
            self,
            result: Dict[str, Any],
            normalized_data: List[Dict[str, List | str]],
            data: List[Dict[str, List | str]],
            normalized_data_all: List[Dict[str, List | str]]
    ):
        result["claims_nli"] = compute_claims(normalized_data)
        result['rougeLsum'] = compute_rouge(normalized_data)
        result['mauve'] = compute_mauve(normalized_data_all)
        result["citation"] = compute_autoais(data, qampari=self.args.dataset_name == 'qampari', at_most_citations=3)
        if self.args.test:
            return {
                'fluency': result['mauve'],
                'correctness': result["claims_nli"],
                'citation': 2 * (result["citation"]['citation_prec'] * result["citation"]['citation_rec']) / (
                        result["citation"]['citation_prec'] + result["citation"]['citation_rec']) if (result["citation"]['citation_prec'] + result["citation"]['citation_rec']) > 0 else 0,
                'rouge-l': result['rougeLsum']
            }
        else:
            return {
                'fluency': (result['mauve'] - self.mean[0])/self.std[0],
                'correctness': (result["claims_nli"]-self.mean[1])/self.std[1],
                'citation': ((2 * (result["citation"]['citation_prec'] * result["citation"]['citation_rec']) / (
                        result["citation"]['citation_prec'] + result["citation"]['citation_rec']))-self.mean[2])/self.std[2] if (result["citation"]['citation_prec'] + result["citation"]['citation_rec']) > 0 else 0,
                'rouge-l': (result['rougeLsum']-self.mean[3])/self.std[3]
            }

    def get_reward(
            self,
            data: Dict[str, List | str],
    ):
        normalized_data, data = self.preprocess_eval_data(data)
        rewards = []
        for i in range(len(data)):
            result = {'length': compute_len([normalized_data[i]])}
            if self.args.dataset_name == 'asqa':
                r = self.get_asqa_reward(result, [normalized_data[i]], [data[i]], normalized_data)
                for key, value in r.items():
                    self.r_every_step[key].append(value)
                rewards.append(r)
            if self.args.dataset_name == 'eli5':
                r = self.get_eli5_reward(result, [normalized_data[i]], [data[i]], normalized_data)
                for key, value in r.items():
                    self.r_every_step[key].append(value)
                rewards.append(r)
        # logger.info(f"rewards: {rewards}")
        return rewards, normalized_data

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
            do_sample: bool = True
    ):
        self.model.eval()

        max_tokens = self.args.max_new_tokens
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
        generation = self.tokenizer.batch_decode(outputs[:, prompt_ids.size(1):], skip_special_tokens=True)
        self.model.train()
        return generation, outputs


class MEAL:
    def __init__(
            self,
            args: Optional[Namespace]
    ):
        self.args = args
        self.print_args()
        self.set_seed()

        self.llm = LLM(self.args)
        self.prompt_data, self.eval_data = self.load_data()
        self.dataset = ALCEDataset(self.args, self.eval_data, self.prompt_data, self.llm.tokenizer)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.dataset.collate_fn
        )

        # ['fluency', 'correctness', 'citation', 'rouge-l']
        self.indicator = [
            'fluency',
            'correctness',
            'citation',
        ]

        average_std_file = f"test_result_{self.args.model_name_or_path.split('/')[-1]}_average_std.json"
        average_std = json.load(open(average_std_file, "r"))
        
        self.indicator_dist = average_std
        self.indicator_prob = [1 / len(self.indicator)] * len(self.indicator)
        self.indicator2prob = {indicator: 1 / len(self.indicator) for indicator in self.indicator}

    def print_args(self):
        for k in self.args.__dict__:
            logger.info(f"{k}: {self.args.__dict__[k]}")
    
    def load_data(self):
        prompt_data = json.load(open(self.args.prompt_path))
        eval_data = json.load(open(self.args.eval_path))
        return prompt_data, eval_data
    
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
            z_list = [np.exp((_x - self.args.tau * min__x) / (1 * step)).item() if _x != 0 else 0 for _x in prob_tmp_avg]
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
                loss, exp, actors_rewards = self.mab_step(batch)
                if step >= self.args.train_steps:
                    break
                if self.args.test:
                    continue
                for actors_reward in actors_rewards:
                    for k, v in actors_reward.items():
                        if k not in actors_rewards2list.keys():
                            actors_rewards2list[k] = []
                        actors_rewards2list[k].append(v)
                        total_actor_rewards[k] = total_actor_rewards.get(k, 0) + v
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
                restore_actor_rewards = [list(actor_rewards.values())[i]*self.llm.std[i]+self.llm.mean[i] for i in range(len(actor_rewards.values()))]
                logger.info(f"restore actor reward: {restore_actor_rewards}, average: {sum(restore_actor_rewards[:3])/3}")
                logger.info(f"indicator_prob: {self.indicator2prob}")
        return total_loss, step, actor_rewards

    def mab_train(self):
        for update_round in range(self.args.Epoch):
            logger.info("update_round: {}".format(update_round))
            total_loss, step, experience = self.mab_epoch(update_round)
            if self.args.test:
                test_result = {}
                for metric, rewards in self.llm.r_every_step.items():
                    test_result[metric] = {"average": np.mean(rewards), "std": np.std(rewards)}
                average_std_file = f"test_result_{self.args.model_name_or_path.split('/')[-1]}_average_std.json"
                with open(average_std_file, 'w') as f:
                    json.dump(test_result, f)
                return 

            logger.info(f"loss: {total_loss / step}")
            # self.llm.engine.ref = self.llm.engine.actor
            self.save(update_round)

    def mab_step(self, batch):
        self.llm.model.train()
        output_array, outputs = self.llm.generate(batch['prompt_ids'], batch['attention_mask'])
        baseline_output_array, _ = self.llm.generate(batch['prompt_ids'], batch['attention_mask'], do_sample=False)
        output_array = self.postprocess(output_array)
        baseline_output_array = self.postprocess(baseline_output_array)
        logger.info(f"output_array: {output_array}")
        reward_input = {'output': output_array}
        reward_input.update(batch)
        rewards, normalized_data = self.llm.get_reward(reward_input)
        
        if self.args.test:
            return 0, 0, 0
        
        baseline_reward_input = {'output': baseline_output_array}
        baseline_reward_input.update(batch)
        baseline_rewards, _ = self.llm.get_reward(baseline_reward_input)

        rewards_removed = [{k:v for k, v in reward.items() if k in self.indicator} for reward in rewards]
        baseline_rewards_removed = [{k:v for k, v in baseline_reward.items() if k in self.indicator} for baseline_reward in baseline_rewards]
        input_ids = []
        attention_mask = []
        for i in range(len(output_array)):
            messages = [
                {'role': 'system',
                'content': 'You are an intelligent assistant to help me answer questions based on context.'},
                {'role': 'user', 'content': batch['prompt'][i]},
                {'role': 'assistant', 'content': output_array[i]},
            ]
            input_ids_ = self.llm.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False) + [self.llm.tokenizer.eos_token_id]
            input_ids.append(input_ids_)
            attention_mask.append([1] * len(input_ids_))
        max_length = max([len(ii) for ii in input_ids])
        input_ids = [[self.llm.tokenizer.eos_token_id] * max(0, max_length-len(ii)) + ii for ii in input_ids]
        attention_mask = [[0] * max(0, max_length-len(ii)) + ii for ii in attention_mask]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.llm.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.llm.device)
        with torch.no_grad():
            output = self.llm.engine.actor(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            output_ref = self.llm.engine.ref(
                input_ids=input_ids,
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
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        ref_logprobs = log_softmax_values_ref.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        with torch.no_grad():
            output = self.llm.engine.actor(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            output_ref = self.llm.engine.ref(
                input_ids=input_ids,
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
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        ref_logprobs = log_softmax_values_ref.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        rl_rewards = []
        rl_baseline_rewards = []

        for i in range(len(rewards_removed)):  # reward_without_rouge: [{fluency:..., correct:... , citation:..}, {fluency:..., correct:... , citation:..}] 2 items (batchsize=2)
            reward_list = [(self.indicator2prob[key], value) for key, value in rewards_removed[i].items()]  # [(prob1, value1), (prob2, value2), (prob3, value3)]
            baseline_reward_list = [(self.indicator2prob[key], value) for key, value in baseline_rewards_removed[i].items()]
            rl_rewards.append(sum([value * indicator for indicator, value in reward_list]))
            rl_baseline_rewards.append(sum([value * indicator for indicator, value in baseline_reward_list]))
        logger.info(f"rl_rewards: {rl_rewards}")
        logger.info(f"rl_baseline_rewards: {rl_baseline_rewards}")
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
            "prompts": batch['prompt_ids'].to(self.llm.model.device),
            "logprobs": logprobs,
            "ref_logprobs": ref_logprobs,
            "rewards": rl_rewards,
            "baseline_rewards": rl_baseline_rewards,
            "full_kl": full_kl,
            "entropy": entropy,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        loss = self.llm.trainers.step(exp)
        return loss, exp, rewards 

    def save(
            self,
            update_round: int
    ):
        # Save the result
        model_name = self.args.model_name_or_path
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        # 保存模型的参数
        # save_dir = os.path.join(
        #     "outputs",
        #     f"{model_name}_weight_avg_{self.args.weight_avg}_delta_{self.args.delta}_epoch_{update_round}"
        # )
        # self.llm.model.save_pretrained(save_dir)
        # self.llm.tokenizer.save_pretrained(save_dir)


# @logger.catch
def main():
    # Load config
    args = parse_args()
    model_name = args.model_name_or_path
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    log_name = f"meal_{model_name}_weight_avg_{args.weight_avg}_delta_{args.delta}_{args.actor_learning_rate}_mab_d_{args.mab_d}.log"
    logger.add(log_name)
    meal = MEAL(args)
    meal.mab_train()


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
