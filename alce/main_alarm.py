'''
Author: NuoJohnChen chennuo@cuhk.edu.cn
Date: 2024-06-07 11:32:15
LastEditors: jijivski 2281255574@qq.com
LastEditTime: 2024-07-24 13:14:38
FilePath: /DP_tuning/mntcephfs/lab_data/chennuo/MOON/remax_main_02.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tqdm import tqdm
import numpy as np
import os
import torch
import json
import copy
from eval import compute_len, compute_str_em, compute_rouge, compute_mauve, compute_autoais, compute_claims, \
    compute_qampari_f1, load_autoais
from args import parse_args
from typing import Optional, List, Dict, Any
from argparse import Namespace
from loguru import logger
from remax_rlhf.rlhf_engine import RLHFEngine
from remax_rlhf.remax_trainer import ReMaxTrainer
from transformers import AutoTokenizer
import random
import torch.nn.functional as F
from AlarmDataset import AlarmDataset
from torch.utils.data import DataLoader
from torch import Tensor
import pickle



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

        self.engine = self.init_remax_engine()
        self.trainers = self.init_trainer()
        self.model = self.engine.actor

    def init_remax_engine(self):
        engine = RLHFEngine(
            actor_model_name_or_path=self.args.model_name_or_path,
            reward_model_name_or_path="",   # reward model will not be used, we get rewards from metrics
            tokenizer=self.tokenizer,
            num_total_iters=100,
            args=self.args,
        )

        return engine

    def init_trainer(self):
        return ReMaxTrainer(self.engine, self.args)

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
        return {
            'fluency': result['mauve'],
            'correctness': result['str_em'],
            'citation': 2 * (result["citation"]['citation_prec'] * result["citation"]['citation_rec']) / (
                    result["citation"]['citation_prec'] + result["citation"]['citation_rec']) if (result["citation"]['citation_prec'] + result["citation"]['citation_rec']) > 0 else 0,
            'rouge-l': result['rougeLsum']
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
        return {
            'fluency': result['mauve'],
            'correctness': result["claims_nli"],
            'citation': 2 * (result["citation"]['citation_prec'] * result["citation"]['citation_rec']) / (
                    result["citation"]['citation_prec'] + result["citation"]['citation_rec']) if (result["citation"]['citation_prec'] + result["citation"]['citation_rec']) > 0 else 0,
            'rouge-l': result['rougeLsum']
        }

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
        rewards = []
        for i in range(len(data)):
            result = {'length': compute_len([normalized_data[i]])}
            if self.args.dataset_name == 'asqa':
                if data[i]['ultra_reward'] < self.args.hierarchical_threshold:
                    rewards.append({"ultra_reward": (data[i]['ultra_reward'] - self.args.hierarchical_mean) / self.args.hierarchical_std})
                else:
                    batch_reward = self.get_asqa_reward(result, [normalized_data[i]], [data[i]], normalized_data)
                    batch_reward['fluency'] = (batch_reward['fluency'] - self.args.fluency_mean) / self.args.fluency_std
                    batch_reward['correctness'] = (batch_reward['correctness'] - self.args.correctness_mean) / self.args.correctness_std
                    batch_reward['citation'] = (batch_reward['citation'] - self.args.citation_mean) / self.args.citation_std
                    batch_reward['rouge-l'] = (batch_reward['rouge-l'] - self.args.rouge_l_mean) / self.args.rouge_l_std
                    batch_reward['ultra_reward'] = 3 * (data[i]['ultra_reward'] - self.args.hierarchical_mean) / self.args.hierarchical_std
                    rewards.append(batch_reward)
            if self.args.dataset_name == 'eli5':
                if data[i]['ultra_reward'] < self.args.hierarchical_threshold:
                    rewards.append({"ultra_reward": (data[i]['ultra_reward'] - self.args.hierarchical_mean) / self.args.hierarchical_std})
                else:
                    batch_reward = self.get_asqa_reward(result, [normalized_data[i]], [data[i]], normalized_data)
                    batch_reward['fluency'] = (batch_reward['fluency'] - self.args.fluency_mean) / self.args.fluency_std
                    batch_reward['correctness'] = (batch_reward[
                                                       'correctness'] - self.args.correctness_mean) / self.args.correctness_std
                    batch_reward['citation'] = (batch_reward[
                                                    'citation'] - self.args.citation_mean) / self.args.citation_std
                    batch_reward['rouge-l'] = (batch_reward['rouge-l'] - self.args.rouge_l_mean) / self.args.rouge_l_std
                    batch_reward['ultra_reward'] = 3 * (data[i][
                                                        'ultra_reward'] - self.args.hierarchical_mean) / self.args.hierarchical_std
                    rewards.append(batch_reward)
        return rewards, normalized_data

    @torch.no_grad()
    def generate(
            self,
            prompt_ids: Tensor,
            attention_mask: Tensor,
            do_sample: bool = True
    ):
        self.model.eval()

        max_tokens = min(self.args.max_new_tokens, self.args.max_length - prompt_ids.shape[1])
        if do_sample:
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
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generation = self.tokenizer.batch_decode(outputs[:, prompt_ids.size(1):], skip_special_tokens=True)
        self.model.train()
        return generation


class MOON:
    def __init__(
            self,
            args: Optional[Namespace]
    ):
        self.args = args
        self.print_args()
        self.set_seed()

        self.eval_data = self.load_data()

        self.llm = LLM(self.args)

        # ['fluency', 'correctness', 'citation', 'rouge-l']
        self.total_sample = 1
    def print_args(self):
        for k in self.args.__dict__:
            logger.info(f"{k}: {self.args.__dict__[k]}")

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    def load_data(self):
        # "eval_samples.pkl" "asqa_llama2_eval_data.pkl" "eli5_llama3_eval_data.pkl" "eli5_llama2_eval_data.pkl"
        eval_data_path = "eli5_llama2_eval_data.pkl"
        with open(eval_data_path, "rb") as f:
            eval_data = pickle.load(f)
        return eval_data

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

    def mab_train(self):
        for update_round in range(self.args.Epoch):
            logger.info("update_round: {}".format(update_round))
            total_loss, step = self.mab_epoch(self.eval_data)
            logger.info(f"loss: {total_loss / step}")
        save_path = os.path.join("/home/nuochen/MOON/ckpt", self.args.log_file.split("/")[-1].replace(".log", ".pt"))
        torch.save(self.llm.engine.actor.state_dict(), save_path)

    def mab_epoch(self, eval_data):
        moon_datasets = AlarmDataset(self.args, eval_data, self.llm.tokenizer)
        dataloader = DataLoader(
            dataset=moon_datasets,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=moon_datasets.collate_fn
        )
        step = 0
        total_loss = 0
        experience = []
        total_actor_rewards = {}
        rewards_count = 0
        for batch in tqdm(dataloader, desc="mab training..."):
            # logger.info(batch['prompt'])
            step += 1
            logger.info("step: {}".format(step))
            loss, exp, actors_rewards = self.mab_step(batch)
            for actors_reward in actors_rewards:
                for k, v in actors_reward.items():
                    total_actor_rewards[k] = total_actor_rewards.get(k, 0) + v
                rewards_count += 1
            experience.append(exp)
            if not self.args.test:
                loss.backward()
                self.llm.engine.optim.step()
                self.llm.engine.lr_scheduler.step()
                self.llm.engine.optim.zero_grad()
                total_loss += loss.item()

            actor_rewards = {k: v / rewards_count for k, v in total_actor_rewards.items()}
            logger.info("actor reward: {}".format(actor_rewards))

            if step == self.args.train_steps:
                break
        return total_loss, step

    def mab_step(self, batch):
        self.llm.model.train()
        output_array = self.llm.generate(batch['prompt_ids'], batch['attention_mask'])
        baseline_output_array = self.llm.generate(batch['prompt_ids'], batch['attention_mask'], do_sample=False)
        output_array = self.postprocess(output_array)
        baseline_output_array = self.postprocess(baseline_output_array)
        # logger.info(f"output_array: {output_array}")
        reward_input = {'output': output_array}
        reward_input.update(batch)

        baseline_reward_input = {'output': baseline_output_array}
        baseline_reward_input.update(batch)

        rewards, normalized_data = self.llm.get_reward(reward_input)
        baseline_rewards, _ = self.llm.get_reward(baseline_reward_input)
        [sum(reward.values()) for reward in rewards]
        rl_rewards = [sum(reward.values()) for reward in rewards]
        rl_baseline_rewards = [sum(reward.values()) for reward in baseline_rewards]
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


@logger.catch
def main():
    args = parse_args() # Load config
    # "/ssd1/models/Meta-Llama-3-8B-Instruct" "/ssd1/models/llama2-7b-chat-hf"
    args.model_name_or_path = "/ssd1/models/llama2-7b-chat-hf"
    # 1.8671875 -8.8125 3.859375 -9.5625
    args.hierarchical_threshold = -9.5625
    # 0.7306481051545606 -9.065507977320674 2.3818270530700683 -9.931796875
    args.hierarchical_mean = -9.931796875
    # 2.124216019145826 0.6760938352528688 2.615242708374331 0.8562585217128408
    args.hierarchical_std = 0.8562585217128408
    # 22.009710088267312 41.488053535054085 47.42473045975248 46.07748721619543
    args.fluency_mean = 46.07748721619543
    # 25.56012764410455 25.54460284429697 26.731940344127622 27.237293497171567
    args.fluency_std = 27.237293497171567
    # 49.475 50.225 18.83333333333333 16.666666666666664
    args.correctness_mean = 16.666666666666664
    # 32.606056654486075 33.09925204760843 23.112646850684246 22.94632861011213
    args.correctness_std = 22.94632861011213
    # 59.375625617072984 42.554343874563166 39.70615139110793 21.4510511986147
    args.citation_mean = 21.4510511986147
    # 35.21258194168109 27.798790277124766 31.36422590162703 22.84157235386101
    args.citation_std = 22.84157235386101
    # 39.11160004491581 35.888450070802776 21.307584183580893 21.258523888863515
    args.rouge_l_mean = 21.258523888863515
    # 10.8355541859102 9.693271762591664 8.221272172526552 8.838835679138286
    args.rouge_l_std = 8.838835679138286
    args.log_file = "alarm_llama2_eli5.log"
    logger.add(args.log_file)
    load_autoais()
    moon = MOON(args)
    moon.mab_train()


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
