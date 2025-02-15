# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
from transformers import AutoModelForCausalLM, get_scheduler
from remax.utils.model.model_utils import (
    create_hf_model,
)
from remax.utils.module.lora import (
    convert_linear_layer_to_lora,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
)
from remax.utils.utils import get_optimizer_grouped_parameters
from torch.optim import Adam
from transformers import (
    AutoModel,
)
from remax.utils.model.reward_model import RewardModel
from remax.utils.utils import load_state_dict_into_model
from safetensors import safe_open

"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""


def log_init(model_name, stime=None):
    # if torch.distributed.get_rank() == 0:
    tag = "start" if stime is None else "end"
    suffix = "ing" if stime is None else "ed"
    duration = ""
    if stime is not None:
        duration = "(duration: {:.2f}s)".format(time.time() - stime)
    msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
    stars = (90 - len(msg)) // 2
    extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
    print("*" * stars + msg + "*" * stars + extra_star)
    return time.time()


class RLHFEngine:
    def __init__(
        self,
        actor_model_name_or_path,
        reward_model_name_or_path,
        tokenizer,
        args,
        num_total_iters,
        device_map="auto"
    ):
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer

        self.actor, self.optim, self.lr_scheduler = self._init_actor(
            actor_model_name_or_path=actor_model_name_or_path,
            device_map=device_map
        )
        self.ref = self._init_ref(actor_model_name_or_path=actor_model_name_or_path)
        self.actor_ema = None
        if self.args.enable_ema:
            self.actor_ema = self._init_ema(
                actor_model_name_or_path=actor_model_name_or_path
            )
        self.critic = None
        # self.reward = self._init_reward(
        #     reward_model_name_or_path=reward_model_name_or_path
        # )
        self.reward = None
        # self.to_device()

    def to_device(self):
        self.actor.to(self.args.device)
        self.ref.to(self.args.device)
        self.reward.to(self.args.device)

    def _init_actor(
            self,
            actor_model_name_or_path,
            device_map="auto"
    ):
        stime = log_init("Actor")
        # Model
        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=actor_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=None,
            disable_dropout=self.args.disable_actor_dropout,
            device_map=device_map
        )

        # LoRA
        if self.args.actor_lora_dim > 0:
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name, self.args.actor_lora_dim
            )
            if self.args.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)
                actor_model = make_model_gradient_checkpointing_compatible(actor_model)

        # Optimizer
        AdamOptimizer = Adam
        optim_params = get_optimizer_grouped_parameters(
            actor_model,
            self.args.actor_weight_decay,
            self.args.actor_lora_learning_rate,
        )
        optim = AdamOptimizer(
            optim_params, lr=self.args.actor_learning_rate, betas=(0.9, 0.95)
        )

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        log_init("Actor", stime=stime)

        return actor_model, optim, lr_scheduler

    def _init_ref(self, actor_model_name_or_path):
        stime = log_init("Ref")

        ref_model = create_hf_model(
            AutoModelForCausalLM, actor_model_name_or_path, self.tokenizer, None
        )
        log_init("Ref", stime=stime)
        return ref_model

    def _init_ema(self, actor_model_name_or_path):
        # stime = log_init("EMA")
        # # DS Config
        # zero_stage = self.args.reference_zero_stage
        # if zero_stage != 3 and zero_stage != 0:
        #     zero_stage = 0
        #     print_rank_0(
        #         f"It is useless to set stage = {zero_stage} for the EMA model (as it does not have optimizer and gradients). We set stage = 0"
        #     )
        # ds_config = get_eval_ds_config(
        #     self.args.offload_reference_model, zero_stage, bf16=self.args.actor_bf16
        # )
        # ds_config[
        #     "train_micro_batch_size_per_gpu"
        # ] = self.args.per_device_training_batch_size
        # # TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        # ds_config["train_batch_size"] = (
        #     self.args.per_device_training_batch_size
        #     * torch.distributed.get_world_size()
        #     * self.args.gradient_accumulation_steps_actor
        # )
        #
        # actor_model_ema = create_hf_model(
        #     AutoModelForCausalLM, actor_model_name_or_path, self.tokenizer, ds_config
        # )
        # if self.args.actor_lora_dim > 0:
        #     actor_model_ema = convert_linear_layer_to_lora(
        #         actor_model_ema,
        #         self.args.actor_lora_module_name,
        #         self.args.actor_lora_dim,
        #     )
        #
        # ema_engine, *_ = deepspeed.initialize(model=actor_model_ema, config=ds_config)
        #
        # log_init("EMA", stime=stime)
        # return ema_engine
        return None

    def _init_reward(self, reward_model_name_or_path):
        stime = log_init("Reward")

        # Model
        reward_model = create_reward_model(
            model_name_or_path=reward_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=None,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            disable_dropout=self.args.disable_reward_dropout,
            zero_stage=0,
        )

        log_init("Reward", stime=stime)
        return reward_model


def create_reward_model(
        model_name_or_path,
        tokenizer,
        ds_config,
        num_padding_at_beginning=0,
        rlhf_training=False,
        disable_dropout=False,
        zero_stage=0,
        eval_mode=False,
):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    critic_model = create_hf_model(
        AutoModel,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        disable_dropout,
    )

    critic_model = RewardModel(
        critic_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning
    )

    if rlhf_training:
        # load critic model from checkpoint
        if "model.safetensors" in os.listdir(model_name_or_path):
            model_ckpt_path = os.path.join(model_name_or_path, "model.safetensors")
            model_ckpt_state_dict = {}
            with safe_open(model_ckpt_path, framework="pt", device='cpu') as f:
                for k in f.keys():
                    model_ckpt_state_dict[k] = f.get_tensor(k)
        else:
            model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
            model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )

    return critic_model
