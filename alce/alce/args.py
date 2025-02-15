import argparse
import os

from transformers import SchedulerType

root_path = os.path.abspath(os.path.dirname("../"))

model_name_or_path = "/home/models/Llama-2-7B-Chat"
eval_path = "/home/moon_alce/alce_data/eli5_eval_bm25_top100_reranked_oracle.json"
prompt_path = "/home/moon_alce/prompts/eli5_default.json"


def parse_args():
    parser = argparse.ArgumentParser(description="MEAL args")

    parser.add_argument("--device", default="cuda", type=str, help="device")

    # Train
    parser.add_argument("--seed", default=2024, type=int, help="seed")
    parser.add_argument(
        "--model_name_or_path",
        default=model_name_or_path,
        type=str,
        help="model_name_or_path",
    )
    parser.add_argument(
        "--prompt_path", default=prompt_path, type=str, help="Path to the prompt file"
    )
    parser.add_argument(
        "--eval_path", default=eval_path, type=str, help="Path to the eval file"
    )
    parser.add_argument(
        "--disable_actor_dropout", action="store_true", help="disable_actor_dropout"
    )

    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--train_steps", type=int)

    parser.add_argument("--enable_ema", action="store_true", help="enable_ema")
    parser.add_argument("--actor_lora_dim", default=0, type=int, help="actor_lora_dim")
    parser.add_argument(
        "--actor_lora_learning_rate",
        default=5e-4,
        type=int,
        help="actor_lora_learning_rate",
    )
    parser.add_argument(
        "--actor_weight_decay", default=0, type=float, help="actor_weight_decay"
    )
    parser.add_argument(
        "--actor_learning_rate", default=2e-5, type=float, help="actor_learning_rate"
    )
    parser.add_argument(
        "--num_warmup_steps", default=0, type=int, help="num_warmup_steps"
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--disable_reward_dropout",
        action="store_true",
        help="Disable the dropout of the reward model.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--mab_warmup_steps", default=10, type=int, help="mab_warmup_steps"
    )
    parser.add_argument("--mab_e", default=4, type=int, help="mab_e")
    parser.add_argument("--mab_d", default=25, type=int, help="mab_d")
    parser.add_argument("--tau", default=0.0, type=float, help="tau")
    parser.add_argument("--batch_size", default=2, type=int, help="batch_size")
    parser.add_argument("--num_workers", default=8, type=int, help="num_workers")

    #
    parser.add_argument(
        "--actor_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model.",
    )
    parser.add_argument(
        "--reference_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Reference model.",
    )
    parser.add_argument(
        "--kl_ctl", type=float, default=0.8, help="KL penalty coefficient."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor for MC return estimate.",
    )
    parser.add_argument("--delta", default=False, help="delta", action="store_true")
    parser.add_argument(
        "--weight_avg", action="store_true", default=False, help="weight_avg"
    )

    # Generation
    parser.add_argument(
        "--max_new_tokens", default=300, type=int, help="max_new_tokens"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max length the model can take. Should set properly wrt the model to avoid position overflow.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.95, help="Temperature for decoding"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Nucleus sampling top-p"
    )

    #
    parser.add_argument("--shot", default=1, type=int, help="shot")
    parser.add_argument(
        "--use_task_specific_prompt",
        default=True,
        type=bool,
        help="use_task_specific_prompt",
    )
    parser.add_argument(
        "--use_letter_choices", default=True, type=bool, help="use_letter_choices"
    )
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
    parser.add_argument("--ndoc", default=3, type=int, help="ndoc")
    parser.add_argument(
        "--no_doc_in_demo", default=False, type=bool, help="no_doc_in_demo"
    )
    parser.add_argument(
        "--fewer_doc_in_demo", default=False, type=bool, help="fewer_doc_in_demo"
    )
    parser.add_argument("--use_shorter", default=None, type=str, help="use_shorter")
    parser.add_argument("--tag", default="gtr", type=str, help="tag")
    parser.add_argument("--Epoch", type=int, default=10, help="整个数据集更新几次")
    parser.add_argument(
        "--retrieve_in_all_docs",
        type=bool,
        default=False,
        help="Retrieve in all documents instead of just top ndoc",
    )
    parser.add_argument(
        "--azure", action="store_true", default=False, help="Azure openai API"
    )

    parser.add_argument("--test", action="store_true", default=False)
    return parser.parse_args()
