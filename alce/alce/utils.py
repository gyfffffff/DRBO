import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time
import torch
import torch.nn.functional as F
import logging
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from typing import Tuple, Callable

histroy_logs = set()
def print_rank_0(info, only_on_cuda0=False):
    if accelerator and not accelerator.is_main_process:
        return
    if only_on_cuda0 and info not in histroy_logs:
        histroy_logs.add(info)
        logging.info(info)
    return

def top_p_logits(logits, topp=0.9, filter_value=0, min_topk=1):
    """
    Filter a distribution of logits using nucleus (top-p) filtering
    https://github.com/OpenLMLab/MOSS/blob/e088f438d1a95d424c6dffef0d73134ebe62cb72/models_jittor/generation.py#L146
    """
    cum_logits = logits.clone()
    if topp > 0:
        logits_sorted, inds = torch.sort(logits, dim=-1, descending=True)
        mask = (logits_sorted.cumsum(dim=-1) - logits_sorted) >= topp
        mask[:, :min_topk] = False
        # Remove tokens with cumulative top_p above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=inds, src=mask)
        cum_logits[mask] = filter_value
        cum_logits.div_(cum_logits.sum(dim=-1, keepdim=True))
        
    return cum_logits

accelerator = None
def setup_accelerator():
    global accelerator
    if accelerator is None:
        accelerator = Accelerator(split_batches=True)
    return accelerator

def synchronize_if_distributed():
    if accelerator.use_distributed:
        accelerator.wait_for_everyone()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=None):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = doc['text']
    if use_shorter is not None:
        text = doc[use_shorter]
    return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))


def get_shorter_text(item, docs, ndoc, key):
    doc_list = []
    for item_id, item in enumerate(docs):
        if key not in item:
            if len(doc_list) == 0:
                # If there aren't any document, at least provide one (using full text)
                item[key] = item['text']
                doc_list.append(item)
            logger.warn(f"No {key} found in document. It could be this data do not contain {key} or previous documents are not relevant. This is document {item_id}. This question will only have {len(doc_list)} documents.")
            break
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
        doc_list.append(item)
        if len(doc_list) >= ndoc:
            break
    return doc_list


def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, use_shorter=None, test=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt


def load_model(model_name_or_path, dtype=torch.float16, int8=False, reserve_memory=10,for_ppo_update=False):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Llama: set up the root dir
    open_source_models = ["llama", "alpaca", "vicuna", "oasst"]
    if any([m in model_name_or_path for m in open_source_models]):
       model_name_or_path = os.path.join(os.environ["LLAMA_ROOT"], model_name_or_path)
    print(model_name_or_path)
    
    # Load the FP16 model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    if int8:
        logger.warn("Use LLM.int8")
    start_time = time.time()
    # print("aaaaaa:",model_name_or_path)
    if for_ppo_update:
        from trl import AutoModelForCausalLMWithValueHead
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name_or_path,
            cache_dir=os.environ["LLAMA_ROOT"],
            # device_map='auto',
            torch_dtype=dtype,
            max_memory=get_max_memory(),
            load_in_8bit=int8
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=os.environ["LLAMA_ROOT"],
            # device_map='auto',
            torch_dtype=dtype,
            max_memory=get_max_memory(),
            load_in_8bit=int8,
        )
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer


def print_gpu_memory_usage():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(i) / 1e9
            cached_memory = torch.cuda.memory_reserved(i) / 1e9
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Allocated Memory: {allocated_memory:.2f} GB")
            print(f"  Cached Memory: {cached_memory:.2f} GB")
    else:
        print("CUDA is not available. No GPU detected.")