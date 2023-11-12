import os
import json
import gzip
import multiprocessing
import regex as re
from functools import partial, wraps
from typing import Iterable, Dict

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from models.codellama import CodeLlamaTokenizer


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if not all(x.isspace() for x in line):
                    yield json.loads(line)

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))
                
                
def mpExceptionHandler(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except BaseException as e:
            print(f"Process {multiprocessing.current_process().pid} is raising: {e}")
            raise e

    return decorated


@mpExceptionHandler
def load_model_tokenizer_to_device(args, i):
    if 'codellama' in args.base_model.lower():
        tokenizer = CodeLlamaTokenizer.from_pretrained(args.base_model)
    elif 'llama' in args.base_model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)  
    else:
        raise NotImplementedError(f"Loading script for {args.base_model} not implemented")
    
    tokenizer.padding_side = 'left'
    
    # If you wish faster inference, set `config.pretraining_tp` to 1, but at the cost of higher GPU memory usage
    # Reference: https://huggingface.co/docs/transformers/v4.32.0/en/model_doc/llama2
    config = LlamaConfig.from_pretrained(args.base_model)
    config.pretraining_tp = args.pretraining_tp
    config.use_cache = True
    
    model = LlamaForCausalLM.from_pretrained(args.base_model, config=config, torch_dtype=torch.float16, device_map={'': i})
    
    if model.generation_config.pad_token_id is None or model.config.pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer
