import sys
import os
from argparse import ArgumentParser

from accelerate import Accelerator
from synthesizer import Synthesizer
from utils import load_model_tokenizer_to_device


def synthesize_instructions():
    parser = ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--task_name', type=str, choices=['humaneval', 'mbpp', 'gsm8k', 'math', 'csqa'], required=True)
    parser.add_argument('--synthesize_num', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--pretraining_tp', type=int, default=1)
    args = parser.parse_args()

    accelerator = Accelerator()
    model, tokenizer = load_model_tokenizer_to_device(args, accelerator.device)
    synthesizer = Synthesizer(args, accelerator, model, tokenizer)
    synthesizer.synthesize()
    

if __name__ == '__main__':
    synthesize_instructions()
