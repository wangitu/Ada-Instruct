import sys
import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed, DefaultDataCollator

from config import get_generation_config
from tasks import get_task
from process_helper import ProcessHelperMixin


class Synthesizer(ProcessHelperMixin):
    def __init__(self, args, accelerator, model, tokenizer):
        ProcessHelperMixin.__init__(self, accelerator)
        
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        
    def _parallel_generation(self, inputs):
        set_seed(42 + self.process_index)
        
        synthesize_num = self.args.synthesize_num
        
        class AllInOneDataset(Dataset):
            def __getitem__(self, index):
                return inputs
                
            def __len__(self):
                return synthesize_num
            
        
        dataloader = DataLoader(AllInOneDataset(), batch_size=self.args.batch_size, collate_fn=DefaultDataCollator())
        # we only wrap data loader to avoid extra memory occupation
        self.model.to(self.device)
        dataloader = self.accelerator.prepare(dataloader)
        
        generation_config = get_generation_config(self.args.task_name)
        
        output_tokens = []
        for batch in tqdm(dataloader, desc='synthesizing instructions...', disable=not self.is_local_main_process):
            # we could avoid `batch.to(self.device)` since we set the accelerator with `device_placement=True`
            output = self.model.generate(
                **batch,
                generation_config=generation_config
            )
            output_ids = output if isinstance(output, torch.Tensor) else output.sequences
            
            # pad across processes before gather
            output_ids = self.pad_across_processes(
                output_ids, dim=1, pad_index=self.tokenizer.pad_token_id
            )
            # gather across processes and offload to cpu
            output_ids = self.gather(output_ids).cpu()
            output_tokens.extend(output_ids)
            
        return output_tokens
        
    def synthesize(self, task_args=None):
        task_name, args = (task_args.task_name, task_args) if task_args is not None else (self.args.task_name, self.args)
        task = get_task(task_name, args)
        prompt = task.get_prompt()
        inputs = self.tokenizer(prompt, add_special_tokens=False)
        
        output_tokens = self._parallel_generation(inputs)
        outputs = []
        for i, output in enumerate(self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)):
            outputs.append({
                'id': i,
                'instruction': task.get_response(output)
            })
            
        if self.is_local_main_process:
            # if `self.model = None` or `del self.model`, there is still reference outside. Loading model in `__init__` could work
            self.model.to('cpu')
            torch.cuda.empty_cache()
                        
            retained, discarded = task.postprocess_synthesized_instructions(outputs)
            
            print(f"{len(retained)} retained, {len(discarded)} discarded")
            
            out_dir, out_file = os.path.split(self.args.out_file)
            out_file_name, ext = os.path.splitext(out_file)
            os.makedirs(out_dir, exist_ok=True)
            
            with open(self.args.out_file, 'w', encoding='utf-8') as f:
                json.dump(retained, f, ensure_ascii=False, indent=2)
                
            with open(os.path.join(out_dir, out_file_name + '_discarded' + ext), 'w', encoding='utf-8') as f:
                json.dump(discarded, f, ensure_ascii=False, indent=2)
    