import sys
import os

from accelerate import Accelerator


class ProcessHelperMixin:
    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator
        
    @property
    def num_processes(self):
        return self.accelerator.num_processes

    @property
    def process_index(self):
        return self.accelerator.process_index

    @property
    def local_process_index(self):
        return self.accelerator.local_process_index

    @property
    def device(self):
        return self.accelerator.device
    
    @property
    def is_main_process(self):
        """True for one process only."""
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self):
        """True for one process per server."""
        return self.accelerator.is_local_main_process
    
    def pad_across_processes(self, tensor, dim=0, pad_index=0, pad_first=False):
        return self.accelerator.pad_across_processes(tensor, dim, pad_index, pad_first)
    
    def gather(self, tensor):
        return self.accelerator.gather(tensor)
    