import sys
import os
import json
import random
from time import time
from collections import deque

import threading
from queue import Empty


class OpenaiAPIKeyPool:
    def __init__(self, key_file):
        self.key_file = key_file
        with open(key_file, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        
        self.ordered_keys = keys['available']
        
        # with lock
        self.pool = deque(keys['available'])
        self.staged = []
        self.aborted = keys['aborted']

        self.mutex = threading.Lock()
        self.not_empty = threading.Condition(self.mutex)
        
    @property
    def available(self):
        """
        This method is not thread safe. Handle with caution!
        """
        return list(self.pool)
        
    def _wait_until_available(self, block=True, timeout=None):
        """
        This method must be used after having acquired `self.mutex`!
        """
        if len(self.pool) == 0 and len(self.staged) == 0:
            raise Empty
        
        if not block:
            if len(self.pool) == 0:
                raise Empty
        elif timeout is None:
            while len(self.pool) == 0:
                self.not_empty.wait()
        elif timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        else:
            endtime = time() + timeout
            while len(self.pool) == 0:
                remaining = endtime - time()
                if remaining <= 0.0:
                    raise Empty
                self.not_empty.wait(remaining)
                
        
    def put(self, key):
        with self.mutex:
            self.pool.append(key)
            if key in self.staged:
                self.staged.remove(key)
            self.not_empty.notify()
    
    def get(self, block=True, timeout=None):
        with self.not_empty:
            self._wait_until_available(block, timeout)
            key = self.pool.popleft()
            self.staged.append(key)
        return key
        
    def random(self, block=True, timeout=None):
        with self.not_empty:
            self._wait_until_available(block, timeout)
            key = random.choice(self.pool)
            self.pool.remove(key)
            self.staged.append(key)
        return key
        
    def delete(self, key):
        with self.mutex:
            if key in self.pool:
                self.pool.remove(key)
            elif key in self.staged:
                self.staged.remove(key)
            else:
                return
            self.aborted.append(key)

    def dump(self, key_file=None):
        key_file = key_file if key_file is not None else self.key_file
        with self.mutex:
            available_keys = set(self.available + self.staged)
            
        ordered_available_keys = []
        for key in self.ordered_keys:
            if key in available_keys:
                ordered_available_keys.append(key)
                
        with open(key_file, 'w', encoding='utf-8') as f:
            json.dump({
                'available': ordered_available_keys, 'aborted': self.aborted
            }, f, ensure_ascii=False, indent=2)
                