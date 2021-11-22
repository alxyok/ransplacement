# MIT License

# Copyright (c) 2021 alxyok

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import config
import numpy as np
import os
import os.path as osp
import pytorch_lightning as pl
import torch

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, batch_size):
        
        self.batch_size = batch_size 
        
        # tensors, invariants, and labels have very different shapes. it poses a problem with the __getitem__ method, that outputs a Tensor. it would be convienient to be able to return a list of Tensors, but the specs are implemented differently, probably for optimization purposes
        self.tensors = np.load(osp.join(config.processed_dir, 'tensors.npy'))
        self.invariants = np.load(osp.join(config.processed_dir, 'invariants.npy'))
        self.labels = np.load(osp.join(config.processed_dir, 'labels.npy'))
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        rowar = (1, -1)
        
        t = torch.tensor(self.tensors[idx, :]).reshape(rowar)
        i = torch.tensor(self.invariants[idx, :]).reshape(rowar)
        l = torch.tensor(self.labels[idx, :]).reshape(rowar)
        
        stacked = torch.hstack((t, i, l))
        return stacked
    

class LitDataset(pl.LightningDataModule):
    
    def __init__(self, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        
    def prepare_data(self):
        dataset = Dataset(self.batch_size)
        
        size = len(dataset)
        
        self.test_dataset = dataset[:size // 10]
        self.val_dataset = dataset[size // 10:size // 5]
        self.train_dataset = dataset[size // 5:]
        
    def setup(self, stage: str):
        dataset = Dataset(self.batch_size)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)