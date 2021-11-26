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
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch

def load_data():
    tensors = np.load(osp.join(config.processed_dir, 'tensors.npy'))
    invariants = np.load(osp.join(config.processed_dir, 'invariants.npy'))
    labels = np.load(osp.join(config.processed_dir, 'labels.npy'))
    
    t = torch.tensor(tensors)
    i = torch.tensor(invariants)
    l = torch.tensor(labels)
    
    return t, i, l
    
    
@DATAMODULE_REGISTRY
class PointwiseDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size: int = 1024):
        super().__init__()
        self.batch_size = batch_size
        self.data = load_data()
        
    def prepare_data(self):
        dataset = torch.utils.data.TensorDataset(*self.data)
        size = len(dataset)
        
        self.test_dataset = torch.utils.data.TensorDataset(*dataset[:size // 10])
        self.val_dataset = torch.utils.data.TensorDataset(*dataset[size // 10:size // 5])
        self.train_dataset = torch.utils.data.TensorDataset(*dataset[size // 5:])
        
    def setup(self, stage: str):
        dataset = torch.utils.data.TensorDataset(*self.data)
        
    def _collate_fn(self, batch):
        return batch
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self._collate_fn,
            shuffle=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )