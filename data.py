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

# class PointwiseDataset(torch.utils.data.TensorDataset):
    
#     def __init__(self):
        
#         # tensors, invariants, and labels have very different shapes. it poses a problem with the __getitem__ method, that outputs a Tensor. it would be convienient to be able to return a list of Tensors, but the specs are implemented differently, probably for optimization purposes
#         self.tensors = np.load(osp.join(config.processed_dir, 'tensors.npy'))
#         self.invariants = np.load(osp.join(config.processed_dir, 'invariants.npy'))
#         self.labels = np.load(osp.join(config.processed_dir, 'labels.npy'))
    
#     def __len__(self):
#         return len(self.tensors)
    
#     def __getitem__(self, idx):
#         print(type(idx))
#         print(idx)
        
#         row = (1, -1)
        
#         t = torch.tensor(self.tensors[idx, :])
#         i = torch.tensor(self.invariants[idx, :])
#         l = torch.tensor(self.labels[idx, :])
#         return t, i, l
        
#         # print(t.shape, i.shape, l.shape)
#         # concat = torch.cat((t, i, l))
        
# #         print('*****')
# #         print(concat.shape)
# #         print('*****')
        
# #         return t, i, l
#         # return concat

def load_data():
    tensors = np.load(osp.join(config.processed_dir, 'tensors.npy'))
    invariants = np.load(osp.join(config.processed_dir, 'invariants.npy'))
    labels = np.load(osp.join(config.processed_dir, 'labels.npy'))
    
    t = torch.tensor(tensors)
    i = torch.tensor(invariants)
    l = torch.tensor(labels)
    
    return t, i, l
    

class PointwiseDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self.data = load_data()
        
    def prepare_data(self):
        dataset = torch.utils.data.TensorDataset(*self.data)
        # loader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=lambda x: x)
        # for batch in loader:
        #     print(len(batch))
        #     print('*****')
        #     break
        # dataset = PointwiseDataset()
        
        size = len(dataset)
        
        self.test_dataset = torch.utils.data.TensorDataset(*dataset[:size // 10])
        self.val_dataset = torch.utils.data.TensorDataset(*dataset[size // 10:size // 5])
        self.train_dataset = torch.utils.data.TensorDataset(*dataset[size // 5:])
        
        # self.test_dataset = torch.utils.data.TensorDataset(*self.data)
        # loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=8, collate_fn=lambda x: x)
        # for batch in loader:
        #     print(len(batch))
        #     break
        
    def setup(self, stage: str):
        dataset = torch.utils.data.TensorDataset(*self.data)
        # dataset = PointwiseDataset()
        
    def _collate_fn(self, batch):
        return batch
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=lambda x: x
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            collate_fn=lambda x: x
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            collate_fn=lambda x: x
        )