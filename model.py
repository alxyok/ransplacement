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

import activation as act
import config
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.functional as F
from typing import List


@MODEL_REGISTRY
class LitTBNN(pl.LightningModule):
    
    def __init__(self, 
                 in_feats: int = 47, 
                 out_feats: int = 10, 
                 num_layers: int = 3, 
                 hidden_feats: int = 20, 
                 activation: str = 'selu', 
                 lr: float = 1e-4):
        
        super().__init__()
        
        self.activation = act.fn(activation)
        self.lr = lr
        self.out_feats = out_feats
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hidden_feats))
        self.layers.extend([nn.Linear(hidden_feats, 
                                      hidden_feats) for _ in range(num_layers)])
        self.layers.append(nn.Linear(hidden_feats, out_feats))
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
    
        tensors, invariants = x
        
        out = invariants
        for layer in self.layers:
            out = layer(out)
            out = self.activation(out)
            
        out = out.reshape((-1, self.out_feats, 1, 1))
        out = torch.einsum('ijkl,ijmn->iklmn', out, tensors)
        out = out.reshape((-1, 9))
        out = torch.hstack((out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 4:5], out[:, 5:6], out[:, 8:9],))
        
        return out
    
    def configure_optimizers(self):
        return optim.NAdam(self.parameters(), lr=self.lr, eps=1e-7)
    
    def _common(self, batch: List[torch.Tensor], batch_idx: int, stage: str) -> float:
        
        tensors = torch.stack([x[0] for x in batch])
        invariants = torch.stack([x[1] for x in batch])
        labels = torch.stack([x[2] for x in batch])
        
        pred = self((tensors, invariants))
        
        loss = F.mean_squared_error(pred, labels)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        
        return loss
    
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> float:
        loss = self._common(batch, batch_idx, 'train')
        return loss
    
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int):
        loss = self._common(batch, batch_idx, 'val')
        
    def test_step(self, batch: List[torch.Tensor], batch_idx: int):
        loss = self._common(batch, batch_idx, 'test')