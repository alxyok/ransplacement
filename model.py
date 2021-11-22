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
import torch
import torch.nn as nn
import torch_optimizer as optim
import torchmetrics.functional as F
from typing import List

class LitMLP(pl.LightningModule):
    
    def __init__(self, 
                 in_features: int = 47, 
                 out_features: int = 10, 
                 num_layers: int = 3, 
                 hidden_features: int = 20, 
                 activation: str = 'selu', 
                 lr: float = 1e-4):
        
        super().__init__()
        
        self.activation = act.fn(activation)
        self.lr = lr
        self.out_features = out_features
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features))
        self.layers.extend([nn.Linear(hidden_features, 
                                      hidden_features) for _ in range(num_layers)])
        self.layers.append(nn.Linear(hidden_features, out_features))
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        
        tensors, invariants = x
        
        out = invariants
        for layer in self.layers:
            out = layer(out)
            out = self.activation(out)
            
        out = out.reshape((-1, self.out_features, 1, 1))
        out = out.broadcast_to((out.shape[0], self.out_features, 3, 3))
        out = out * tensors
        out = out.sum(dim=1)
        out = out.reshape((-1, 9))
        
        out = torch.hstack((out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 4:5], out[:, 5:6], out[:, 8:9],))
            
        return out
    
    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)
    
    def _common(self, batch: List[torch.Tensor], batch_idx: int, stage: str) -> float:
        
        i_tensors = config.batch_size * 10 * 3 * 3
        i_invariants = config.batch_size * 47
        i_labels = config.batch_size * 6
        
        tensors = batch[:i_tensors].reshape((config.batch_size, 10, 3, 3))
        invariants = batch[i_tensors:i_tensors + i_invariants].reshape((config.batch_size, 47))
        labels = batch[i_tensors + i_invariants:].reshape((config.batch_size, 6))
        
        # tensors, invariants, labels = batch
        pred = self((tensors, invariants))
        
        loss = F.mean_squared_error(pred, labels)
        r2 = F.r2_score(pred, labels)
        
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        self.log(f"{stage}_r2", r2, on_step=True)
        
        return loss, r2
    
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> float:
        loss, _ = self._common(batch, batch_idx, 'train')
        
        return loss
    
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int):
        loss = self._common(batch, batch_idx, 'val')
        
    def test_step(self, batch: List[torch.Tensor], batch_idx: int):
        loss = self._common(batch, batch_idx, 'test')