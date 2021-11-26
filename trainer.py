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
import data
import json
import model
import os.path as osp
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
import torch
from typing import List

class Trainer(pl.Trainer):
    def __init__(self, 
                 accelerator: str = 'cpu', 
                 devices: List[int] = None, 
                 max_epochs: int = 1000, 
                 gradient_clip_val: int = 1000,
                 fast_dev_run: int = False, 
                 callbacks: List[pl.callbacks.Callback] = None):
        
        if accelerator == 'cpu':
            devices = None
            
        logger = pl.loggers.TensorBoardLogger(config.logs_path, name=None, log_graph=True)
        
        super().__init__(
            default_root_dir=config.logs_path,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,)
    
    def test(self, **kwargs):
        results = super().test(**kwargs)[0]
        
        with open(osp.join(config.artifacts_path, "results.json"), "w") as f:
            json.dump(results, f)
        
        torch.save(self.model, osp.join(config.artifacts_path, 'model.pth'))
        
def main():
    
    cli = LightningCLI(trainer_class=Trainer)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)
    
    
if __name__ == '__main__':
    
    torch.set_default_dtype(torch.double)
    main()