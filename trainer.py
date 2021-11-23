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
import model
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
import torch

torch.set_default_dtype(torch.double)

class LitTrainer(pl.Trainer):
    
    def __init__(self, accelerator='cpu'):
        logger = pl.loggers.TensorBoardLogger(config.logs_path, name=None, log_graph=True)
        
        super().__init__(
            default_root_dir=config.logs_path,
            logger=logger)

cli = LightningCLI(model_class=model.LitMLP, 
                   datamodule_class=data.LitDataModule,
                   trainer_class=LitTrainer)