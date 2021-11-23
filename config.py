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

import names
import os
import os.path as osp
    
root_path = osp.join(osp.dirname(osp.realpath(__file__)))
data_path = osp.join(root_path, 'data')
raw_dir = osp.join(data_path, 'raw')
processed_dir = osp.join(data_path, 'processed')

purge = True
dataset = 'kepsilon'

name = names.get_last_name().lower()

experiments_path = osp.join(root_path, 'experiments')
experiment_path = osp.join(experiments_path, name)
logs_path = osp.join(experiment_path, 'logs')
artifacts_path = osp.join(experiment_path, 'artifacts')
plots_path = osp.join(experiment_path, 'plots')

paths = [experiments_path,
         experiment_path, 
         logs_path, 
         artifacts_path, 
         plots_path]

for path in paths:
    os.makedirs(path, exist_ok=True)
    
batch_size = 128