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
from metaflow import FlowSpec, step
import numpy as np
import os
import os.path as osp

class BuildDatasetFlow(FlowSpec):
    
    @step
    def start(self):
        """
        Purge the data/processed directory, and create the list of files (use-case based) to load the data for.
        """
        
        os.makedirs(config.processed_dir, exist_ok=True)
        
        if config.purge:
            for file in os.listdir(config.processed_dir):
                try:
                    os.remove(os.path.join(config.processed_dir, file))
                except:
                    pass
        
        self.cases = [
            'DUCT_1100',
            'DUCT_1150',
            'DUCT_1250',
            'DUCT_1300',
            'DUCT_1350',
            'DUCT_1400',
            'DUCT_1500',
            'DUCT_1600',
            'DUCT_1800',
            'DUCT_2205',
            'DUCT_2400',
            'DUCT_2600',
            'DUCT_2900',
            'DUCT_3200',
            'PHLL_case_0p5',
            'PHLL_case_0p8',
            'PHLL_case_1p0',
            'PHLL_case_1p5',
            'BUMP_h20',
            'BUMP_h26',
            'BUMP_h31',
            'BUMP_h42',
            'CNDV_12600',
            'CNDV_20580',
            'CBFS_13700'
         ]
            
        self.next(self.load_data, foreach="cases")
                  
                  
    @step
    def load_data(self):
        """
        For each use-case, load the data flavors (tensors, invariants and labels) into a NumPy array.
        """
        
        self.tensors = np.load(os.path.join(config.raw_dir, config.dataset, f'{config.dataset}_{self.input}_Tensors.npy'))
        self.invariants = np.load(os.path.join(config.raw_dir, config.dataset, f'{config.dataset}_{self.input}_I1.npy'))
        
        # x = np.hstack((tensors, invariants))
        
        self.labels = np.load(os.path.join(config.raw_dir, 'labels', f'{self.input}_b.npy'))
        self.labels = np.reshape(self.labels, (-1, 9))
        self.labels = np.delete(self.labels, (3, 6, 7), axis=1)
        
        self.next(self.join)
        
        
    @step
    def join(self, inputs):
        """
        Join the parallel branches, stack the previously loaded files, and save into a single output file for each of the data flavor.
        """
        
        # merge
        tensors = np.concatenate([input_.tensors for input_ in inputs])
        invariants = np.concatenate([input_.invariants for input_ in inputs])
        labels = np.concatenate([input_.labels for input_ in inputs])
        
        # data = np.concatenate((x, y), axis=1)
        np.save(os.path.join(config.processed_dir, 'tensors.npy'), tensors)
        np.save(os.path.join(config.processed_dir, 'invariants.npy'), invariants)
        np.save(os.path.join(config.processed_dir, 'labels.npy'), labels)
        
        self.next(self.end)
        
        
    @step
    def end(self):
        """
        End the flow.
        """
        
        pass
            
if __name__ == '__main__':
    
    BuildDatasetFlow()