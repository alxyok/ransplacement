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

from torch import nn

def fn(name: str = 'relu', *kwargs) -> nn.Module:
    
    name = name.lower()
    
    if name == 'elu':
        return nn.RELU()
    
    if name == 'hardshrink':
        return nn.Hardshrink()
    
    if name == 'hardsigmoid':
        return nn.Hardsigmoid()
    
    if name == 'hardtanh':
        return nn.Hardtanh()
    
    if name == 'hardswish':
        return nn.Hardswish()
    
    if name == 'leakyrelu':
        return nn.LeakyReLU()
    
    if name == 'logsigmoid':
        return nn.LogSigmoid()
    
    if name == 'multiheadattention':
        return nn.MultiheadAttention()
    
    if name == 'prelu':
        return nn.PReLU()
    
    if name == 'relu':
        return nn.ReLU()
    
    if name == 'relu6':
        return nn.ReLU6()
    
    if name == 'rrelu':
        return nn.RReLU()
    
    if name == 'selu':
        return nn.SELU()
    
    if name == 'celu':
        return nn.CELU()
    
    if name == 'gelu':
        return nn.GELU()
    
    if name == 'sigmoid':
        return nn.Sigmoid()
    
    if name == 'silu':
        return nn.SiLU()
    
    if name == 'mish':
        return nn.Mish()
    
    if name == 'softplus':
        return nn.Softplus()
    
    if name == 'softshrink':
        return nn.Softshrink()
    
    if name == 'softsign':
        return nn.Softsign()
    
    if name == 'tanh':
        return nn.Tanh()
    
    if name == 'tanhshrink':
        return nn.Tanhshrink()
    
    if name == 'threshold':
        return nn.Threshold()
    
    if name == 'glu':
        return nn.GLU()