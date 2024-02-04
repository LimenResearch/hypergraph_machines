from torch import nn
import numpy as np
from hypergraph_machines.custom_layers import LinearNoBias, Conv1d_pad, Conv2d_pad, Conv3d_pad

EQ_MORPHS = {
    0: [LinearNoBias],
    1: [Conv1d_pad],
    2: [Conv2d_pad],
    3: [Conv3d_pad],
}
ACTIVATIONS = [nn.ReLU6(inplace=False)]
RESAMPLING = {
    0: [nn.Identity()],
    1: [nn.Identity(), nn.MaxPool1d(2), nn.Upsample(scale_factor=2)],
    2: [nn.Identity(), nn.MaxPool2d(2), nn.Upsample(scale_factor=2)],
    3: [nn.Identity(), nn.MaxPool3d(2), nn.Upsample(scale_factor=2)]
}

class ParamSampler:
    def __init__(self, dim):
        self.dim = dim
        self.activations = ACTIVATIONS
        self.resamplers = RESAMPLING[dim]
        self.morphisms = EQ_MORPHS[dim]
        
    def activation(self):
        return np.random.choice(self.activations)
    
    def morphism(self):
        rand_ind = np.random.randint(len(self.morphisms))
        return self.morphisms[rand_ind]
    
    def resampler(self, in_size):
        if isinstance(in_size, (int, list, tuple)):
            in_size = np.array(in_size)
            
        res = np.random.choice(self.resamplers)
        out_size = (self.get_coeff(res)*np.asarray(in_size)).astype(int)
        return res, in_size, out_size
    
    @staticmethod
    def get_coeff(res):
        if "MaxPool" in res._get_name():
            c = .5
        elif "Upsampl" in res._get_name():
            c = 2 
        else:
            c = 1
        return c
        
        