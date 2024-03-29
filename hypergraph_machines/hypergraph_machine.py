from torch import nn
import torch.nn.functional as F
import numpy as np
from hypergraph_machines.modules import Space, Morphism
from hypergraph_machines.utils import get_graph
from hypergraph_machines.sampler import ParamSampler
from hypergraph_machines.custom_layers import LinearNoBias


class HgMachine(nn.Module):
    def __init__(self, input_size, number_of_spaces=10, num_channels = 4, 
                 kernel_size=3, limit_upsample = 3, number_of_input_spaces=1,
                 tol = 1e-6, output_dim = 10, prune = True,
                 out_activation=None, out_activation_params={}):
        super(HgMachine, self).__init__()
        self.number_of_spaces = number_of_spaces
        self.input_size = input_size #(ch, h, w)
        self.sizes = [self.input_size[1:]]
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.tol = tol
        self.number_of_input_spaces = number_of_input_spaces
        self.output_dim = output_dim
        self.limit_upsample = limit_upsample
        self.prune_it = prune
        self.out_activation = out_activation
        self.out_activation_params = out_activation_params
        self.sampler = ParamSampler(self.dim)
        self.spaces = nn.ModuleList([])
        self.build_spaces()
        
    @property
    def input_channels(self):
        return self.input_size[0]
    
    @property
    def dim(self):
        return len(self.input_size[1:])
        
    @property
    def out_activation(self):
        return self._out_activation
    
    @out_activation.setter
    def out_activation(self, val):
        if val is not None:
            self._out_activation = lambda x: val(x, **self.out_activation_params)
        else:
            self._out_activation = lambda x: x

    def resample_cond(self, out_size):
        cond1 = out_size[0] > self.input_size[1] * self.limit_upsample
        cond2 = out_size[0] < self.input_size[1] / self.limit_upsample
        return cond1 or cond2
        
    def sample_activation(self):
        lim = True

        while lim:
            rand_ind = np.random.randint(len(self.sizes))
            in_size = self.sizes[rand_ind]
            res, in_size, out_size = self.sampler.resampler(in_size)
            lim = self.resample_cond(out_size)

        act = self.sampler.activation()
        return [act, res], in_size, out_size


    def get_input_spaces(self):
        for i in range(self.number_of_input_spaces):
            self.spaces.append(Space(self.sizes[0], self.sizes[0], index = i,
                                     num_channels = self.num_channels,
                                     prunable=False, is_input = True))

        [setattr(s, "_depth", 0) for s in self.spaces]

    def get_space(self, index):
        ar, in_size, out_size = self.sample_activation()
        self.sizes.extend([in_size, out_size])
        space = Space(in_size, out_size, incoming_morphisms = [],
                      activations = ar, index = index,
                      num_channels = self.num_channels)
        self.set_incoming_morphisms(space)
        return space

    def set_spaces(self):
        num_inputs = self.number_of_input_spaces

        for i in range(num_inputs, self.number_of_spaces + num_inputs):
            space = self.get_space(i)
            self.spaces.append(space)

    def build_spaces(self):
        self.get_input_spaces()
        self.set_spaces()
        self.build_output_embedding()
        self.graph = get_graph(self)

    def init_out_spaces(self, space):
        out_shape = ((space.num_channels,) + tuple(space.out_size))
        return self.get_output_space(np.prod(out_shape), self.output_dim,
                                     space.index, space.index)

    def get_output_morphism(self, in_features, out_features, i, j):
        return Morphism(in_features, out_features, equivariance=LinearNoBias,
                        origin = i, destination = j)

    def get_output_space(self, in_features, out_features, i, j):
        index = j + self.number_of_spaces
        s = Space(in_features, out_features, 1, activations=[nn.Identity()],
                  index = index, is_output = True)
        s.add_incoming([self.get_output_morphism(in_features, out_features, i,
                                                  index)])
        self.set_depth(s)
        return s

    def build_output_embedding(self):
        in_spaces = self.number_of_input_spaces
        out_spaces = [self.init_out_spaces(s) for s in self.spaces[in_spaces:]]
        self.output_spaces = nn.ModuleList(out_spaces)

    def set_incoming_morphisms(self, space):
        inc = []

        for j, s in enumerate(self.spaces):
            if np.all(s.out_size == space.in_size):
                if j < self.number_of_input_spaces:
                    num_in_ch = self.input_channels
                    prunable = False
                else:
                    num_in_ch = self.num_channels
                    prunable = True
                m = Morphism(num_in_ch, self.num_channels, kernel_size=self.kernel_size,
                             equivariance = self.sampler.morphism(), prunable = prunable,
                             origin = j, destination = space.index)
                inc.append(m)

        space.add_incoming(inc)
        self.set_depth(space)

    def get_outgoing(self, space):
        spaces = [m for s in self.spaces for m in s.incoming
                  if m.origin == space.index]
        output = [m for s in self.output_spaces for m in s.incoming
                  if m.origin == space.index]
        return spaces + output

    def set_depth(self, space):
        depths = [self.spaces[morphism.origin].depth for morphism in
                     space.incoming]
        depth = max(depths)
        setattr(space, "_depth", depth + 1)

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        x, y = inputs, None

        for i, space in enumerate(self.spaces[:self.number_of_spaces]):
            if i < self.number_of_input_spaces:
                space.volume_buffer = x[i]
            elif not space.pruned and i >= self.number_of_input_spaces:
                x = space.forward(self.spaces)
                ind = space.index - self.number_of_input_spaces
                if not self.output_spaces[ind].pruned:
                    if y is None:
                        y = self.output_spaces[ind].forward(self.spaces)
                    else:
                        y += self.output_spaces[ind].forward(self.spaces)
        
        out = self.out_activation(y)
        
        if self.prune_it:
            self.set_depth_and_prune(self.spaces[self.number_of_input_spaces:])
            self.set_depth_and_prune(self.output_spaces)
        
        return out 

    def set_depth_and_prune(self, space_list):
        for space in space_list:
            self.set_depth(space)
            self.prune(space)

    def prune(self, space):
        [m.prune(self.tol) for m in space.incoming if not m.pruned]
        outgoing = self.get_outgoing(space)
        if (len(space.incoming) == 0 or space.pruned) and not space.is_input:
            [setattr(m, "_pruned", True) for m in outgoing]
        elif len(outgoing) == 0 or len([m for m in outgoing if not m.pruned])==0:
            [setattr(m, "_pruned", True) for m in space.incoming
             if not space.is_output]

    def get_space_by_index(self, index):
        s1 = [s for s in self.spaces if s.index == index]
        s2 = [s for s in self.output_spaces if s.index == index]
        return (s1+s2)[0]


class HgClassifier(HgMachine):
    def __init__(self, input_size, **kwargs):
        kwargs['out_activation']=F.log_softmax
        kwargs['out_activation_params']={"dim":1}
        super(HgClassifier, self).__init__(input_size, **kwargs)


class HgPredictor(HgMachine):
    def __init__(self, input_size, **kwargs):
        kwargs['out_activation']=nn.Identity()
        kwargs['out_activation_params']={}
        super(HgClassifier, self).__init__(input_size, **kwargs)