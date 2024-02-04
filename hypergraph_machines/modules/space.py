import torch
from torch import nn


class Space(nn.Module):
    def __init__(self, in_size, out_size, num_channels, incoming_morphisms = [],
                 activations = None, index = None, prunable = True,
                 is_input = False, is_output = False):
        super(Space, self).__init__()
        self.incoming = nn.ModuleList(incoming_morphisms)
        self.in_size = in_size
        self.out_size = out_size
        self.num_channels = num_channels
        if isinstance(self.out_size, int): 
            self.out_size = [self.out_size]
        self.size = (self.num_channels,) + tuple(self.out_size)
        activations = activations or [nn.Identity()]
        self.activation = nn.Sequential(*activations)
        self.index = index
        self._depth = 0
        self._pruned = False
        self.prunable = prunable
        self.is_output = is_output
        self.is_input = is_input

    @property
    def pruned(self):
        if not self.prunable:
            return torch.BoolTensor([False])
        pruned_incoming = torch.BoolTensor([m.pruned for m in self.incoming])
        return pruned_incoming.all()

    @property
    def depth(self):
        return self._depth

    def add_incoming(self, morphisms):
        self.incoming += morphisms

    def forward(self, spaces):
        out = None

        for i, morphism in enumerate(self.incoming):
            space = spaces[morphism.origin]
            if not morphism.pruned:
                if i == 0 or out is None:
                    out = morphism.forward(space.volume_buffer)
                else:
                    out += morphism.forward(space.volume_buffer)

        out = self.activation(out)
        self.volume_buffer = out
        return out

    def __repr__(self):
        return "ind: {}\nact:\t{}\nio_size\t{} -> {}".format(self.index, 
                                                             self.activation,
                                                             self.in_size,
                                                             self.out_size)

