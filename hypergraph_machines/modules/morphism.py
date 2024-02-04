import torch
from torch import nn
from hypergraph_machines.utils import l2_norm


class Morphism(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = None,
                 equivariance = None, prunable = True, origin = None,
                 destination = None):
        super(Morphism, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        params = self.get_equivariance_params(equivariance)
        self.equivariance = equivariance
        self.model = nn.Sequential(equivariance(*params))
        self.volume_buffer = None
        self.prunable = prunable
        self.origin = origin
        self.destination = destination
        self._pruned = False

    def get_equivariance_params(self, equivariance):
        if "Linear" in equivariance.__name__:
            return [self.in_ch, self.out_ch]
        elif "Conv" in equivariance.__name__:
            return [self.in_ch, self.out_ch, self.kernel_size]
        else:
            raise ValueError("Expected morphism are either dense or convolutional layers.")

    @property
    def pruned(self):
        return self._pruned

    def pruning_condition(self, tol):
        return torch.lt(self.weight_l2_norm(), tol) if not self.pruned else True

    def prune(self, tol):
        if self.prunable and self.pruning_condition(tol):
            self._pruned = True

    def weight_l1_norm(self):
        for i, param in enumerate(self.parameters()):
            if i == 0:
                norm = torch.sum(torch.abs(param))
            else:
                norm += torch.sum(torch.abs(param))

        return norm

    def weight_l2_norm(self):
        for i, param in enumerate(self.parameters()):
            if i == 0:
                norm = l2_norm(param)
            else:
                norm += l2_norm(param)

        return norm

    def forward(self, x):
        if "Linear" in self.equivariance.__name__:
            x = x.view(x.size(0), -1)
        return self.model(x)
