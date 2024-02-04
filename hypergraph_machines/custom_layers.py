from torch import nn


class Conv1d_pad(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(Conv1d_pad, self).__init__(in_channels, out_channels, kernel_size,
                                         bias=bias)
        self.padding =  self.kernel_size // 2

class Conv2d_pad(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(Conv2d_pad, self).__init__(in_channels, out_channels, kernel_size,
                                         bias=bias)
        self.padding =  tuple([s // 2 for s  in self.kernel_size])

    def __repr__(self):
        return "{}\nkernel size{}".format("Conv2d", self.kernel_size)
    
class Conv3d_pad(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(Conv3d_pad, self).__init__(in_channels, out_channels, kernel_size,
                                         bias=bias)
        self.padding =  tuple([s // 2 for s  in self.kernel_size])

    def __repr__(self):
        return "{}\nkernel size{}".format("Conv2d", self.kernel_size)


class LinearNoBias(nn.Linear):
    def __init__(self, in_features, out_features):
        super(LinearNoBias, self).__init__(in_features, out_features,
                                           bias=False)
        self.equivariance = "identity"

    def __repr__(self):
        return "Linear, no bias"
    
