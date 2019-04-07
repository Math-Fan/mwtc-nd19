# This file creates Pytorch modules useful in constructing features
import torch

# returns a product of powers of the inputs
class ProductLayer(torch.nn.Module):
    def __init__(self, d_in, d_out, scalar_mult=False):
        super(ProductLayer, self).__init__()
        self.layer = torch.nn.Linear(d_in, d_out, bias=scalar_mult)

    # pseudo-logarithm function that is used for multiplying absolute values
    # without producing huge negative values for the logarithm
    def plog(self, input_tensor):
        # eliminate values too close to 0
        nonzero = input_tensor.abs().clamp(1e-5)
        return nonzero.log()

    def forward(self, input):
        input = self.plog(input)
        inter = self.layer(input)
        output = inter.exp()
        return output

class LinearLayer(torch.nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super(LinearLayer, self).__init__()
        self.layer = torch.nn.Linear(d_in, d_out, bias)

    def forward(self, input):
        output = self.layer(input)
        return output

# some features you might not have to take the log of to be predictive, so concatenate logs with standard
class LogOutput(torch.nn.Module):
    def __init__(self):
        super(LogOutput, self).__init__()

    def forward(self, input):
        # output = torch.cat( (input, input.log()) , dim=1)
        output = input.log()
        return output
