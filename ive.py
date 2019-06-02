# modified based on https://github.com/nicola-decao/s-vae-pytorch
import torch
import numpy as np
import scipy.special
from numbers import Number


class IveFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, z):
        assert isinstance(v, Number), 'v must be a scalar'
        ctx.v = v
        ctx.save_for_backward(z)

        z_cpu = z.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=np.double)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=np.double)
        else:
            output = scipy.special.ive(v, z_cpu, dtype=np.double)

        if z.is_cuda:
            output = torch.from_numpy(output).cuda()
        else:
            output = torch.from_numpy(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[-1].double()
        return None, (grad_output.double() *
                      (ive(ctx.v - 1, z) - ive(ctx.v, z) *
                      torch.autograd.Variable((ctx.v + z) / z))).float()


class Ive(torch.nn.Module):

    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)


ive = IveFunction.apply
