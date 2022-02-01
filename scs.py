import torch
import numpy as np 
import torch.nn.functional as F
import math 
from torch import nn 
import pdb

def unfold2d(x, kernel_size, stride, padding):
    ### using torch.nn.functional.unfold is also okay and the effiency will be compared later.

    x = F.pad(x, [padding]*4)
    bs, in_c, h, w = x.size()
    ks = kernel_size
    strided_x = x.as_strided((bs, in_c, (h - ks) // stride + 1, (w - ks) // stride + 1, ks, ks),
        (in_c * h * w, h * w, stride * w, stride, w, 1))
    return strided_x

class CosSim2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, stride=1,
        padding=0, eps=1e-12, bias=True):
        super(CosSim2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = padding

        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.xavier_normal_(w)
        self.w = nn.Parameter(w.view(out_channels, in_channels, -1), requires_grad=True)
        
        self.p = nn.Parameter(torch.empty(out_channels))
        nn.init.constant_(self.p, 2)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.constant_(self.bias, 0)
        else:
            self.bias = None
    
    def sigplus(self, x):
        return nn.Sigmoid()(x) * nn.Softplus()(x)
        
    def forward(self, x):
        x = unfold2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding) # nchwkk
        n, c, h, w, _, _ = x.shape
        x = x.reshape(n,c,h,w,-1)
        x = F.normalize(x, p=2.0, dim=-1, eps=self.eps)

        w = F.normalize(self.w, p=2.0, dim=-1, eps=self.eps)
        x = torch.einsum('nchwl,vcl->nvhw', x, w)
        sign = torch.sign(x)

        x = torch.abs(x) + self.eps
        x = x.pow(self.sigplus(self.p).view(1, -1, 1, 1))
        # pdb.set_trace()
        x = sign * x

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x

if __name__ == '__main__':
    layer = CosSim2d(4, 8, 7, 2, padding=3).cuda()
    data = torch.randn(10, 4, 128, 128).cuda()
    print(layer(data).shape)
