import torch.nn as nn
class KLGLoss(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.gconv = nn.Conv2d(dim,3,1)
    def forward(self,x):
        x = x.permute(0,3,1,2).contiguous()
        # print('gconv',x.size())
        x = self.gconv(x)
        return x

class KLLLoss(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lconv = nn.Conv2d(dim, 3, 1)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        # print('lconv', x.size())
        x = self.lconv(x)
        return x
