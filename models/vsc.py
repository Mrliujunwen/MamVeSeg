import torch
import torch.nn as nn
from timm.models.layers import  DropPath, trunc_normal_

class ChannelCompressAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=256, window=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.conv1=nn.Conv2d(dim, 1, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c
        agent_tokens = self.conv1(q[:, :, :].reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, h*w, -1)
        agent_tokens1 = agent_tokens.permute(0,2,1)
        # print(agent_tokens1.shape)

        k = k.reshape(b, n,  head_dim).permute(0, 1, 2)
        v = v.reshape(b, n,  head_dim).permute(0, 1, 2)

        agent_attn = self.softmax((agent_tokens1 * self.scale) @ k)

        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v.transpose(-1, -2)
        agent_v = self.softmax(agent_v)
        agent_v = agent_v@v

        agent_tokens = self.softmax(agent_tokens)

        x=agent_tokens@agent_v

        return x
if __name__ == '__main__':
    attn = ChannelCompressAttention(dim=96, window_size=(16,16), num_heads=1,
                               qkv_bias=True, qk_scale=True, attn_drop=True, proj_drop=0,
                               agent_num=256)
    imput=torch.rand(1,16384,96)
    out=attn(imput)
    print(out.shape)