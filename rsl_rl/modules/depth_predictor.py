# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import torch.nn.functional as F
from dreamer import tools

class DepthPredictor(nn.Module):
    def __init__(self, forward_heightamp_dim = 525,
                 prop_dim = 33,
                 depth_image_dims = [64, 64],
                 encoder_hidden_dims=[256, 128],
                 depth=32,
                 act="ELU",
                 norm=True,
                 kernel_size=4,
                 minres=4,
                 outscale=1.0,
                 cnn_sigmoid=False,):


        # add this to fully recover the process of conv encoder
        h, w = depth_image_dims
        stages = int(np.log2(w) - np.log2(minres))
        self.h_list = []
        self.w_list = []
        for i in range(stages):
            h, w = (h+1) // 2, (w+1) // 2
            self.h_list.append(h)
            self.w_list.append(w)
        self.h_list = self.h_list[::-1]
        self.w_list = self.w_list[::-1]
        self.h_list.append(depth_image_dims[0])
        self.w_list.append(depth_image_dims[1])

        super(DepthPredictor, self).__init__()
        act = getattr(torch.nn, act)
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = len(self.h_list) - 1
        # layer_num = int(np.log2(shape[2]) - np.log2(minres))
        # self._minres = minres
        # out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        out_ch = self.h_list[0] * self.w_list[0] * depth * 2 ** (len(self.h_list) - 2)
        self._embed_size = out_ch

        in_dim = out_ch // (self.h_list[0] * self.w_list[0])
        out_dim = in_dim // 2

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(forward_heightamp_dim + prop_dim, encoder_hidden_dims[0]))
        encoder_layers.append(act())
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], self._embed_size))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(act())
        self.encoder = nn.Sequential(*encoder_layers)


        layers = []
        # h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = 1
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            if(self.h_list[i] * 2 == self.h_list[i+1]):
                pad_h, outpad_h = 1, 0
            else:
                pad_h, outpad_h = 2, 1

            if(self.w_list[i] * 2 == self.w_list[i+1]):
                pad_w, outpad_w = 1, 0
            else:
                pad_w, outpad_w = 2, 1

            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            # h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)


    def forward(self, forward_heightmap, prop):
        x = torch.concat((forward_heightmap, prop), dim=-1)
        x = self.encoder(x)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self.h_list[0], self.w_list[0], self._embed_size // (self.h_list[0] * self.w_list[0])]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        # print('init decoder shape:', x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape)
        x = self.layers(x)
        mean = x.permute(0, 2, 3, 1)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        # else:
        #     mean += 0.5
        return mean


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
