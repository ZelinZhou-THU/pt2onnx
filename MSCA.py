# MSCAAttention — derived from SegNeXt (NeurIPS 2022)
# Copyright 2022 SegNeXt Authors (Meng-Hao Guo et al.)
# Source: https://github.com/Visual-Attention-Network/SegNeXt
# Originally licensed under the Apache License, Version 2.0.
#
# This file has been modified from the original:
#   - Extracted MSCAAttention class only
#   - Removed dependencies on mmseg / mmcv framework
#   - Adapted for standalone usage as a custom YOLO module
#
# A copy of the Apache License, Version 2.0 is included in the NOTICE file.
#
# As part of the pt2onnx-converter project, the overall combined work is
# licensed under AGPL-3.0 (see LICENSE file). The original Apache 2.0
# terms for this file continue to apply per Section 4 of the Apache License.
#
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
from torch.nn import functional as F


class MSCAAttention(nn.Module):
    # SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u
