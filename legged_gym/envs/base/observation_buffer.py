# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch

class ObservationBuffer:
    def __init__(self, num_envs, num_obs, include_history_steps, device):

        self.num_envs = num_envs
        self.num_obs = num_obs
        self.include_history_steps = include_history_steps
        self.device = device

        self.num_obs_total = num_obs * include_history_steps

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs_total, device=self.device, dtype=torch.float)

    def reset(self, reset_idxs, new_obs):
        self.obs_buf[reset_idxs] = new_obs.repeat(1, self.include_history_steps)

    def insert(self, new_obs):
        # Shift observations back.
        self.obs_buf[:, : self.num_obs * (self.include_history_steps - 1)] = self.obs_buf[:,self.num_obs : self.num_obs * self.include_history_steps]

        # Add new observation.
        self.obs_buf[:, -self.num_obs:] = new_obs

    def get_obs_vec(self, obs_ids):
        """Gets history of observations indexed by obs_ids.
        
        Arguments:
            obs_ids: An array of integers with which to index the desired
                observations, where 0 is the latest observation and
                include_history_steps - 1 is the oldest observation.
        """

        obs = []
        for obs_id in reversed(sorted(obs_ids)):
            slice_idx = self.include_history_steps - obs_id - 1
            obs.append(self.obs_buf[:, slice_idx * self.num_obs : (slice_idx + 1) * self.num_obs])
        return torch.cat(obs, dim=-1)

