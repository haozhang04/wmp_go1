# Copyright (c) 2026-2027 zh
"""
内容：
    可部署 WMP 策略，将actor与world model组件导出为单模块。
"""

import copy
import os

import torch
import torch.nn.functional as F
from torch import nn


class DeploymentPolicy(nn.Module):
    """Stateful end-to-end WMP policy for one deployed robot.

    - obs_45: 45-dimensional observations with the same scaling and ordering as training.
    - Observation order: angular velocity (3), projected gravity (3), commands (3),
      joint position error (12), joint velocity (12), and previous applied action (12).
    - depth: a normalized 64x64 depth image, approximately in [-0.5, 0.5], following the training convention.
    - Depth normalization is performed outside the exported policy to match the input contract of play.py.
    """

    def __init__(
        self,
        actor_critic,
        world_model,
        update_interval,
        sample_rssm_state=True,
    ):
        super().__init__()
        if not world_model.encoder.use_camera:
            raise ValueError("End-to-end deployment export requires a camera-enabled World Model")
        if not world_model.dynamics._discrete:
            raise ValueError("DeploymentPolicy currently supports only the discrete RSSM used by this project")

        self.update_interval = int(update_interval)
        if self.update_interval < 1:
            raise ValueError("update_interval must be positive")
        if world_model.dynamics._rec_depth != 1:
            raise ValueError("DeploymentPolicy requires RSSM rec_depth=1")
        self.sample_rssm_state = bool(sample_rssm_state)
        self.rssm_stoch = int(world_model.dynamics._stoch)
        self.rssm_discrete = int(world_model.dynamics._discrete)
        self.rssm_unimix_ratio = float(world_model.dynamics._unimix_ratio)

        # Inference-only copies of the four deployed network blocks.
        self.image_encoder = copy.deepcopy(world_model.encoder._cnn.layers).cpu()
        self.prop_encoder = copy.deepcopy(world_model.encoder._mlp.layers).cpu()
        self.rssm_img_in = copy.deepcopy(world_model.dynamics._img_in_layers).cpu()
        self.rssm_cell = copy.deepcopy(world_model.dynamics._cell.layers).cpu()
        self.rssm_obs_out = copy.deepcopy(world_model.dynamics._obs_out_layers).cpu()
        self.rssm_obs_stat = copy.deepcopy(world_model.dynamics._obs_stat_layer).cpu()
        self.history_encoder = copy.deepcopy(actor_critic.history_encoder).cpu()
        self.wm_feature_encoder = copy.deepcopy(actor_critic.wm_feature_encoder).cpu()
        self.actor = copy.deepcopy(actor_critic.actor).cpu()

        with torch.no_grad():
            initial_state = world_model.dynamics.initial(1)
        initial_deter = initial_state["deter"].detach().cpu()
        initial_stoch = initial_state["stoch"].detach().cpu()

        self.register_buffer("initial_deter", initial_deter)
        self.register_buffer("initial_stoch", initial_stoch)
        self.register_buffer("rssm_deter_state", initial_deter.clone())
        self.register_buffer("rssm_stoch_state", initial_stoch.clone())
        self.register_buffer("wm_feature", torch.zeros_like(initial_deter))
        self.register_buffer("history_buffer", torch.zeros(1, 5, 42))
        self.register_buffer("action_history", torch.zeros(1, self.update_interval, 12))
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))
        self.register_buffer("is_first", torch.ones((), dtype=torch.bool))

        self.eval()

    @torch.jit.export
    def reset(self):
        """Reset all temporal state before startup or after a robot reset."""
        self.rssm_deter_state.copy_(self.initial_deter)
        self.rssm_stoch_state.copy_(self.initial_stoch)
        self.wm_feature.zero_()
        self.history_buffer.zero_()
        self.action_history.zero_()
        self.step_counter.zero_()
        self.is_first.fill_(True)

    def _encode_observation(self, prop, depth):
        # ConvEncoder.forward subtracts another 0.5 in the training implementation.
        image_embedding = self.image_encoder(depth - 0.5).flatten(1)
        prop_embedding = self.prop_encoder(torch.sign(prop) * torch.log(torch.abs(prop) + 1.0))
        return torch.cat((image_embedding, prop_embedding), dim=-1)

    def _sample_stoch(self, logits):
        probs = torch.softmax(logits, dim=-1)
        probs = probs * (1.0 - self.rssm_unimix_ratio) + self.rssm_unimix_ratio / float(self.rssm_discrete)
        flat_probs = probs.reshape(-1, self.rssm_discrete)
        if self.sample_rssm_state:
            indices = torch.multinomial(flat_probs, 1).squeeze(-1)
        else:
            indices = torch.argmax(flat_probs, dim=-1)
        return F.one_hot(indices, self.rssm_discrete).to(logits.dtype).reshape(logits.shape[0], self.rssm_stoch, self.rssm_discrete)

    def _update_world_model(self, embed):
        previous_action = self.action_history.flatten(1)
        if bool(self.is_first.item()):
            previous_action = torch.zeros_like(previous_action)

        previous_stoch = self.rssm_stoch_state.flatten(1)
        rssm_input = self.rssm_img_in(torch.cat((previous_stoch, previous_action), dim=-1))

        cell_parts = self.rssm_cell(torch.cat((rssm_input, self.rssm_deter_state), dim=-1))
        reset, candidate, update = torch.chunk(cell_parts, 3, dim=-1)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update - 1.0)
        deter = update * candidate + (1.0 - update) * self.rssm_deter_state

        obs_hidden = self.rssm_obs_out(torch.cat((deter, embed), dim=-1))
        logits = self.rssm_obs_stat(obs_hidden).reshape(obs_hidden.shape[0], self.rssm_stoch, self.rssm_discrete)
        stoch = self._sample_stoch(logits)

        self.rssm_deter_state.copy_(deter.detach())
        self.rssm_stoch_state.copy_(stoch.detach())
        self.wm_feature.copy_(deter.detach())
        self.is_first.fill_(False)

    def _prepare_inputs(self, obs_45, depth):
        if obs_45.dim() == 1:
            obs_45 = obs_45.unsqueeze(0)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)

        torch._assert(obs_45.shape[0] == 1 and obs_45.shape[1] == 45, "obs_45 must have shape (1, 45)")
        torch._assert(
            depth.shape[0] == 1
            and depth.shape[1] == 1
            and depth.shape[2] == 64
            and depth.shape[3] == 64,
            "depth must have shape (1, 1, 64, 64)",
        )
        return obs_45, depth

    def _forward_training_depth(self, obs_45, depth):
        prop = obs_45[:, :33]
        history_frame = torch.cat((obs_45[:, :6], obs_45[:, 9:45]), dim=-1)
        history = torch.cat((self.history_buffer[:, 1:], history_frame.unsqueeze(1)), dim=1)
        self.history_buffer.copy_(history.detach())

        if int(self.step_counter.item()) % self.update_interval == self.update_interval - 1:
            embed = self._encode_observation(prop, depth)
            self._update_world_model(embed)

        history_feature = self.history_encoder(self.history_buffer.flatten(1))
        wm_feature = self.wm_feature_encoder(self.wm_feature)
        actor_input = torch.cat((history_feature, obs_45[:, 6:9], wm_feature), dim=-1)
        action = self.actor(actor_input).detach()

        self.action_history.copy_(torch.cat((self.action_history[:, 1:], action.unsqueeze(1)), dim=1))
        self.step_counter.add_(1)
        return action

    def forward(self, obs_45, depth):
        obs_45, depth = self._prepare_inputs(obs_45, depth)
        return self._forward_training_depth(obs_45, depth)

    @torch.jit.export
    def forward_normalized_depth(self, obs_45, depth):
        """Backward-compatible alias for forward()."""
        return self.forward(obs_45, depth)


def compare_deployment_policy_error(
    actor_critic,
    world_model,
    update_interval,
    num_steps=None,
    seed=0,
):
    """Compare the deployment wrapper against the original play.py-style path.

    The comparison is deterministic: RSSM posterior states use the categorical
    mode, so a correct export should be close to floating point noise.
    """
    if num_steps is None:
        num_steps = int(update_interval) * 2 + 3

    was_actor_training = actor_critic.training
    was_wm_training = world_model.training
    actor_critic.eval()
    world_model.eval()

    ref_device = next(world_model.parameters()).device
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    deployment_policy = DeploymentPolicy(
        actor_critic=actor_critic,
        world_model=world_model,
        update_interval=update_interval,
        sample_rssm_state=False,
    )
    deployment_policy.reset()

    with torch.no_grad():
        wm_latent = None
        wm_action = None
        wm_is_first = torch.ones(1, device=ref_device)
        wm_feature = torch.zeros((1, deployment_policy.rssm_deter_state.shape[-1]), device=ref_device)
        action_history = torch.zeros((1, int(update_interval), 12), device=ref_device)
        history_buffer = torch.zeros((1, 5, 42), device=ref_device)

        max_abs = 0.0
        mean_abs_sum = 0.0

        for step in range(int(num_steps)):
            obs_45_cpu = torch.randn((1, 45), generator=generator)
            depth_cpu = torch.rand((1, 1, 64, 64), generator=generator) - 0.5
            obs_45 = obs_45_cpu.to(ref_device)
            depth = depth_cpu.to(ref_device)

            history_frame = torch.cat((obs_45[:, :6], obs_45[:, 9:45]), dim=-1)
            history_buffer = torch.cat((history_buffer[:, 1:], history_frame.unsqueeze(1)), dim=1)

            if step % int(update_interval) == int(update_interval) - 1:
                wm_obs = {
                    "prop": obs_45[:, :33],
                    "is_first": wm_is_first,
                    "image": depth.permute(0, 2, 3, 1),
                }
                wm_embed = world_model.encoder(wm_obs)
                wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_is_first, sample=False)
                wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
                wm_is_first[:] = 0

            history_feature = actor_critic.history_encoder(history_buffer.flatten(1))
            wm_latent_vector = actor_critic.wm_feature_encoder(wm_feature)
            actor_input = torch.cat((history_feature, obs_45[:, 6:9], wm_latent_vector), dim=-1)
            ref_action = actor_critic.actor(actor_input)

            export_action = deployment_policy(obs_45_cpu, depth_cpu).to(ref_device)
            error = (export_action - ref_action).abs()
            if not torch.isfinite(error).all():
                raise RuntimeError("Non-finite deployment parity error")
            max_abs = max(max_abs, float(error.max().item()))
            mean_abs_sum += float(error.mean().item())

            action_history = torch.cat((action_history[:, 1:], ref_action.unsqueeze(1)), dim=1)
            wm_action = action_history.flatten(1)

    if was_actor_training:
        actor_critic.train()
    if was_wm_training:
        world_model.train()

    return {
        "num_steps": int(num_steps),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs_sum / float(num_steps),
    }


def export_deployment_policy(
    actor_critic,
    world_model,
    path,
    filename,
    update_interval,
    sample_rssm_state=True,
    compare_error=True,
):
    """Build, validate, and save the stateful end-to-end TorchScript policy."""
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, filename)

    deployment_policy = DeploymentPolicy(
        actor_critic=actor_critic,
        world_model=world_model,
        update_interval=update_interval,
        sample_rssm_state=sample_rssm_state,
    )
    scripted = torch.jit.script(deployment_policy)

    # Exercise both the 50 Hz path and the 10 Hz World Model update path.
    with torch.inference_mode():
        dummy_obs = torch.zeros(1, 45)
        dummy_depth = torch.zeros(1, 1, 64, 64)
        scripted.reset()
        for _ in range(update_interval):
            dummy_action = scripted(dummy_obs, dummy_depth)
        if tuple(dummy_action.shape) != (1, 12):
            raise RuntimeError(f"Unexpected deployment action shape: {tuple(dummy_action.shape)}")
        scripted.reset()

    if compare_error:
        metrics = compare_deployment_policy_error(
            actor_critic=actor_critic,
            world_model=world_model,
            update_interval=update_interval,
        )
        print(
            "Deployment parity vs play path: "
            f"steps={metrics['num_steps']} "
            f"max_abs={metrics['max_abs_error']:.6e} "
            f"mean_abs={metrics['mean_abs_error']:.6e}"
        )
        if metrics["max_abs_error"] > 1e-4:
            raise RuntimeError(f"Deployment parity check failed: {metrics['max_abs_error']:.6e}")

    scripted.save(output_path)
    return output_path
