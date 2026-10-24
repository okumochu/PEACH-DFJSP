import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from src.utils.training_settings import ValidationStopController, require_training_controls
import numpy as np
import time
import subprocess
import os
import random
import sys
import math
import json
import hashlib
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import wandb

from .hgt_network import HGT
from .hgt_env import FJSPEnvHGT, HGTEnvState
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_generator import create_data_generator
from src.utils.checkpoint_util import (
    checkpoint_safe_metadata, training_contract, validate_resume_contract)
from src.utils.config_util import resolve_dp_lb_diff_enabled, resolve_dp_lb_measure, put_range_aliases
from src.utils.training_evidence import TrainingJournal, source_identity, atomic_bytes, json_value
from src.utils.reproducibility import derive_seed, seed_context, set_global_seed
from src.utils.rl_utils import sample_action, eval_actions, greedy_select_action
from src.utils.event_utils import merge_and_sort_events
from src.utils.scale_curriculum import (
    ScaleCurriculumController, ADVANCE, DIVERGED, STUCK, DONE, DESCEND)
from src.utils.segment_util import (
    pop_next_event_set, freeze_segment_ops, absorb_breakdowns)
from src.utils.gantt_util import extract_schedule_from_env




class Memory:
    """Memory for PPO training following DANIEL structure"""
    
    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Input variables for HGT
        self.candidate_mask_seq = []  # [N, tensor[sz_b, J, M]]
        self.candidate_seq = []  # [N, tensor[sz_b, J]]
        self.candidate_pt_indices_seq = []  # [N, tensor[sz_b, J, M]] PT node indices
        self.graph_seq = []  # [N, HeteroData] - Additional for HGT
        
        # Other variables
        self.action_seq = []  # action index with shape [N, tensor[sz_b]]
        self.reward_seq = []  # reward value with shape [N, tensor[sz_b]]
        self.val_seq = []  # state value with shape [N, tensor[sz_b]]
        self.log_probs = []  # log(p_{\theta_old}(a_t|s_t)) with shape [N, tensor[sz_b]]
        # Per-step event-boundary flags ([N, tensor[sz_b]], 1.0 where a dynamic
        # event fired). Used only by the legacy single-trajectory (static /
        # inline) GAE path; the anchored-segment path does not need it.
        self.event_boundary_seq = []

        # Anchored-segment rollout bookkeeping (empty for the legacy path).
        # ``active_mask_seq`` is a per-step [sz_b] bool: True where the env took
        # a real action this step (False once it finished its segment, since
        # segments seeded from different-size partial schedules end at different
        # steps). ``segment_lengths`` records each segment's step count so GAE
        # runs per-segment — each env's last active step in a segment is a TRUE
        # terminal, so value targets never bootstrap across a dynamic event.
        self.active_mask_seq = []
        self.segment_lengths = []
        # Flat [B*N] bool of valid (active) samples, set by get_gae_advantages();
        # PPO.update() filters minibatches to these. None -> all valid.
        self.active_flat = None

    def clear_memory(self):
        self.clear_state()
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.log_probs[:]
        del self.event_boundary_seq[:]
        del self.active_mask_seq[:]
        del self.segment_lengths[:]
        self.active_flat = None
    
    def clear_state(self):
        del self.candidate_mask_seq[:]
        del self.candidate_seq[:]
        del self.candidate_pt_indices_seq[:]
        del self.graph_seq[:]
    
    def push(self, state: HGTEnvState):
        """Push a state into the memory"""
        self.candidate_mask_seq.append(state.candidate_mask_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.candidate_pt_indices_seq.append(state.candidate_pt_indices)
        self.graph_seq.append(state.graph_tensor)
    
    @staticmethod
    def _pad_seq_to_jmax(seq, fill):
        """Right-pad each [B, J, ...] tensor in ``seq`` along the job axis to the
        max J across the sequence. Anchored segments collected after a job
        arrival have a larger J than earlier segments; padding lets them stack.
        The HGT actor masks padded slots (candidate_mask=1, pt_idx=-1) and
        zero-pads job embeddings, and the flattened action index ``job*M+mch``
        is invariant to J, so padding does not affect the policy. No-op (returns
        ``seq``) when all entries already share J (the rectangular path)."""
        if not seq:
            return seq
        j_max = max(t.shape[1] for t in seq)
        if all(t.shape[1] == j_max for t in seq):
            return seq
        out = []
        for t in seq:
            if t.shape[1] == j_max:
                out.append(t)
                continue
            pad_shape = (t.shape[0], j_max - t.shape[1]) + tuple(t.shape[2:])
            pad = torch.full(pad_shape, fill, dtype=t.dtype, device=t.device)
            out.append(torch.cat([t, pad], dim=1))
        return out

    def transpose_data(self):
        """Transpose the first and second dimension of collected variables"""
        # Pad ragged-J segments (arrivals grow the job count) so they stack.
        cmask_seq = self._pad_seq_to_jmax(self.candidate_mask_seq, fill=1.0)
        cand_seq = self._pad_seq_to_jmax(self.candidate_seq, fill=0)
        cpt_seq = self._pad_seq_to_jmax(self.candidate_pt_indices_seq, fill=-1)
        t_candidateMask_seq = torch.stack(cmask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_seq = torch.stack(cand_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_pt_indices_seq = torch.stack(cpt_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)
        
        # Flatten graph sequence to align EXACTLY with tensor flattening order (env-major, then time)
        # Original tensors are stacked as [T, B, ...] -> transpose to [B, T, ...] -> flatten(0,1)
        # So the order is: for each env b in 0..B-1, for each t in 0..T-1
        t_graph_seq = []
        if len(self.graph_seq) > 0:
            num_envs = len(self.graph_seq[0])  # Number of environments (B)
            num_time = len(self.graph_seq)      # Number of time steps (T)
            for env_idx in range(num_envs):
                for time_step in range(num_time):
                    t_graph_seq.append(self.graph_seq[time_step][env_idx])
        
        return (t_candidateMask_seq,
                t_candidate_seq, t_candidate_pt_indices_seq, t_graph_seq,
                t_action_seq, t_reward_seq, t_val_seq, t_logprobs_seq)
    
    def get_gae_advantages(self):
        """Compute GAE advantages and value targets.

        Dispatches on whether this rollout was collected as anchored segments
        (``segment_lengths`` populated) or as a single trajectory (legacy
        static / inline path). Sets ``self.active_flat`` so PPO.update() can
        drop masked (inactive) samples.
        """
        if self.segment_lengths:
            return self._segmented_gae()
        self.active_flat = None
        return self._single_trajectory_gae()

    def _single_trajectory_gae(self):
        """Legacy single-trajectory GAE (used by the static / inline path).

        Event-boundary truncation: at steps where a dynamic event fired,
        ``event_boundary_seq`` is 1.0, zeroing the GAE carry term. For the
        static path the mask is all-zero, so this is plain GAE.
        """
        reward_arr = torch.stack(self.reward_seq, dim=0)
        t_old_val_seq = self.t_old_val_seq  # reuse from transpose_data()
        values = t_old_val_seq.transpose(0, 1)
        len_trajectory, len_envs = reward_arr.shape

        if len(self.event_boundary_seq) == len_trajectory:
            boundary = torch.stack(self.event_boundary_seq, dim=0).to(
                device=values.device, dtype=reward_arr.dtype)
        else:
            boundary = torch.zeros(
                len_trajectory, len_envs, device=values.device, dtype=reward_arr.dtype)

        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []
        for i in reversed(range(len_trajectory)):
            if i == len_trajectory - 1:
                delta_t = reward_arr[i] - values[i].squeeze()
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1].squeeze() - values[i].squeeze()
            advantage = delta_t + self.gamma * self.gae_lambda * (1.0 - boundary[i]) * advantage
            advantage_seq.append(advantage)
        advantage_seq.reverse()

        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)
        v_target_seq = (t_advantage_seq + t_old_val_seq).flatten(0, 1)

        # A one-transition episode has no within-episode variation. Keep its
        # centered advantage zero without evaluating an undefined sample std.
        # Multi-transition episodes retain the historical sample convention.
        if len_trajectory == 1:
            t_advantage_seq = torch.zeros_like(t_advantage_seq)
        else:
            t_advantage_seq = (t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True)) / \
                              (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)

        return t_advantage_seq.flatten(0, 1), v_target_seq

    def _segmented_gae(self):
        """Anchored-segment GAE.

        The rollout is a concatenation (along time) of per-segment blocks given
        by ``segment_lengths``. Each segment is a complete static episode, so
        GAE runs INDEPENDENTLY per block: an env's last active step in a block
        is a true terminal (delta = r - V, no bootstrap), and no advantage ever
        crosses an event boundary -> the critic only estimates the return
        conditioned on already-realized events. ``active_mask_seq`` marks which
        envs were still scheduling at each step (envs seeded from larger fixed
        prefixes finish earlier); inactive steps contribute nothing.

        Flattening order matches ``transpose_data`` (env-major: flat index
        b*N + n), so ``active_flat`` aligns with the flattened tensors.
        """
        device = self.t_old_val_seq.device
        reward_arr = torch.stack(self.reward_seq, dim=0).to(device)         # [N, B]
        values = self.t_old_val_seq.transpose(0, 1)                         # [N, B]
        active = torch.stack(self.active_mask_seq, dim=0).to(
            device=device, dtype=torch.bool)                               # [N, B]
        N, B = reward_arr.shape

        advantages = torch.zeros(N, B, device=device)
        seg_start = 0
        for seg_len in self.segment_lengths:
            seg_end = seg_start + seg_len            # block [seg_start, seg_end)
            adv = torch.zeros(B, device=device)
            for i in reversed(range(seg_start, seg_end)):
                act_i = active[i]                    # [B] bool
                if i + 1 < seg_end:
                    next_act = active[i + 1]
                    v_next = torch.where(
                        next_act, values[i + 1], torch.zeros_like(values[i + 1]))
                else:
                    next_act = torch.zeros(B, dtype=torch.bool, device=device)
                    v_next = torch.zeros(B, device=device)
                delta = reward_arr[i] + self.gamma * v_next - values[i]
                adv = delta + self.gamma * self.gae_lambda * next_act.float() * adv
                adv = torch.where(act_i, adv, torch.zeros_like(adv))
                advantages[i] = adv
            seg_start = seg_end

        t_advantage_seq = advantages.transpose(0, 1).to(torch.float32)      # [B, N]
        v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)  # [B*N]

        # Per-env advantage normalization over ACTIVE entries only (inactive
        # zeros would otherwise bias the mean/std).
        active_bn = active.transpose(0, 1).float()                          # [B, N]
        cnt = active_bn.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (t_advantage_seq * active_bn).sum(dim=1, keepdim=True) / cnt
        var = (((t_advantage_seq - mean) ** 2) * active_bn).sum(
            dim=1, keepdim=True) / cnt
        std = var.sqrt()
        t_advantage_seq = (t_advantage_seq - mean) / (std + 1e-8)
        t_advantage_seq = t_advantage_seq * active_bn                       # zero inactive

        self.active_flat = active_bn.flatten(0, 1).bool()                   # [B*N]
        return t_advantage_seq.flatten(0, 1), v_target_seq


class PPO:
    """PPO algorithm following DANIEL structure.

    Subclasses can override ``policy_cls`` to select the policy network.
    ``entropy_coef`` controls the policy-entropy term in the loss.
    """

    policy_cls = HGT

    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs

        self.ploss_coef = config.ploss_coef
        self.vloss_coef = config.vloss_coef
        self.minibatch_size = config.minibatch_size
        # Gradient accumulation: run forward/backward in chunks of
        # ``micro_batch_size`` while the effective (statistical) batch stays at
        # ``minibatch_size``. Lets a large size train at the same effective
        # batch as a small one instead of shrinking the batch to fit VRAM.
        if type(config.micro_batch_size) is not int or config.micro_batch_size < 1:
            raise ValueError('micro_batch_size must be a positive integer')
        self.micro_batch_size = config.micro_batch_size
        self.max_grad_norm = config.max_grad_norm
        self.entropy_coef = float(config.entropy_coef)

        # Critic warm-up: for the first N updates, train ONLY the critic head
        # and leave the policy bit-identical. Value targets scale with problem
        # size, so a critic trained at one size starts badly miscalibrated at a
        # larger one (measured: value_loss 1.5 at 20x10 vs 41 at 50x10 for the
        # same checkpoint). With vloss_coef=0.5, no return normalisation, and
        # max_grad_norm=1.0 forcing a unit-length step, that critic error
        # dominates the gradient DIRECTION and drags the actor off a cliff.
        # Letting the critic re-calibrate first fixes the cause without
        # touching the actor's training recipe.
        self.critic_warmup = int(getattr(config, 'critic_warmup', 0) or 0)

        self.policy = self.policy_cls(config)
        self.policy_old = deepcopy(self.policy)
        # The behaviour policy MUST be deterministic. HGTFeatureExtractor
        # applies nn.Dropout(config.dropout) to every node embedding at every
        # layer (hgt_network.py residual_dropout), and deepcopy inherits
        # train() mode, so the log_prob(a_t) stored during the rollout was a
        # single random draw. Re-evaluating in update() draws a DIFFERENT mask,
        # so the PPO ratio exp(logp_new - logp_old) was a random variable even
        # at bit-identical weights. Measured on the 100x20 warm start
        # (scratch/probe_ratio_noise.py): 43.8% of ratios fell outside the
        # eps_clip band [0.8, 1.2] with the weights literally unchanged; the
        # eval-mode control gives ratio == 1.0000, std 0.0000. The tell was
        # already in the logs — during critic-only warm-up the actor is frozen,
        # so policy_loss must be -mean(z-scored advantage) ~ -0.0001, and it
        # logged ~0.025, which is exactly what this noise predicts (0.0220).
        # Noise grows with graph size (25% outside the band at 20x10 vs 43.8%
        # at 100x20), which is why small sizes trained fine and 100x20 did not.
        self.policy_old.eval()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)
        self.wandb_enabled = getattr(config, 'wandb_enabled', True)
    
    def _set_actor_trainable(self, flag: bool):
        """Toggle grad on everything the actor's output depends on.

        Critic-side parameter groups are ``critic``, ``critic_node_pools`` and
        ``critic_candidate_gate``; everything else (``feature_extractor``,
        ``actor``, ``actor_node_pools``) feeds the action distribution. The
        shared feature extractor is frozen too, so during warm-up the policy is
        not merely un-optimised, it is provably unchanged.

        Frozen params keep ``grad=None`` after ``zero_grad(set_to_none=True)``,
        so Adam skips them and ``clip_grad_norm_`` ignores them.
        """
        for name, p in self.policy.named_parameters():
            if not name.startswith('critic'):
                p.requires_grad_(flag)

    def update(self, memory, update_step=0, raw_rewards=None, raw_makespan=None, raw_twt=None,
               makespan_weight=1.0, twt_weight=1.0):
        """Update policy using PPO"""
        # Warm-up updates train the critic alone; the actor sees them as a
        # no-op. Budget them ON TOP of the intended actor budget, i.e. pass
        # max_updates = critic_warmup + N so every arm still gets N updates
        # that actually move the policy.
        in_warmup = update_step < self.critic_warmup
        self._set_actor_trainable(not in_warmup)
        t_data = memory.transpose_data()
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        # Restrict the update to valid (active) samples. For the anchored-segment
        # rollout, inactive steps (envs that already finished their segment) are
        # masked out entirely so they never enter a minibatch / the loss. For
        # the legacy path active_flat is None -> all samples are valid.
        active_flat = getattr(memory, 'active_flat', None)
        if active_flat is None:
            valid_idx = torch.arange(len(t_data[-1]))
        else:
            valid_idx = torch.nonzero(active_flat, as_tuple=True)[0].cpu()
        full_batch_size = len(valid_idx)
        num_batch = np.ceil(full_batch_size / self.minibatch_size)

        # Explained variance of the critic over the valid samples of this
        # batch. EV = 1 - Var(target - pred) / Var(target) is DIMENSIONLESS,
        # so unlike value_loss it is directly comparable across problem
        # sizes and needs no return normalization to be meaningful. That is
        # what lets the curriculum's advance gate be independent of the
        # normalizer. NaN when the target variance is degenerate.
        # PopArt: fold this rollout's REAL-UNIT targets into the running
        # statistics and rescale the critic head. Exactly once, HERE:
        # before the k_epochs loop and before the micro-batch
        # accumulation loop. Inside either one, the same rollout would
        # enter the statistics up to k_epochs times and the head would
        # be rescaled between accumulated gradients of one step.
        if getattr(self.policy, 'use_popart', False):
            self.policy.popart_update_stats(v_target_seq[valid_idx])

        explained_var = float('nan')
        try:
            _vt = v_target_seq.reshape(-1)[valid_idx]
            _vp = memory.t_old_val_seq.reshape(-1)[valid_idx]
            if _vt.numel() > 1:
                _den = torch.var(_vt.float())
                if float(_den) > 1e-12:
                    explained_var = float(
                        1.0 - torch.var(_vt.float() - _vp.float()) / _den)
        except Exception:
            pass
        
        loss_epochs = 0
        v_loss_epochs = 0
        p_loss_epochs = 0
        
        # Accumulators for logging metrics across all minibatches
        total_v_loss = 0
        total_p_loss = 0
        total_total_loss = 0
        total_grad_norm = 0
        num_minibatch_updates = 0
        
        # Evaluate the SAME deterministic function the rollout used, so the
        # ratio reflects the policy change and nothing else. Dropout is the
        # only train/eval-dependent module in the HGT (no BatchNorm; the
        # actor/critic MLP heads are already built with dropout=0.0), and
        # gradients still flow normally in eval mode.
        _prev_training_mode = self.policy.training
        self.policy.eval()

        for epoch in range(self.k_epochs):
            perm = valid_idx[torch.randperm(full_batch_size)]
            for i in range(int(num_batch)):
                start_idx = i * self.minibatch_size
                end_idx = min((i + 1) * self.minibatch_size, full_batch_size)
                group = perm[start_idx:end_idx]
                group_size = len(group)
                if group_size == 0:
                    continue

                # One effective minibatch, optionally split into micro-batches.
                # Each micro-batch loss is weighted by its share of the
                # effective batch (micro_size / group_size), so the accumulated
                # gradient equals the gradient of the full minibatch mean loss
                # exactly, including a short trailing micro-batch. Clipping and
                # the optimizer step run ONCE, after accumulation; clipping per
                # micro-batch would NOT be equivalent.
                self.optimizer.zero_grad()
                g_v_loss = 0.0
                g_p_loss = 0.0
                g_total_loss = 0.0

                for j in range(0, group_size, self.micro_batch_size):
                    indices = group[j:j + self.micro_batch_size]
                    weight = len(indices) / group_size

                    candidate_mask = t_data[0][indices]
                    candidate = t_data[1][indices]
                    candidate_pt_indices = t_data[2][indices]
                    graph_seq = [t_data[3][idx] for idx in indices.tolist()]

                    pis, vals = self.policy(
                        candidate=candidate,
                        candidate_mask=candidate_mask,
                        candidate_pt_indices=candidate_pt_indices,
                        graph=graph_seq
                    )

                    action_batch = t_data[4][indices]
                    logprobs, entropy = eval_actions(pis, action_batch)
                    ratios = torch.exp(logprobs - t_data[7][indices].detach())

                    advantages = t_advantage_seq[indices]
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                    if getattr(self.policy, 'use_popart', False):
                        # Critic trains in NORMALIZED units: both sides go
                        # through the same affine, so the loss stays well
                        # scaled however large the real objective is. vals
                        # is real-unit (forward denormalizes), so
                        # normalizing it recovers the head's own output
                        # exactly - the round trip is an exact identity.
                        v_loss = self.V_loss_2(
                            self.policy.popart_normalize(vals.squeeze(-1)),
                            self.policy.popart_normalize(
                                v_target_seq[indices]))
                    else:
                        v_loss = self.V_loss_2(
                            vals.squeeze(-1), v_target_seq[indices])
                    p_loss = -torch.min(surr1, surr2)
                    if in_warmup:
                        # Critic only. v_loss is already the micro-batch mean,
                        # so weighting by micro/group below still reconstructs
                        # the exact full-minibatch mean.
                        loss = self.vloss_coef * v_loss
                    else:
                        loss = (self.vloss_coef * v_loss
                                + self.ploss_coef * p_loss
                                - self.entropy_coef * entropy)

                    loss_mean = loss.mean()
                    v_loss_mean = v_loss.mean()
                    p_loss_mean = p_loss.mean()

                    (loss_mean * weight).backward()

                    g_v_loss += v_loss_mean.item() * weight
                    g_p_loss += p_loss_mean.item() * weight
                    g_total_loss += loss_mean.item() * weight

                loss_epochs += g_total_loss
                v_loss_epochs += g_v_loss
                p_loss_epochs += g_p_loss

                # Clip gradients and get pre-clip norm in one pass. With
                # accumulation this is per EFFECTIVE minibatch, so the logged
                # gradient_norm is not directly comparable to runs made before
                # accumulation existed (there it was per forward pass).
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm).item()

                if not math.isfinite(grad_norm):
                    raise FloatingPointError('Non-finite PPO gradient; update not committed')
                self.optimizer.step()

                # Accumulate metrics for averaged logging
                total_v_loss += g_v_loss
                total_p_loss += g_p_loss
                total_total_loss += g_total_loss
                total_grad_norm += grad_norm
                num_minibatch_updates += 1
            
        
        self.policy.train(_prev_training_mode)

        # Build metrics dict for caller to log (single wandb.log per update)
        update_metrics = {}
        if num_minibatch_updates > 0:
            update_metrics = {
                'samples': int(full_batch_size),
                'optimizer_steps': int(num_minibatch_updates),
                'value_loss': total_v_loss / num_minibatch_updates,
                'policy_loss': total_p_loss / num_minibatch_updates,
                'total_loss': total_total_loss / num_minibatch_updates,
                'gradient_norm': total_grad_norm / num_minibatch_updates,
                'raw_reward': np.mean(raw_rewards) if raw_rewards is not None else 0,
                'raw_makespan': np.mean(raw_makespan) if raw_makespan is not None else 0,
                'raw_twt': np.mean(raw_twt) if raw_twt is not None else 0,
                'raw_objective': (makespan_weight * np.mean(raw_makespan) if raw_makespan is not None else 0) +
                                 (twt_weight * np.mean(raw_twt) if raw_twt is not None else 0),
                'explained_variance': explained_var,
                'popart/mu': (float(self.policy.popart_mu.item())
                              if getattr(self.policy, 'use_popart', False)
                              else 0.0),
                'popart/sigma': (float(self.policy.popart_sigma.item())
                                 if getattr(self.policy, 'use_popart', False)
                                 else 1.0),
            }

        # Leave every parameter trainable again so nothing downstream (the next
        # update, a resume, a checkpoint reload) inherits warm-up's frozen flags.
        self._set_actor_trainable(True)
        update_metrics['critic_warmup'] = float(in_warmup)

        # Hard copy of policy to policy_old after each update
        for p_old, p in zip(self.policy_old.parameters(), self.policy.parameters()):
            p_old.data.copy_(p.data)
        for b_old, b in zip(self.policy_old.buffers(), self.policy.buffers()):
            b_old.data.copy_(b.data)
        
        # These are plain floats now (accumulation sums per-effective-batch
        # scalars rather than per-forward tensors).
        return (loss_epochs / self.k_epochs,
                v_loss_epochs / self.k_epochs,
                p_loss_epochs / self.k_epochs,
                update_metrics)


class HGTTrainer:
    """HGT Trainer following DANIEL structure"""
    
    def __init__(self, config):
        require_training_controls(config)
        self.n_j = config.n_j
        self.n_m = config.n_m
        self.config = config
        if getattr(config, 'resume_path', None) and not Path(config.resume_path).is_file():
            raise FileNotFoundError(f'Requested resume snapshot is missing: {config.resume_path}')
        self.max_updates = config.max_updates
        self.reset_env_episode = config.reset_env_episode
        if self.reset_env_episode != 1:
            raise ValueError('Resumable training requires reset_env_episode=1')
        self.num_envs = config.num_envs
        self.seed = config.seed
        self.reproducibility = set_global_seed(self.seed)
        config.reproducibility = self.reproducibility
        
        # Data generator. With CL on the pacer overrides ``num_*`` event
        # counts per update; with CL off they stay at the configured target.
        self.training_events = config.training_events
        self.data_generator = self._create_training_data_generator()

        # Curriculum learning
        cl_params = getattr(config, 'curriculum_learning', {})
        self.use_curriculum = cl_params.get('enabled', False)
        if self.use_curriculum:
            self.data_generator.init_curriculum(cl_params, config.max_updates)

        self.use_lowerbound_differencing = resolve_dp_lb_diff_enabled(config)
        # Which admissible bound the reward differences: 'dp' (consults machine
        # free times) or 'chain' (the bound prior lower-bound rewards use).
        self.reward_measure = resolve_dp_lb_measure(config)

        self.use_popart = config.use_popart

        # Anchored-segment rollout (dynamic design): collect each rollout as a
        # chain of linked static episodes split at dynamic-event boundaries, so
        # value targets never bootstrap across an event. Defaults to the dynamic
        # design flag (on when use_dynamic_event_features is True), off for the
        # old/static design. See _collect_segmented_rollout.
        self.use_anchored_segments = bool(getattr(
            config, 'use_anchored_segments',
            getattr(config, 'use_dynamic_event_features', True)))

        # Use pre-defined result directory from config if provided, otherwise use a default
        self.result_dir = getattr(config, 'result_dir', f'./result')
        
        self.data_name = f'{self.n_j}x{self.n_m}'
        # Get model_name directly from config (should be in common_params)
        self.model_name = getattr(config, "model_name", self.data_name)
        
        # Objective weights (saved in checkpoint for critic pruning)
        self.objective_weights = getattr(config, 'objective_weights',
                                         {'makespan_weight': 0.5, 'twt_weight': 0.5})

        # Initialize environment
        self.env = self._create_environment()
        self.env_class = type(self.env)

        # Set feature dimensions from environment (important for dynamic mode with extended features)
        config.op_feature_dim = self.env.op_fea_dim
        config.job_feature_dim = self.env.job_fea_dim
        config.machine_feature_dim = self.env.mch_fea_dim
        config.pt_feature_dim = self.env.pt_fea_dim

        # Initialize PPO (after feature dims are set)
        self.ppo = PPO(config)
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

        # Validation parameters. A single fixed dynamic-validation fixture is
        # sampled from target parameters and reused throughout the run.
        self.update_per_validate = config.update_per_validate
        self.stopping_rule = config.stopping_rule
        self.early_stop_patience = config.early_stop_patience
        self._stop_ctl = ValidationStopController(self.stopping_rule, self.early_stop_patience)
        self._resume_contract = training_contract(config)
        self._source_identity = source_identity()
        self._resume_rng = None
        self._training_seconds = 0.0
        self._val_no_improve = 0
        self._early_stop_outcome = None
        self._early_stop_update = None
        # Derived, not used raw. seed_context() restores the numpy stream to
        # the position it entered at, so a validation seed EQUAL to the
        # training seed (the shipped default was 42 for both) makes the fixture
        # start from exactly the position the first training rollout starts
        # from — validation instance i was then training-batch instance i, i.e.
        # the model was selected on instances it had just trained on. Folding
        # through derive_seed puts the fixture on its own stream while keeping
        # it deterministic from the configured value.
        self.validation_seed = int(derive_seed(config.validation_seed, 'validation_fixture'))
        self.validation_set_size = config.validation_set_size
        self.validation_data = self._generate_validation_set()
        fixture_bytes = (json.dumps(checkpoint_safe_metadata(self.validation_data),
                                    sort_keys=True, allow_nan=False) + '\n').encode()
        self.validation_fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
        root = Path(self.result_dir)
        root.mkdir(parents=True, exist_ok=True)
        if not getattr(config, 'resume_path', None) and (
                (root / 'resume_state.pth').exists() or (root / 'train_metrics.jsonl').exists()):
            raise ValueError('Existing run requires --resume; refusing a fresh overwrite')
        fixture_path = root / 'validation-fixture.json'
        if fixture_path.exists() and fixture_path.read_bytes() != fixture_bytes:
            raise ValueError('Existing validation fixture differs from the resolved configuration')
        self.best_val_objective = float('inf')
        self._best_model_payload = None
        self._selected_update_index = None
        self._next_update = 0
        self._resume_completed = False

        # Resumable checkpointing. ``checkpoint_every`` updates -> a full
        # training-state snapshot (resume_state.pth) is overwritten so an
        # OOM-/SIGTERM-killed run can continue instead of restarting. The
        # curriculum pacer is a pure function of the update index, so resume
        # only needs (weights, optimizer, RNG, update index, best-val, wandb id).
        self.checkpoint_every = config.checkpoint_every
        self.resume_path = getattr(config, 'resume_path', None)
        self.start_update = 0
        self._resume_wandb_id = None
        self._wandb_run_id = None

        # Load full state BEFORE wandb.init so we can rejoin the same W&B run.
        if self.resume_path and os.path.exists(self.resume_path):
            self._load_resume_checkpoint(self.resume_path)

        # Cross-size warm start. Unlike --resume (same run, restores optimizer
        # state / RNG / update index), --init_from takes ONLY the policy
        # weights from another run's checkpoint and starts a fresh run: new
        # result dir, fresh Adam state, update counter back at 0. The network
        # is size-agnostic (all dims are feature dims, not problem dims), so a
        # checkpoint trained at one size loads unchanged at another.
        self.init_from = getattr(config, 'init_from', None)
        if self.init_from and not self.resume_path:
            self._load_policy_weights(self.init_from)

        # ---- Scale curriculum: grow the training size when the critic is
        # ready. Motivated by the 50x10 fine-tune divergence (value targets
        # ~4x larger -> inherited critic never recovers; a critic-only warm-up
        # only delayed the same collapse). Stages remove the return-scale jump
        # by climbing a ladder of sizes, gated on value-loss convergence.
        # Within one update the batch stays single-size (the vectorized env
        # shares n_j/n_m across envs); scales mix across updates, not inside.
        sc_cfg = getattr(config, 'scale_curriculum', None) or {}
        self._scale_ctl = None
        self._scale_stages = None
        # How the curriculum ENDED. None while running; one of done /
        # diverged / stuck / budget_exhausted afterwards. Recorded because a
        # run that merely ran out of updates used to be indistinguishable from
        # one that completed its hold, in the checkpoint and in the exit code.
        self._scale_outcome = getattr(self, '_resumed_scale_outcome', None)
        if sc_cfg.get('enabled'):
            if self.use_curriculum:
                raise ValueError(
                    'scale_curriculum cannot run with the DDT curriculum: the '
                    'DDT pacer is a pure function of the global update index '
                    'and does not survive stage rebuilds. Pass '
                    '--disable_curriculum.')
            self._scale_stages = [tuple(s) for s in sc_cfg['stages']]
            fj, fm = self._scale_stages[-1][0], self._scale_stages[-1][1]
            if (fj, fm) != (self.n_j, self.n_m):
                raise ValueError(
                    f'config size {self.n_j}x{self.n_m} must equal the final '
                    f'curriculum stage {fj}x{fm} (the run is named, validated '
                    'and checkpointed at the final size)')
            # Validation stays pinned to the FINAL size for the whole run so
            # best-model selection is comparable across stages. The fixture
            # above was generated at this size, before any stage switch;
            # Validation reads validation_n_j/n_m for its environment.
            self.validation_n_j, self.validation_n_m = self.n_j, self.n_m
            self._scale_ctl = ScaleCurriculumController(
                n_stages=len(self._scale_stages),
                min_stage_updates=sc_cfg['min_stage_updates'],
                max_stage_updates=sc_cfg['max_stage_updates'],
                window=sc_cfg['window'],
                plateau_tol=sc_cfg['plateau_tol'],
                cv_tol=sc_cfg['cv_tol'],
                div_factor=sc_cfg['div_factor'],
                div_patience=sc_cfg['div_patience'],
                final_hold_updates=sc_cfg['final_hold_updates'],
                ev_threshold=sc_cfg['ev_threshold'],
                descend_patience=sc_cfg['descend_patience'],
                settle_updates=sc_cfg['settle_updates'],
                # Sizes only: changing a stage's micro_batch is a memory
                # decision and must not block a resume, while changing a SIZE
                # means the snapshot belongs to a different experiment.
                ladder_id=','.join(f'{s[0]}x{s[1]}'
                                   for s in self._scale_stages),
            )
            resumed_state = getattr(self, '_scale_resume_state', None)
            if resumed_state:
                self._scale_ctl.load_state_dict(resumed_state)
            self._apply_scale_stage(self._scale_ctl.stage_idx)

        if self.stopping_rule == 'readiness_patience' and self._scale_ctl is None:
            raise ValueError('readiness_patience requires a resolved scale controller')
        atomic_bytes(fixture_path, fixture_bytes)
        self._journal = TrainingJournal(self.result_dir, self.seed,
                                        self.validation_fixture_sha256, self.start_update)
        # Initialize wandb
        self.wandb_project = getattr(config, 'wandb_project', 'exp_hgt')
        self.wandb_enabled = getattr(config, 'wandb_enabled', True)

        if self.wandb_enabled:
            run_name = getattr(config, 'wandb_run_name', None)
            if not run_name or not isinstance(run_name, str):
                run_name = f"HGT_{self.data_name}_{self.model_name}"
            init_kwargs = dict(
                project=self.wandb_project,
                name=run_name,
                config=vars(config),
            )
            # Rejoin the prior run on resume so the learning curve is continuous.
            if self._resume_wandb_id:
                init_kwargs['id'] = self._resume_wandb_id
                init_kwargs['resume'] = 'allow'
            wandb.init(**init_kwargs)
            self._wandb_run_id = wandb.run.id if wandb.run is not None else None
        if self._resume_rng is not None:
            self._restore_rng(self._resume_rng)
    
    
    def _create_training_data_generator(self):
        """Disable only training events; target validation stays unchanged."""
        if self.training_events:
            return create_data_generator(self.config)
        config = deepcopy(self.config)
        config.problem_params = dict(self.config.problem_params)
        for key in ('num_new_job_arrival_waves', 'num_machine_breakdown_waves',
                    'num_priority_change_waves', 'num_new_job_arrivals',
                    'num_machine_breakdowns', 'num_priority_changes'):
            put_range_aliases(config.problem_params, key, 0, 0)
        return create_data_generator(config)

    def _create_environment(self):
        env = FJSPEnvHGT(self.n_j, self.n_m, device=self.config.device,
                         use_lowerbound_differencing=self.use_lowerbound_differencing,
                         use_pt_nodes=getattr(self.config, 'use_pt_nodes', True),
                         use_dynamic_event_features=getattr(
                             self.config, 'use_dynamic_event_features', True),
                         reward_measure=self.reward_measure)
        env.makespan_weight = self.objective_weights['makespan_weight']
        env.twt_weight = self.objective_weights['twt_weight']
        return env

    def _high_water_idx(self):
        """Highest stage reached, clamped into the current ladder.

        Clamped because this index is used inside save_model(), which runs at
        every best-validation and again right after a DIVERGED or STUCK stop.
        An IndexError there would destroy a multi-day run at the exact moment
        it is trying to record its result, which is the one place in this file
        that must never raise.
        """
        n = len(self._scale_stages or []) - 1
        return max(0, min(int(getattr(self._scale_ctl, 'max_stage_idx', 0)), n))

    def _apply_scale_stage(self, idx):
        """Switch training to curriculum stage ``idx``: rebuild the data
        generator and the vectorized env at the stage size.

        The policy, optimizer and PPO state carry over untouched — the network
        is size-agnostic (all dims are feature dims), and keeping the same
        Adam state across stages is the point: no fresh-optimizer shock at
        each boundary, unlike chaining separate --init_from runs. Feature
        widths do not change with size, so the PPO built at the final size
        stays valid at every stage.
        """
        from src.utils.config_util import retarget_config_size

        n_j, n_m, micro = self._scale_stages[idx]
        self.n_j, self.n_m = n_j, n_m
        # Rebuilding problem_params by hand here is what froze the dynamic-
        # event MAGNITUDES at the final stage's size: retarget_config_size
        # routes through _sync_training_event_params, the only place
        # --scale_event_magnitudes_to_size is ever applied. Measured on
        # 10x5:100x20:2 before the fix, every breakdown wave removed 6-10
        # machines from the 5-machine first stage (a total shutdown, every
        # wave) and every arrival added 15-20 jobs to a 10-job shop: precisely
        # the confound that flag exists to remove, and worst at the bottom of
        # the ladder where a descending run spends the most time.
        retarget_config_size(self.config, n_j, n_m)
        # Generator captures size at construction; env allocates from n_j/n_m.
        self.data_generator = self._create_training_data_generator()
        self.env = self._create_environment()
        self.ppo.micro_batch_size = micro
        # Invalidate the anchored-rollout data cache and force a resample so
        # the next rollout cannot touch stale previous-stage data.
        if hasattr(self, '_segment_data'):
            del self._segment_data
        self._force_resample = True
        # The event magnitudes are printed, not just the size. They were the
        # blocker nobody could see: they stayed frozen at the final stage's
        # amounts for the whole ladder, and nothing in any log said so. A
        # breakdown wave that is most of the shop, or an arrival wave larger
        # than the shop, is now visible on the line that announces the stage.
        gp = self.data_generator.get_params()
        def _rng(key):
            return (gp.get(f'{key}_min'), gp.get(f'{key}_max'))
        bd, ar = _rng('machines_per_breakdown'), _rng('new_jobs_per_arrival')
        tqdm.write(
            f'[scale-curriculum] stage {idx + 1}/{len(self._scale_stages)}: '
            f'training size -> {n_j}x{n_m}'
            + f', micro_batch={self.ppo.micro_batch_size}'
            + f'; events: breakdown {bd[0]}-{bd[1]} of {n_m} machines, '
            f'arrival {ar[0]}-{ar[1]} onto {n_j} jobs')
    
    def sample_training_instances(self):
        """
            Sample training instances using data generator.
            
        :return: Dict with 'static' data (and 'dynamic_events', 'dynamic_jobs' for dynamic mode)
        """
        # Both static and dynamic modes use sample_env which returns a dict
        return self.data_generator.sample_env(self.num_envs)

    # ------------------------------------------------------------------
    # Anchored-segment rollout (dynamic design)
    # ------------------------------------------------------------------

    def _collect_segmented_rollout(self, data):
        """Collect one rollout as a chain of linked static episodes.

        ``E_0`` = the original problem solved with NO events, to termination
        (the anchor whose return supplies low-variance value targets for the
        pre-event states). At each event boundary we freeze the ops that started
        before the event into a per-env fixed prefix, absorb the event, and seed
        the next segment from that partial schedule via
        ``set_initial_data_with_fixed_schedule`` -- the exact path inference
        uses (``src/utils/segment_util.py`` is the shared freeze/absorb rule).

        Every segment is a complete static episode, so GAE never bootstraps
        across an event: the critic only estimates the return conditioned on
        already-realized events. Transitions for all segments are appended to
        ``self.memory`` with per-step active masks and per-segment lengths.

        Returns ``(final_env, ep_rewards)`` where ``final_env`` holds the full
        realized schedule (fixed prefix + last segment) for makespan/TWT logging
        and ``ep_rewards`` is the per-env total reward (for monitoring).
        """
        # Lazy import to avoid an import cycle (inference_env imports hgt_env).
        from src.HGT.inference_env import (
            _copy_static, _wrap_as_static_data,
            _append_arriving_job, _apply_priority_change)

        B = self.num_envs
        n_m = self.n_m
        # Event schedule is shared across the B envs (same boundary times); each
        # env's frozen prefix differs because its schedule differs.
        event_queue = merge_and_sort_events(data.get('dynamic_events', {}))
        current_static = _copy_static(data['static'])
        dyn_jobs = data.get('dynamic_jobs') or {}
        ddt_min = float(dyn_jobs.get('ddt_min', 0.5))
        ddt_max = float(dyn_jobs.get('ddt_max', 1.5))
        n_j0 = len(current_static['job_length'][0])

        fixed_ops = [[] for _ in range(B)]                 # per-env carry-forward
        mch_unavailable_until = np.zeros((B, n_m), dtype=np.float32)
        job_release_times = [[0.0] * n_j0 for _ in range(B)]
        current_time = 0.0

        self.memory.clear_memory()
        last_env = self.env
        seg_idx = 0
        n_arr = n_bd = n_pc = 0          # applied-event counts (for logging)
        while True:
            wrapped = _wrap_as_static_data(current_static)
            if seg_idx == 0:
                state = self.env.set_initial_data_static(wrapped)
            else:
                state = self.env.set_initial_data_with_fixed_schedule(
                    wrapped,
                    fixed_ops=fixed_ops,
                    mch_unavailable_until=mch_unavailable_until.copy(),
                    job_release_times=np.asarray(job_release_times, dtype=np.float32),
                    current_time=current_time,
                )

            # Roll this segment (no events inside) to termination.
            self._roll_segment_into_memory(state)
            last_env = self.env

            if not event_queue:
                break

            # Extract each env's full segment schedule, then freeze + absorb the
            # next event-set into the seed for the following segment.
            seg_scheds = [extract_schedule_from_env(self.env, e)['schedule']
                          for e in range(B)]
            event_set, t = pop_next_event_set(event_queue)
            for e in range(B):
                fixed_ops[e], _interrupted = freeze_segment_ops(
                    seg_scheds[e], fixed_ops[e], event_set, t, env_idx=e)
                absorb_breakdowns(fixed_ops[e], mch_unavailable_until[e], event_set)

            # Arrivals/priority change the shared static problem (same for all
            # envs); release times grow per arrival.
            for evt in event_set:
                if evt['type'] == 'job_arrival':
                    _append_arriving_job(current_static, data, evt, ddt_min, ddt_max)
                    for e in range(B):
                        job_release_times[e].append(float(evt['time']))
                    n_arr += 1
                elif evt['type'] == 'machine_breakdown':
                    n_bd += 1
                elif evt['type'] == 'priority_change':
                    _apply_priority_change(current_static, evt)
                    n_pc += 1
            current_time = t
            seg_idx += 1

        # Expose applied-event counts in the same shape the legacy inline path
        # used (per-env arrays) so the wandb logging block is unchanged.
        self.env.fired_event_counts = {
            'job_arrival': np.full(B, n_arr, dtype=np.int64),
            'machine_breakdown': np.full(B, n_bd, dtype=np.int64),
            'priority_change': np.full(B, n_pc, dtype=np.int64),
        }

        if self.memory.reward_seq:
            ep_rewards_t = torch.stack(self.memory.reward_seq, dim=0).sum(dim=0)
            ep_rewards = ep_rewards_t.detach().cpu().numpy()
        else:
            ep_rewards = np.zeros(B, dtype=np.float32)
        return last_env, ep_rewards

    def _roll_segment_into_memory(self, state):
        """Roll one static segment to termination, appending transitions.

        Only active (not-yet-done) envs are forwarded/sampled each step; the
        env's step() skips done envs so their terminal state is preserved. A
        per-step active mask and the segment's step count are recorded for the
        per-segment GAE.
        """
        B = self.num_envs
        dev = self.ppo.device
        seg_len = 0
        while True:
            active = ~self.env.done()                       # [B] bool
            if not active.any():
                break
            self.memory.push(state)                         # full-B state
            a_idx = np.where(active)[0]
            a_idx_t = torch.as_tensor(a_idx, device=dev, dtype=torch.long)

            graph_active = [state.graph_tensor[e] for e in a_idx]
            with torch.no_grad():
                pi_envs, vals_envs = self.ppo.policy_old(
                    candidate=state.candidate_tensor[a_idx_t],
                    candidate_mask=state.candidate_mask_tensor[a_idx_t],
                    candidate_pt_indices=state.candidate_pt_indices[a_idx_t],
                    graph=graph_active,
                )
            act_sub, logp_sub = sample_action(pi_envs)

            # Scatter active-env outputs into full-B tensors (inactive -> 0).
            action_full = np.zeros(B, dtype=np.int64)
            action_full[a_idx] = act_sub.detach().cpu().numpy()
            val_full = torch.zeros(B, device=dev)
            val_full[a_idx_t] = vals_envs.squeeze(-1).to(dev)
            logp_full = torch.zeros(B, device=dev)
            logp_full[a_idx_t] = logp_sub.to(dev)

            state, reward, _done = self.env.step(action_full)

            active_t = torch.as_tensor(active, device=dev)
            reward_t = torch.from_numpy(
                np.asarray(reward, dtype=np.float32)).to(dev) * active_t.float()

            self.memory.action_seq.append(
                torch.as_tensor(action_full, device=dev, dtype=torch.long))
            self.memory.log_probs.append(logp_full)
            self.memory.val_seq.append(val_full)
            self.memory.reward_seq.append(reward_t)
            self.memory.active_mask_seq.append(active_t)
            seg_len += 1

        self.memory.segment_lengths.append(seg_len)

    def _generate_validation_set(self):
        """Sample one fixed target-distribution dynamic validation batch."""
        validation_config = deepcopy(self.config)
        problem_params = dict(getattr(self.config, 'problem_params', {}) or {})
        if not problem_params:
            problem_params = dict(getattr(self.config, 'static_params', {}) or {})
        dynamic_event_params = dict(
            getattr(self.config, 'dynamic_event_params', {}) or {})
        validation_config.problem_params = {
            **problem_params,
            **dynamic_event_params,
        }

        with seed_context(self.validation_seed):
            validation_generator = create_data_generator(validation_config)
            return validation_generator.sample_env(self.validation_set_size)

    def _greedy_segmented_rollout(self, val_data):
        """Validate the full fixture with the training/inference event semantics.

        At each event, freeze operations that started before it, absorb the
        event, and re-seed the next segment using `src/utils/segment_util.py`.
        Actions are greedy and transitions are not written to `self.memory`.
        """
        from src.HGT.inference_env import (
            _copy_static, _wrap_as_static_data,
            _append_arriving_job, _apply_priority_change)

        v_n_j = getattr(self, 'validation_n_j', self.n_j)
        v_n_m = getattr(self, 'validation_n_m', self.n_m)
        val_env = FJSPEnvHGT(
            v_n_j, v_n_m, device=self.config.device,
            use_lowerbound_differencing=self.use_lowerbound_differencing,
            use_pt_nodes=getattr(self.config, 'use_pt_nodes', True),
            use_dynamic_event_features=getattr(
                self.config, 'use_dynamic_event_features', True),
            reward_measure=self.reward_measure)
        val_env.makespan_weight = self.objective_weights['makespan_weight']
        val_env.twt_weight = self.objective_weights['twt_weight']

        B = self.validation_set_size
        event_queue = merge_and_sort_events(val_data.get('dynamic_events', {}))
        current_static = _copy_static(val_data['static'])
        dyn_jobs = val_data.get('dynamic_jobs') or {}
        ddt_min = float(dyn_jobs.get('ddt_min', 0.5))
        ddt_max = float(dyn_jobs.get('ddt_max', 1.5))
        n_j0 = len(current_static['job_length'][0])

        fixed_ops = [[] for _ in range(B)]
        mch_unavailable_until = np.zeros((B, v_n_m), dtype=np.float32)
        job_release_times = [[0.0] * n_j0 for _ in range(B)]
        current_time = 0.0
        seg_idx = 0

        self.ppo.policy.eval()
        try:
            while True:
                wrapped = _wrap_as_static_data(current_static)
                if seg_idx == 0:
                    state = val_env.set_initial_data_static(wrapped)
                else:
                    state = val_env.set_initial_data_with_fixed_schedule(
                        wrapped,
                        fixed_ops=fixed_ops,
                        mch_unavailable_until=mch_unavailable_until.copy(),
                        job_release_times=np.asarray(
                            job_release_times, dtype=np.float32),
                        current_time=current_time,
                    )

                while True:
                    if val_env.done().all():
                        break
                    with torch.no_grad():
                        pi, _ = self.ppo.policy(
                            candidate=state.candidate_tensor,
                            candidate_mask=state.candidate_mask_tensor,
                            candidate_pt_indices=state.candidate_pt_indices,
                            graph=state.graph_tensor,
                        )
                    action = greedy_select_action(pi)
                    state, _reward, _done = val_env.step(action.cpu().numpy())

                if not event_queue:
                    break

                seg_scheds = [extract_schedule_from_env(val_env, e)['schedule']
                              for e in range(B)]
                event_set, t = pop_next_event_set(event_queue)
                for e in range(B):
                    fixed_ops[e], _ = freeze_segment_ops(
                        seg_scheds[e], fixed_ops[e], event_set, t, env_idx=e)
                    absorb_breakdowns(
                        fixed_ops[e], mch_unavailable_until[e], event_set)

                for evt in event_set:
                    if evt['type'] == 'job_arrival':
                        _append_arriving_job(
                            current_static, val_data, evt, ddt_min, ddt_max)
                        for e in range(B):
                            job_release_times[e].append(float(evt['time']))
                    elif evt['type'] == 'priority_change':
                        _apply_priority_change(current_static, evt)
                current_time = t
                seg_idx += 1
        finally:
            self.ppo.policy.train()

        makespan = float(np.mean(val_env.current_makespan))
        twt = float(np.mean(getattr(
            val_env, 'current_twt', np.zeros(B))))
        mw = val_env.makespan_weight
        tw = val_env.twt_weight
        return makespan, twt, mw * makespan + tw * twt

    def _run_validation(self):
        """Run fixed dynamic validation and return ``(makespan, twt, objective)``."""
        return self._greedy_segmented_rollout(self.validation_data)

    def train(self, *, stop_after_updates=None):
        """Train the model following DANIEL structure"""
        self.train_st = time.time()
        if self._resume_completed:
            if not (Path(self.result_dir) / f'{self.model_name}.pth').exists():
                self._checkpoint_update_index = self._next_update - 1
                self.save_model()
            self._write_completion()
            if self.wandb_enabled:
                wandb.finish()
            return

        # Persist an initial resume point immediately, before the first update.
        # earlyoom on this shared box SIGTERMs the largest python process the
        # moment host memory gets tight, which can happen within minutes -- well
        # before the first periodic snapshot (every ``checkpoint_every``
        # updates). Without this initial write, such an early kill leaves no
        # resume_state.pth, so the relaunch (a) restarts training from scratch
        # and (b) cannot rejoin the W&B run (its id is stored in the snapshot),
        # spawning a brand-new run. Writing it here lets both survive a kill at
        # any point. Skipped on resume (an up-to-date snapshot already exists).
        if self.start_update == 0:
            self._save_resume_checkpoint(next_update=0)

        # A terminal decision saved before final artifact closure must finish
        # that closure on resume without taking another optimization step.
        loop_end = (self.start_update if self._early_stop_outcome or self._scale_outcome
                    else self.max_updates)
        for i_update in tqdm(range(self.start_update, loop_end),
                             initial=self.start_update, total=self.max_updates,
                             file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()
            stop_after_update = False
            validation_stop = False
            val_obj = None
            trained_size = (self.n_j, self.n_m)
            
            # Curriculum learning: always resample (DDT changes each step). The
            # first iteration after a resume (i_update == start_update) always
            # resamples so neither rollout path touches a stale/never-seeded env
            # (the legacy path's env.reset() needs a prior set_initial_data).
            should_resample = (self.use_curriculum
                               or i_update == self.start_update
                               or i_update % self.reset_env_episode == 0
                               or getattr(self, '_force_resample', False))
            self._force_resample = False
            if should_resample and self.use_curriculum:
                ddt_min, ddt_max = self.data_generator.update_curriculum(i_update)

            if self.use_anchored_segments:
                # Anchored-segment rollout: a chain of linked static episodes
                # (see _collect_segmented_rollout). The driver re-seeds self.env
                # per segment and fills self.memory; after the last segment
                # self.env holds the full realized schedule.
                if should_resample or not hasattr(self, '_segment_data'):
                    self._segment_data = self.sample_training_instances()
                _last_env, ep_rewards = self._collect_segmented_rollout(
                    self._segment_data)
            else:
                if should_resample:
                    data = self.sample_training_instances()
                    state = self.env.set_initial_data_static(data)
                else:
                    state = self.env.reset()

                ep_rewards = -self.env.init_quality.copy()

                while True:
                    # Store state
                    self.memory.push(state)

                    with torch.no_grad():
                        pi_envs, vals_envs = self.ppo.policy_old(
                            candidate=state.candidate_tensor,  # [sz_b, J]
                            candidate_mask=state.candidate_mask_tensor,  # [sz_b, J, M]
                            candidate_pt_indices=state.candidate_pt_indices,  # [sz_b, J, M]
                            graph=state.graph_tensor  # List of HeteroData for batched processing
                        )

                    # Sample action
                    action_envs, action_logprob_envs = sample_action(pi_envs)

                    # State transition
                    state, reward, done = self.env.step(actions=action_envs.cpu().numpy())
                    ep_rewards += reward
                    reward = torch.from_numpy(reward).to(self.ppo.device)

                    # Collect transition
                    self.memory.reward_seq.append(reward)
                    self.memory.action_seq.append(action_envs)
                    self.memory.log_probs.append(action_logprob_envs)
                    self.memory.val_seq.append(vals_envs.squeeze(1))
                    # Event-boundary flag for GAE truncation (zeros when no event
                    # fired this step, or when the env exposes no event hook).
                    ev_fired = getattr(self.env, 'last_events_fired', None)
                    if ev_fired is None:
                        ev_fired = np.zeros(self.env.number_of_envs, dtype=np.float32)
                    self.memory.event_boundary_seq.append(
                        torch.as_tensor(np.asarray(ev_fired, dtype=np.float32),
                                        device=self.ppo.device))

                    if done.all():
                        break

            rollout_seconds = time.time() - ep_st
            ppo_st = time.time()
            # PPO update
            loss, v_loss, p_loss, update_metrics = self.ppo.update(
                self.memory, update_step=i_update,
                raw_rewards=ep_rewards, raw_makespan=self.env.current_makespan,
                raw_twt=getattr(self.env, 'current_twt', None),
                makespan_weight=getattr(self.env, 'makespan_weight', 1.0),
                twt_weight=getattr(self.env, 'twt_weight', 1.0))

            ppo_seconds = time.time() - ppo_st
            if not all(math.isfinite(float(v)) for v in (loss, v_loss, p_loss)):
                raise FloatingPointError('Non-finite PPO loss')
            self.memory.clear_memory()

            # Collect all metrics for this update step to emit a single wandb log
            metrics_to_log = dict(update_metrics)

            # Scale-curriculum gate, judged on this update's value_loss. The
            # decision is only ACTED on at the end of the iteration (after
            # logging and the periodic snapshot) so stage boundaries land
            # cleanly in both wandb and resume_state.
            scale_decision = None
            if self._scale_ctl is not None:
                scale_decision = self._scale_ctl.observe(
                    update_metrics.get('value_loss', float('nan')),
                    ev=update_metrics.get('explained_variance'))
                metrics_to_log.update({
                    'scale/train_n_j': self.n_j,
                    'scale/train_n_m': self.n_m,
                })
                gm = self._scale_ctl.gate_metrics()
                metrics_to_log.update({f'scale/{k}': v for k, v in gm.items()})
                # Periodic gate readout in the LOG, not only in wandb. Without
                # it a stage that is not advancing looks identical to a stage
                # that is progressing fine, right up until STUCK stops the run
                # hours later: the log says nothing about how close the gate
                # is. Reading the distance to the threshold is how an operator
                # decides whether to wait or to restart with a lower one.
                if (i_update + 1) % 10 == 0:
                    evm = gm.get('ev_window_median')
                    tqdm.write(
                        f'[scale-gate] stage {gm["stage_idx"] + 1}/'
                        f'{len(self._scale_stages)} ({self.n_j}x{self.n_m}) '
                        f'update {gm["stage_updates"]} of this stage; '
                        + (f'EV median {evm:.3f} vs threshold '
                           f'{self._scale_ctl.ev_threshold}'
                           if evm is not None
                           and self._scale_ctl.ev_threshold is not None
                           else f'value_loss {update_metrics.get("value_loss", float("nan")):.4g}')
                        + f'; may leave at {gm["min_dwell"]}, '
                        f'STUCK at {self._scale_ctl.max_stage_updates}')

            self._stop_ctl.observe_gate(i_update + 1, self._scale_ctl, scale_decision)

            # Always log DDT — flat line (CL off) and ramp (CL on) overlay cleanly in wandb.
            if self.wandb_enabled:
                gen_params = self.data_generator.get_params()
                fired_counts = getattr(self.env, 'fired_event_counts', {}) or {}
                zeros = np.zeros(self.num_envs, dtype=np.int64)
                metrics_to_log.update({
                    'curriculum/ddt_min': gen_params['due_date_tightness_min'],
                    'curriculum/ddt_max': gen_params['due_date_tightness_max'],
                    'curriculum/avg_new_job_arrivals': float(np.mean(
                        fired_counts.get('job_arrival', zeros))),
                    'curriculum/avg_machine_breakdowns': float(np.mean(
                        fired_counts.get('machine_breakdown', zeros))),
                    'curriculum/avg_priority_changes': float(np.mean(
                        fired_counts.get('priority_change', zeros))),
                })

            # Periodic validation on the fixed target dynamic batch. Always
            # validate on the final update so the fully-trained model can be
            # selected as best (otherwise update max_updates-1 is never scored).
            if (i_update % self.update_per_validate == 0
                    or i_update == self.max_updates - 1):
                val_ms, val_twt, val_obj = self._run_validation()
                if not math.isfinite(float(val_obj)):
                    raise FloatingPointError('Validation objective must be finite for checkpoint selection')
                tqdm.write(
                    f'[Validation @ {i_update}] '
                    f'obj={val_obj:.2f} (ms={val_ms:.2f}, twt={val_twt:.2f})')

                self._checkpoint_update_index = i_update
                improved = val_obj < self.best_val_objective
                if improved:
                    self.best_val_objective = val_obj
                    self._selected_update_index = i_update
                    self._best_model_payload = deepcopy(
                        self.save_model(model_name=f"best_{self.model_name}"))
                    # Keep every improving selection: a matched-budget ablation
                    # must be able to select the best within any earlier prefix.
                    self._save_torch_atomic(self._best_model_payload,
                        f'{self.result_dir}/selected_update_{i_update + 1:06d}.pth')
                    tqdm.write(f'  -> New best model (obj: {val_obj:.2f})')
                validation_stop = self._stop_ctl.observe_validation(i_update + 1, val_obj, improved)
                self._val_no_improve = self._stop_ctl.non_improving
                if self.wandb_enabled:
                    metrics_to_log.update({
                        'val/makespan': val_ms,
                        'val/twt': val_twt,
                        'val/objective': val_obj,
                        'val/best_objective': self.best_val_objective,
                    })

            # Log all combined metrics at once
            if self.wandb_enabled and metrics_to_log:
                metrics_to_log['update_step'] = i_update
                wandb.log(metrics_to_log)

            # A failure/demotion has already cleared readiness above; it can
            # never be hidden by a simultaneously exhausted validation counter.
            if validation_stop and scale_decision not in (DESCEND, DIVERGED, STUCK):
                self._early_stop_outcome = 'early_stopped'
                self._early_stop_update = i_update
                self._scale_outcome = 'early_stopped' if self._scale_ctl is not None else None
                tqdm.write(f'[early-stop] target ready at completed update '
                           f'{self._stop_ctl.ready_update}; '
                           f'{self._val_no_improve} later non-improving validations; '
                           f'stopping after {i_update + 1} updates.')
                stop_after_update = True

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)
            mean_twt_all_env = np.mean(getattr(self.env, 'current_twt', np.zeros(self.num_envs)))
            # Compute combined objective with environment weights
            makespan_w = getattr(self.env, 'makespan_weight', 1.0)
            twt_w = getattr(self.env, 'twt_weight', 1.0)
            mean_objective_all_env = makespan_w * mean_makespan_all_env + twt_w * mean_twt_all_env

            ep_et = time.time()

            # Act on the scale-curriculum decision last, so the boundary is
            # recorded cleanly and a kill right after it resumes into the NEW
            # stage rather than replaying the old one.
            if not stop_after_update and scale_decision == ADVANCE:
                self._scale_ctl.advance()
                self._apply_scale_stage(self._scale_ctl.stage_idx)
            elif not stop_after_update and scale_decision == DESCEND:
                # The critic explains less than nothing at this size. Step
                # back and consolidate rather than aborting: with PopArt the
                # value function survives the size change, so a smaller shop
                # is a cheaper place to recover before trying again.
                self._scale_ctl.descend()
                self._apply_scale_stage(self._scale_ctl.stage_idx)
                tqdm.write(
                    f'[scale-curriculum] DESCEND to stage '
                    f'{self._scale_ctl.stage_idx + 1}/'
                    f'{len(self._scale_stages)} ({self.n_j}x{self.n_m}): the '
                    'critic was not coping above. Minimum dwell here is now '
                    f'{self._scale_ctl._min_dwell()} updates.')
                # Clearing here and not at the next validation matters: the
                # dwell after a demotion can be shorter than update_per_validate,
                # so a descend-and-return can carry a spent counter back to the
                # final stage and stop the run on evidence measured against a
                # policy from before the descend.
                self._val_no_improve = 0
            elif not stop_after_update and scale_decision == DONE:
                tqdm.write(
                    f'[scale-curriculum] DONE: final stage '
                    f'({self.n_j}x{self.n_m}) has now spent '
                    f'{self._scale_ctl.final_hold_updates} updates in total '
                    'across all visits; stopping the run successfully.')
                self._scale_outcome = scale_decision
                if self.wandb_enabled and wandb.run is not None:
                    wandb.run.summary['scale_curriculum_outcome'] = scale_decision
                    wandb.run.summary['scale_curriculum_stage'] = (
                        f'{self.n_j}x{self.n_m}')
                stop_after_update = True
            elif not stop_after_update and scale_decision in (DIVERGED, STUCK):
                hi = self._scale_stages[self._high_water_idx()]
                tqdm.write(
                    f'[scale-curriculum] {scale_decision.upper()} at stage '
                    f'{self._scale_ctl.stage_idx + 1}/{len(self._scale_stages)}'
                    f' ({self.n_j}x{self.n_m}) after '
                    f'{self._scale_ctl.stage_updates} stage updates; '
                    'stopping the run. The ladder reached '
                    f'{hi[0]}x{hi[1]} (stage '
                    f'{self._scale_ctl.max_stage_idx + 1}) at its highest — '
                    'that is the result, not a crash.')
                self._scale_outcome = scale_decision
                if self.wandb_enabled and wandb.run is not None:
                    wandb.run.summary['scale_curriculum_outcome'] = scale_decision
                    wandb.run.summary['scale_curriculum_stage'] = (
                        f'{hi[0]}x{hi[1]}')
                    wandb.run.summary['scale_curriculum_final_stage'] = (
                        f'{self.n_j}x{self.n_m}')
                stop_after_update = True

            # Commit a whole logical update: weights, best selection, terminal
            # decision and next-stage state must describe the same boundary.
            self._next_update = i_update + 1
            self._training_seconds += time.time() - ep_st
            self._journal.append(dict(update_metrics, update_step=i_update,
                total_updates=self._next_update, training_events=self.training_events,
                training_size=list(trained_size), stage_decision=scale_decision,
                stage_index=metrics_to_log.get('scale/stage_idx', 0),
                rollout_seconds=rollout_seconds, ppo_seconds=ppo_seconds,
                cumulative_training_seconds=self._training_seconds,
                learning_rate=self.ppo.optimizer.param_groups[0]['lr'],
                validation_objective=val_obj, best_validation_objective=self.best_val_objective,
                selected_update_index=self._selected_update_index,
                stopping=self._stop_ctl.state_dict(), scale=(
                    self._scale_ctl.gate_metrics() if self._scale_ctl else None)))
            if (self._next_update % self.checkpoint_every == 0
                    or scale_decision in (ADVANCE, DESCEND)
                    or stop_after_update or self._next_update == self.max_updates):
                self._save_resume_checkpoint(next_update=self._next_update)
            if stop_after_update:
                break
            if stop_after_updates is not None and self._next_update >= stop_after_updates:
                self._save_resume_checkpoint(next_update=self._next_update)
                if self.wandb_enabled:
                    wandb.finish()
                return

        # Fell out of the loop without any terminal decision: the global
        # update budget ran out. For a laddered run that is NOT the same thing
        # as finishing, and until this branch existed nothing said so - no
        # outcome was recorded, the process exited 0, and the chain script
        # printed ARM COMPLETE for a run that never reached its target size.
        # A checkpoint from such a run was byte-identical in its provenance to
        # one that completed its hold. It is the normal ending of any ladder
        # launched without --stage_final_hold, so it must be nameable.
        # `ran_any` matters: relaunching a FINISHED run gives start_update >=
        # max_updates, so the loop above is a no-op and this branch would
        # otherwise rewrite a run that stopped as DONE into budget_exhausted.
        # Both auto-restart loops resume from the newest snapshot without
        # checking whether it is complete, so that relabelling is one stray
        # relaunch away, and it turns a successful curriculum into a failed
        # one in the permanent record.
        # True only when the loop ended by exhausting its range rather than by
        # any break above. The early-stop fallback below needs this: _scale_outcome
        # is assigned 'budget_exhausted' a few lines down, BEFORE the early-stop
        # artifact is written, so a fallback that reads _scale_outcome would file
        # every laddered cap exhaustion under the ladder's ordinary name and the
        # alarm would only ever appear on non-laddered runs.
        fell_out_of_loop = (self._early_stop_outcome is None
                            and self._scale_outcome is None)
        ran_any = (self.start_update < self._next_update
                   or self._early_stop_outcome is not None
                   or self._scale_outcome is not None
                   or self._next_update >= self.max_updates)
        if (self._scale_ctl is not None
                and self._scale_outcome is None
                and ran_any):
            hi = self._scale_stages[self._high_water_idx()]
            self._scale_outcome = 'budget_exhausted'
            tqdm.write(
                f'[scale-curriculum] BUDGET EXHAUSTED: {self.max_updates} '
                f'updates spent, still on stage '
                f'{self._scale_ctl.stage_idx + 1}/{len(self._scale_stages)} '
                f'({self.n_j}x{self.n_m}). The ladder reached '
                f'{hi[0]}x{hi[1]} at its highest. This run did NOT complete '
                'its curriculum: read it as "the ladder got this far in this '
                'budget", not as a finished policy at the target size.')
            if self.wandb_enabled and wandb.run is not None:
                wandb.run.summary['scale_curriculum_outcome'] = 'budget_exhausted'
                wandb.run.summary['scale_curriculum_stage'] = f'{hi[0]}x{hi[1]}'
                wandb.run.summary['scale_curriculum_final_stage'] = (
                    f'{self.n_j}x{self.n_m}')

        self.train_et = time.time()
        
        # Log training wall time
        effective_train_time = self.train_et - self.train_st
        if self.wandb_enabled:
            wandb.log({'train/training_wall_clock_s': effective_train_time})

        # The early-stop outcome as a standalone artifact, written whenever the
        # rule was ARMED, not only when it fired. An arm that ran to its cap and
        # an arm that stopped early are different results and the difference is not
        # otherwise recoverable from the checkpoint: both leave a best_*.pth and
        # neither says why the run ended. A row group reporting "each arm trains
        # until its validation objective stops improving" is only honest if the
        # arms that never stopped are named.
        _es_path = f'{self.result_dir}/early_stop_outcome.json'
        # `ran_any` for the same reason the curriculum outcome below carries it:
        # a relaunch of a finished run trains nothing and would otherwise rewrite
        # a real early_stopped into cap_reached_without_early_stopping while
        # best_*.pth is untouched.
        if self.stopping_rule == 'readiness_patience' and (ran_any or not os.path.exists(_es_path)):
            # 'cap_reached_without_early_stopping' is an ALARM, not an ending
            # to report.
            # --max_updates is slack under this rule, so reaching it means the
            # run never saw its patience run out, which is a reason to look at
            # the run rather than to write it up. The scale curriculum's own
            # 'budget_exhausted' is a different thing and keeps its name.
            import json as _json
            with open(_es_path, 'w') as fh:
                _json.dump({
                    # A run that ended on the ladder (DONE, DIVERGED, STUCK)
                    # did not reach its cap, and saying so would raise a false
                    # alarm. Only a run that fell out of the loop did.
                    'outcome': (self._early_stop_outcome
                                or ('cap_reached_without_early_stopping' if fell_out_of_loop
                                    else self._scale_outcome)),
                    'stage_at_stop': (None if self._scale_ctl is None
                                      else self._scale_ctl.stage_idx + 1),
                    'stopped_at_update': self._early_stop_update,
                    'max_updates': self.max_updates,
                    'patience_validations': self.early_stop_patience,
                    'stopping_rule': self.stopping_rule,
                    'ready_update': self._stop_ctl.ready_update,
                    'target_updates': (self._scale_ctl._final_stage_updates
                                       if self._scale_ctl is not None
                                       else self._next_update),
                    'update_per_validate': self.update_per_validate,
                    'best_val_objective': float(self.best_val_objective),
                    'val_no_improve': int(self._val_no_improve),
                }, fh, indent=2)

        # The curriculum outcome as a standalone artifact, not only inside a
        # checkpoint. best_*.pth is written mid-run, before the outcome
        # exists, and it is the artifact every downstream script actually
        # loads, so burying the outcome in the FINAL checkpoint alone would
        # leave the one people read unable to say how its run ended.
        outcome_path = f'{self.result_dir}/scale_curriculum_outcome.json'
        # A process that trained nothing must not overwrite what an earlier one
        # recorded. Same reason as the guard above: a relaunch of a finished
        # run would replace its real outcome with this process's empty view.
        if self._scale_ctl is not None and (
                ran_any or not os.path.exists(outcome_path)):
            import json as _json
            hi = self._scale_stages[self._high_water_idx()]
            with open(outcome_path, 'w') as fh:
                _json.dump({
                    'outcome': self._scale_outcome,
                    'reached_size': f'{hi[0]}x{hi[1]}',
                    # 1-based, for a person reading the file and for the
                    # launcher's reached_stage == n_stages test. The 0-based
                    # INDICES are recorded alongside under explicit names,
                    # because the checkpoint provenance stores indices and a
                    # consumer that mixed the two conventions would be wrong by
                    # exactly one stage without anything raising.
                    'reached_stage': self._high_water_idx() + 1,
                    'final_stage': self._scale_ctl.stage_idx + 1,
                    'reached_stage_index': self._high_water_idx(),
                    'final_stage_index': self._scale_ctl.stage_idx,
                    'n_stages': len(self._scale_stages),
                    'ladder': [f'{s[0]}x{s[1]}' for s in self._scale_stages],
                    'demotions': {f'{self._scale_stages[i][0]}x'
                                  f'{self._scale_stages[i][1]}': n
                                  for i, n in
                                  (self._scale_ctl._demotions or {}).items()
                                  if 0 <= i < len(self._scale_stages)},
                    'final_stage_updates_total': self._scale_ctl._final_stage_updates,
                    # The raw per-update EV series of the CURRENT stage. The
                    # log only carries a window median every ten updates,
                    # which is too coarse to distinguish noisy validation from a slow
                    # climb - the question that decides whether an absolute
                    # advance threshold is the right gate at all.
                    'ev_hist_current_stage': [round(v, 6)
                                              for v in self._scale_ctl._ev_hist],
                    'ev_threshold': self._scale_ctl.ev_threshold,
                }, fh, indent=2)

        # Save model once at the end of training
        self._checkpoint_update_index = self._next_update - 1
        self.save_model()

        # Final resume snapshot, flagged complete so a relaunch skips this run.
        self._save_resume_checkpoint(next_update=self._next_update, completed=True)
        self._write_completion()

        # Finish wandb logging
        if self.wandb_enabled:
            wandb.finish()
    
    def save_model(self, model_name=None):
        """Save the model with all network hyperparameters for reproducibility."""
        if model_name is None:
            model_name = self.model_name
        # Extract all HGT network hyperparameters explicitly
        hgt_hyperparams = {
            'checkpoint_update_index': getattr(self, '_checkpoint_update_index', self._next_update - 1),
            'checkpoint_role': 'validation_selected' if model_name.startswith('best_') else 'final',
            'validation_fixture_sha256': self.validation_fixture_sha256,
            'training_events': self.training_events,
            # The run's TARGET size, not whichever ladder stage happened to be
            # training when this checkpoint was written. Best-model selection
            # runs on the target-size validation set at every stage, so a
            # checkpoint saved during stage 1 is still a target-size policy;
            # labelling it 10x5 would mis-describe every laddered artifact.
            # validation_n_j/n_m are pinned to the final stage and never move.
            'n_j': getattr(self, 'validation_n_j', self.config.n_j),
            'n_m': getattr(self, 'validation_n_m', self.config.n_m),
            'train_stage_n_j': self.config.n_j,
            'train_stage_n_m': self.config.n_m,
            'op_feature_dim': getattr(self.config, 'op_feature_dim', 9),
            'job_feature_dim': getattr(self.config, 'job_feature_dim', 7),
            'machine_feature_dim': getattr(self.config, 'machine_feature_dim', 6),
            'pt_feature_dim': getattr(self.config, 'pt_feature_dim', 2),
            'hidden_dim': getattr(self.config, 'hidden_dim', 32),
            'num_hgt_layers': getattr(self.config, 'num_hgt_layers', 4),
            'num_heads': getattr(self.config, 'num_heads', 2),
            'dropout': getattr(self.config, 'dropout', 0.1),
            'lr': getattr(self.config, 'lr', 3e-5),
            'gamma': getattr(self.config, 'gamma', 1.0),
            'gae_lambda': getattr(self.config, 'gae_lambda', 0.98),
            'eps_clip': getattr(self.config, 'eps_clip', 0.2),
            'k_epochs': getattr(self.config, 'k_epochs', 4),
            'max_grad_norm': getattr(self.config, 'max_grad_norm', 1.0),
            'vloss_coef': getattr(self.config, 'vloss_coef', 0.5),
            'ploss_coef': getattr(self.config, 'ploss_coef', 1.0),
            'objective_weights': self.objective_weights,
            'seed': self.seed,
            'reproducibility': dict(getattr(
                self.config, 'reproducibility', self.reproducibility)),
            # Recorded so a checkpoint is identifiable afterwards, and,
            # more importantly, because model_utils rebuilds a model's
            # config from THIS dict alone: without it every evaluation
            # script would load a PopArt checkpoint with the denormalizer
            # off and silently read normalized values as real ones.
            'use_popart': bool(getattr(self, 'use_popart', False)),
            'popart_beta': float(getattr(
                getattr(self.ppo, 'policy', None), 'popart_beta', 0.1)),
            'use_pt_nodes': bool(getattr(self.config, 'use_pt_nodes', True)),
            'use_dynamic_event_features': bool(getattr(
                self.config, 'use_dynamic_event_features', True)),
            'use_anchored_segments': bool(self.use_anchored_segments),
            'dp_lb_diff': {
                'enabled': self.use_lowerbound_differencing,
                'measure': self.reward_measure,
            },
            # Record the actual curriculum and normalization conditions; the
            # authorized control disables scale CL, including its PopArt normalization.
            # getattr throughout: a resumed or partially-built trainer must never
            # crash at save time, which would lose a multi-day run.
            'curriculum': {
                'scale_ladder': (
                    [list(s) for s in getattr(self, '_scale_stages', None) or []]
                    or None),
                # HIGH-WATER mark, not the current stage. With a bidirectional
                # ladder those differ: a run that climbs to 68x16 and then
                # oscillates back down would report the bottom stage, and how
                # far the ladder got IS this experiment's answer. The stage the
                # run ended on is recorded separately.
                'scale_stage_reached': (
                    getattr(getattr(self, '_scale_ctl', None),
                            'max_stage_idx', None)),
                'scale_stage_final': (
                    getattr(getattr(self, '_scale_ctl', None), 'stage_idx', None)),
                'scale_size_reached': (
                    'x'.join(str(v) for v in
                             self._scale_stages[self._high_water_idx()][:2])
                    if getattr(self, '_scale_ctl', None) is not None
                    and getattr(self, '_scale_stages', None) else None),
                # Keyed by SIZE, not by stage index. checkpoint_safe_metadata
                # stringifies every dict key for the weights-only loader, so an
                # int-keyed map would arrive as {'2': 1} and a reader indexing
                # it by stage number would miss. The size is also what someone
                # reading provenance actually wants to know.
                'scale_outcome': getattr(self, '_scale_outcome', None),
                'scale_demotions': ({
                    f'{self._scale_stages[i][0]}x{self._scale_stages[i][1]}': n
                    for i, n in sorted((getattr(self._scale_ctl, '_demotions', {})
                                        or {}).items())
                    if 0 <= i < len(self._scale_stages)
                } or None) if (getattr(self, '_scale_ctl', None) is not None
                               and getattr(self, '_scale_stages', None)) else None,
                'due_date_ramp': bool(getattr(self, 'use_curriculum', False)),
            },
            'actor_channels': self.ppo.policy.actor.channel_list,
            'critic_channels': self.ppo.policy.critic.channel_list,
        }

        save_obj = {
            'state_dict': self.ppo.policy.state_dict(),
            'hgt_hyperparams': checkpoint_safe_metadata(hgt_hyperparams),
            'config': checkpoint_safe_metadata(self.config),
            'resume_contract': self._resume_contract,
            'source_identity': self._source_identity,
        }
        self._save_torch_atomic(save_obj, f"{self.result_dir}/{model_name}.pth")
        return save_obj

    # ------------------------------------------------------------------
    # Resumable full-state checkpoint (distinct from the best/final model)
    # ------------------------------------------------------------------

    def resume_checkpoint_path(self):
        """Canonical resume-snapshot path inside this run's result dir."""
        return os.path.join(self.result_dir, 'resume_state.pth')

    def _save_resume_checkpoint(self, next_update, completed=False):
        """Atomically write a full training-state snapshot for resume.

        Stores everything needed to continue training byte-for-byte-ish: the
        policy / policy_old weights, the Adam optimizer state, the global RNG
        states, the next update index, the best validation objective so far, and
        the W&B run id (to rejoin the same run). The curriculum pacer is a pure
        function of the update index, so it is reconstructed for free. Written to
        a temp file then os.replace()'d so a kill mid-write can't corrupt it.
        """
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        state = {
            'next_update': int(next_update),
            'training_settings': getattr(self.config, 'training_settings', None),
            'resume_contract': self._resume_contract,
            'stopping_state': self._stop_ctl.state_dict(),
            'source_identity': self._source_identity,
            'validation_fixture_sha256': self.validation_fixture_sha256,
            'cumulative_training_seconds': self._training_seconds,
            'max_updates': int(self.max_updates),
            'completed': bool(completed),
            'actual_total_updates': int(next_update),
            'selected_update_index': self._selected_update_index,
            'best_model_payload': self._best_model_payload,
            'early_stop_outcome': self._early_stop_outcome,
            'early_stop_update': self._early_stop_update,
            'scale_outcome': self._scale_outcome,
            'best_val_objective': float(self.best_val_objective),
            # Persisted because best_val_objective is. Restoring the bar
            # while resetting the patience would hand every relaunch a
            # fresh budget against an unchanged best, and on a box where
            # earlyoom SIGTERMs long runs that makes the early-stopping rule
            # unreachable and the arm reports a false cap alarm.
            'val_no_improve': int(self._val_no_improve),
            'model_name': self.model_name,
            'seed': self.seed,
            'policy_state_dict': self.ppo.policy.state_dict(),
            'policy_old_state_dict': self.ppo.policy_old.state_dict(),
            # This flag is deliberately outside the model state_dict: loading
            # policy weights still means a warm start, while resuming an
            # initial (zero-update) snapshot must retain cold-start PopArt.
            'popart_head_state': {
                'policy': bool(getattr(self.ppo.policy, '_popart_head_is_warm', True)),
                'policy_old': bool(getattr(self.ppo.policy_old, '_popart_head_is_warm', True)),
            },
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'wandb_run_id': self._wandb_run_id,
            'scale_curriculum_state': (
                self._scale_ctl.state_dict()
                if getattr(self, '_scale_ctl', None) is not None else None),
            'rng': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.random.get_rng_state(),
                'cuda': cuda_rng,
            },
        }
        self._save_torch_atomic(state, self.resume_checkpoint_path())

    @staticmethod
    def _save_torch_atomic(state, path):
        path = os.fspath(path)
        with open(path + '.tmp', 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path + '.tmp', path)

    def _load_policy_weights(self, path):
        """Warm start: load ONLY the policy weights from another run.

        Accepts either a training snapshot (``resume_state.pth``, key
        ``policy_state_dict``) or a saved model (``best_*.pth`` / ``*.pth``,
        key ``state_dict``). Optimizer state, RNG state, update index and
        best-validation score are deliberately NOT restored: this starts a new
        run that happens to begin from trained weights.

        Raises if the checkpoint's architecture does not match the current
        config, so a silent partial load can never happen.
        """
        ckpt = torch.load(path, map_location=self.ppo.device,
                          weights_only=False)
        if 'policy_state_dict' in ckpt:
            state = ckpt['policy_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            raise KeyError(
                f"--init_from checkpoint {path!r} has neither "
                "'policy_state_dict' nor 'state_dict'")

        missing, unexpected = self.ppo.policy.load_state_dict(
            state, strict=False)
        # PopArt buffers are new state, so a pre-PopArt checkpoint
        # legitimately lacks them and must still warm-start. Any OTHER
        # missing or unexpected key stays fatal.
        # Only tolerate ABSENT popart buffers (a pre-PopArt checkpoint).
        # An UNEXPECTED popart key means the checkpoint carries a DIFFERENT
        # PopArt schema, e.g. the pre-rename popart_ema_nu, and swallowing it
        # silently loads zeroed statistics onto a warm critic.
        known = set(getattr(type(self.ppo.policy), 'POPART_BUFFER_KEYS', ()))
        missing = [k for k in missing
                   if not (k.startswith('popart_') and k in known)]
        if missing or unexpected:
            raise RuntimeError(
                f"--init_from architecture mismatch for {path!r}: "
                f"missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
        self.ppo.policy_old.load_state_dict(self.ppo.policy.state_dict())
        print(f"[init_from] warm-started policy weights from {path} "
              "(fresh optimizer, update counter at 0)")

    def _load_resume_checkpoint(self, path):
        """Restore training state saved by :meth:`_save_resume_checkpoint`.

        Loads onto the training device (so model + optimizer state land on the
        right device); RNG state tensors are forced back to CPU because
        ``set_rng_state`` requires CPU ByteTensors. Sets ``self.start_update``,
        ``self.best_val_objective`` and ``self._resume_wandb_id`` for the caller.
        """
        # weights_only=False: our own trusted snapshot carries NumPy/Python RNG
        # state (not plain tensors), which the PyTorch>=2.6 weights_only loader
        # rejects. Safe here because we wrote this file ourselves.
        ckpt = torch.load(
            path, map_location=self.ppo.device, weights_only=False)

        validate_resume_contract(ckpt.get('resume_contract'), self._resume_contract,
                                 ckpt.get('optimizer_state_dict'), self.config.lr)
        if ckpt.get('source_identity') != self._source_identity:
            raise ValueError('Resume scientific source differs from the saved source')
        if ckpt.get('validation_fixture_sha256') != self.validation_fixture_sha256:
            raise ValueError('Resume validation fixture differs')
        count = ckpt.get('next_update')
        if type(count) is not int or not 0 <= count <= self.max_updates:
            raise ValueError('Invalid committed update count in resume snapshot')
        selected = ckpt.get('selected_update_index')
        payload = ckpt.get('best_model_payload')
        if selected is None:
            if payload is not None or math.isfinite(float(ckpt.get('best_val_objective', float('inf')))):
                raise ValueError('Resume best selection metadata is inconsistent')
        elif (type(selected) is not int or not 0 <= selected < count
              or not isinstance(payload, dict)
              or payload.get('hgt_hyperparams', {}).get('checkpoint_update_index') != selected):
            raise ValueError('Resume selected checkpoint is outside the committed prefix')
        TrainingJournal(self.result_dir, self.seed, self.validation_fixture_sha256,
                        count, repair=False)
        self._stop_ctl.load_state_dict(ckpt.get('stopping_state'))
        self._training_seconds = float(ckpt['cumulative_training_seconds'])
        same_budget = int(ckpt.get('max_updates', self.max_updates)) == int(self.max_updates)
        self._resume_completed = bool(ckpt.get('completed')) and same_budget
        if self._resume_completed:
            print(f"[resume] {path} is already complete "
                  f"(update {ckpt.get('next_update')}/{ckpt.get('max_updates')}); "
                  "nothing left to train.")

        self.ppo.policy.load_state_dict(ckpt['policy_state_dict'])
        if 'policy_old_state_dict' in ckpt:
            self.ppo.policy_old.load_state_dict(ckpt['policy_old_state_dict'])
        else:
            self.ppo.policy_old.load_state_dict(ckpt['policy_state_dict'])
        # Older snapshots do not contain this additive metadata. Preserve
        # their existing warm-load semantics rather than guessing whether a
        # zero-update snapshot came from random or pretrained initialization.
        popart_head_state = ckpt.get('popart_head_state', {})
        for name in ('policy', 'policy_old'):
            if name in popart_head_state:
                flag = popart_head_state[name]
                if type(flag) is not bool:
                    raise ValueError(f'Invalid PopArt warm-head state for {name}')
                getattr(self.ppo, name)._popart_head_is_warm = flag
        self.ppo.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.best_val_objective = float(
            ckpt.get('best_val_objective', float('inf')))
        self._val_no_improve = self._stop_ctl.non_improving
        self.start_update = int(ckpt.get('next_update', 0))
        self._next_update = self.start_update
        self._selected_update_index = ckpt.get('selected_update_index')
        self._best_model_payload = ckpt.get('best_model_payload')
        # Restore the committed best artifact, undoing a newer selection
        # written before an interrupted update's snapshot. Old snapshots
        # lacking this metadata keep their historical on-disk artifact.
        if 'best_model_payload' in ckpt:
            best_path = os.path.join(self.result_dir, f'best_{self.model_name}.pth')
            if self._best_model_payload is not None:
                self._save_torch_atomic(self._best_model_payload, best_path)
            elif os.path.exists(best_path):
                os.unlink(best_path)
        if same_budget or not ckpt.get('completed'):
            self._early_stop_outcome = ckpt.get('early_stop_outcome')
            self._early_stop_update = ckpt.get('early_stop_update')
            self._resumed_scale_outcome = ckpt.get('scale_outcome')
        self._resume_wandb_id = ckpt.get('wandb_run_id', None)
        # Consumed by the scale-curriculum setup block in __init__, which runs
        # after this load: restores stage index, in-stage update count and the
        # gate's value-loss window so a killed run resumes mid-ladder.
        self._scale_resume_state = ckpt.get('scale_curriculum_state')

        for path_to_selection in Path(self.result_dir).glob('selected_update_*.pth'):
            if int(path_to_selection.stem.rsplit('_', 1)[1]) > self.start_update:
                path_to_selection.unlink()
        if self._best_model_payload is not None:
            self._save_torch_atomic(self._best_model_payload,
                Path(self.result_dir) / f'selected_update_{self._selected_update_index + 1:06d}.pth')
        self._resume_rng = ckpt['rng']
        self._restore_rng(self._resume_rng)

        print(f"[resume] Loaded {path}: continuing from update "
              f"{self.start_update}/{self.max_updates}, "
              f"best_val_objective={self.best_val_objective:.4f}")

    @staticmethod
    def _restore_rng(rng):
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.random.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])

    def _write_completion(self):
        outcome = self._early_stop_outcome or self._scale_outcome or (
            'fixed_budget_completed' if self.stopping_rule == 'fixed_budget' else 'budget_exhausted')
        record = dict(outcome=outcome, actual_total_updates=self._next_update,
            selected_update_index=self._selected_update_index,
            best_validation_objective=self.best_val_objective,
            stopping=self._stop_ctl.state_dict(),
            validation_fixture_sha256=self.validation_fixture_sha256)
        atomic_bytes(Path(self.result_dir) / 'completion.json',
                     (json.dumps(json_value(record), sort_keys=True, allow_nan=False) + '\n').encode())

    def load_model(self, model_path=None):
        """
        Load the trained model.
        
        The network is reconstructed using the hyperparameters saved with the model,
        ensuring compatibility even if the current config has different settings.
        """
        if model_path is None:
            model_path = f"{self.result_dir}/{self.model_name}.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        loaded = torch.load(model_path, map_location=self.ppo.device if hasattr(self, 'ppo') else 'cuda')
        
        if isinstance(loaded, dict) and 'state_dict' in loaded:
            # If saved hyperparams exist, use them to reconstruct the network
            if 'hgt_hyperparams' in loaded:
                saved_hp = loaded['hgt_hyperparams']
                self.saved_hgt_hyperparams = saved_hp
                
                # Update config with saved hyperparameters for network reconstruction
                for param, value in saved_hp.items():
                    setattr(self.config, param, value)
                
                # Reinitialize PPO with the saved hyperparameters
                self.ppo = PPO(self.config)
            
            self.ppo.policy.load_state_dict(loaded['state_dict'])
            self.saved_config = loaded.get('config', None)

    # Removed CL generation helpers; data must be prepared before training
