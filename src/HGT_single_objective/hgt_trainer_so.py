"""Static makespan PPO with scale stages and recoverable training evidence."""
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import wandb

from src.HGT.hgt_trainer import Memory, PPO as BasePPO
from src.utils.checkpoint_util import (checkpoint_safe_metadata, training_contract,
                                       validate_resume_contract)
from src.utils.reproducibility import derive_seed, seed_context, set_global_seed
from src.utils.rl_utils import sample_action, greedy_select_action
from src.utils.scale_curriculum import (ScaleCurriculumController, ADVANCE, DESCEND,
                                        DIVERGED, STUCK)
from src.utils.training_settings import ValidationStopController, require_training_controls
from src.data_generator.data_generator_single_obj import create_data_generator_single_obj
from .hgt_env_so import FJSPEnvHGTSingleObj
from .hgt_network_so import HGTSingleObj


class PPOSingleObj(BasePPO):
    policy_cls = HGTSingleObj


def _json_value(value):
    value = checkpoint_safe_metadata(value)
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class HGTSingleObjTrainer:
    """No events or TWT; validation remains fixed at the requested target size."""
    SNAPSHOT_VERSION = 1

    def __init__(self, config):
        require_training_controls(config, scale_required=True)
        self.config = config
        if config.training_events:
            raise ValueError('Single-objective public training cannot enable events')
        config.training_events = False
        config.use_dynamic_event_features = False
        config.objective_weights = {'makespan_weight': 1.0, 'twt_weight': 0.0}
        self.n_j, self.n_m = int(config.n_j), int(config.n_m)
        self.num_envs = int(config.num_envs)
        self.max_updates = int(config.max_updates)
        self.reset_env_episode = int(config.reset_env_episode)
        if self.reset_env_episode != 1:
            raise ValueError('Static resumable protocol requires reset_env_episode=1')
        self.seed = int(config.seed)
        self.reproducibility = set_global_seed(self.seed)
        config.reproducibility = self.reproducibility
        self.result_dir = str(config.result_dir)
        self.root = Path(self.result_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_name = config.model_name
        self.data_name = f'{self.n_j}x{self.n_m}'
        self.use_lowerbound_differencing = True
        self.update_per_validate = int(config.update_per_validate)
        if self.update_per_validate < 1 or self.max_updates < 1:
            raise ValueError('Validation interval and maximum updates must be positive')
        self.validation_seed = derive_seed(config.validation_seed, 'validation_fixture')
        self.validation_set_size = int(config.validation_set_size)
        self.data_generator = self._generator(self.n_j, self.n_m)
        self.env = FJSPEnvHGTSingleObj(self.n_j, self.n_m, device=config.device)
        for key, value in [('op_feature_dim', self.env.op_fea_dim),
                           ('job_feature_dim', self.env.job_fea_dim),
                           ('machine_feature_dim', self.env.mch_fea_dim),
                           ('pt_feature_dim', self.env.pt_fea_dim)]:
            setattr(config, key, value)
        self.ppo = PPOSingleObj(config)
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)
        self.validation_data = self._generate_validation_set()
        fixture = checkpoint_safe_metadata(self.validation_data)
        fixture_bytes = (json.dumps(fixture, sort_keys=True, allow_nan=False) + '\n').encode()
        self.validation_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
        self._fixture_bytes = fixture_bytes
        fixture_path = self.root / 'validation-fixture.json'
        if fixture_path.exists() and fixture_path.read_bytes() != fixture_bytes:
            raise ValueError('Existing validation fixture differs')
        sc = deepcopy(config.scale_curriculum)
        self.stages = [tuple(s) for s in sc.pop('stages')]
        if self.stages[-1][:2] != (self.n_j, self.n_m):
            raise ValueError('Last static scale stage must equal the validation target')
        sc.pop('enabled', None)
        allowed = set(inspect.signature(ScaleCurriculumController).parameters) - {'n_stages', 'ladder_id'}
        sc = {k: v for k, v in sc.items() if k in allowed}
        sc['final_hold_updates'] = None
        self.scale_ctl = ScaleCurriculumController(len(self.stages),
            ladder_id=','.join(f'{j}x{m}:{micro}' for j,m,micro in self.stages), **sc)
        config.scale_curriculum = dict(sc, enabled=True, stages=self.stages)
        self.stop_ctl = ValidationStopController(
            rule=config.stopping_rule, patience=config.early_stop_patience)
        self.resume_contract = training_contract(config)
        repo = Path(__file__).resolve().parents[2]
        source_files = sorted((repo / 'src').rglob('*.py')) + [repo / 'config.py', repo / 'training/train_single_obj.py']
        self.source_identity = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
        self.next_update = 0
        self.best_val_makespan = float('inf')
        self.selected_update_index = None
        self.best_model_payload = None
        self.outcome = None
        self.completed = False
        self.cumulative_training_seconds = 0.0
        self.wandb_run_id = None
        self.wandb_enabled = bool(getattr(config, 'wandb_enabled', True))
        self.journal_path = self.root / 'train_metrics.jsonl'
        resume = getattr(config, 'resume_path', None)
        if resume:
            self._load_resume(resume)
        else:
            if (self.journal_path.exists() or (self.root / 'resume_state.pth').exists()
                    or (self.root / (self.model_name + '.pth')).exists()
                    or (self.root / ('best_' + self.model_name + '.pth')).exists()):
                raise ValueError('Existing training artifacts require an explicit resume')
            self._verify_or_create(fixture_path, fixture_bytes)
            self._apply_scale_stage()
            self._save_resume()
        if self.wandb_enabled and not self.completed:
            run = wandb.init(project=config.wandb_project,
                name=getattr(config, 'wandb_run_name', self.model_name),
                config=checkpoint_safe_metadata(config), id=self.wandb_run_id,
                resume='allow' if self.wandb_run_id else None)
            self.wandb_run_id = run.id

    def _generator(self, n_j, n_m):
        cfg = deepcopy(self.config)
        cfg.n_j, cfg.n_m = n_j, n_m
        # Reject accidental dynamic plumbing rather than generate ignored events.
        for name in ('num_new_job_arrivals', 'num_machine_breakdowns', 'num_priority_changes',
                     'num_new_job_arrival_waves', 'num_machine_breakdown_waves', 'num_priority_change_waves'):
            setattr(cfg, name, 0)
        cfg.problem_params = dict(getattr(cfg, 'problem_params', {}) or {})
        for name in list(cfg.problem_params):
            if name.startswith(('num_new_job_arrival', 'num_machine_breakdown', 'num_priority_change')):
                cfg.problem_params[name] = 0
        return create_data_generator_single_obj(cfg)

    def _apply_scale_stage(self):
        j, m, micro = self.stages[self.scale_ctl.stage_idx]
        self.train_n_j, self.train_n_m = j, m
        self.data_generator = self._generator(j, m)
        self.env = FJSPEnvHGTSingleObj(j, m, device=self.config.device)
        self.ppo.micro_batch_size = micro

    def sample_training_instances(self):
        return self.data_generator.sample_env(self.num_envs)

    def _generate_validation_set(self):
        with seed_context(self.validation_seed):
            return self._generator(self.n_j, self.n_m).sample_env(self.validation_set_size)

    @staticmethod
    def _policy_forward(policy, state):
        return policy(candidate=state.candidate_tensor, candidate_mask=state.candidate_mask_tensor,
                      candidate_pt_indices=state.candidate_pt_indices, graph=state.graph_tensor)

    def _greedy_rollout(self, val_data):
        env = FJSPEnvHGTSingleObj(self.n_j, self.n_m, device=self.config.device)
        state = env.set_initial_data_static(val_data)
        was_training = self.ppo.policy.training
        self.ppo.policy.eval()
        try:
            while True:
                with torch.no_grad():
                    pi, _ = self._policy_forward(self.ppo.policy, state)
                state, _, done = env.step(greedy_select_action(pi).cpu().numpy())
                if done.all():
                    break
        finally:
            self.ppo.policy.train(was_training)
        return float(np.mean(env.current_makespan))

    def _collect_update(self, update_index):
        rollout_started = time.perf_counter()
        state = self.env.set_initial_data_static(self.sample_training_instances())
        rewards = -self.env.init_quality.copy()
        while True:
            self.memory.push(state)
            with torch.no_grad():
                pi, vals = self._policy_forward(self.ppo.policy_old, state)
            actions, logprobs = sample_action(pi)
            state, reward, done = self.env.step(actions.cpu().numpy())
            rewards += reward
            self.memory.reward_seq.append(torch.from_numpy(reward).to(self.ppo.device))
            self.memory.action_seq.append(actions)
            self.memory.log_probs.append(logprobs)
            self.memory.val_seq.append(vals.squeeze(1))
            if done.all():
                break
        samples = sum(a.numel() for a in self.memory.action_seq)
        rollout_seconds = time.perf_counter() - rollout_started
        before = max((int(s.get('step', 0)) for s in self.ppo.optimizer.state.values()), default=0)
        ppo_started = time.perf_counter()
        _, _, _, metrics = self.ppo.update(self.memory, update_step=update_index,
            raw_rewards=rewards, raw_makespan=self.env.current_makespan,
            makespan_weight=1.0, twt_weight=0.0)
        metrics.update(samples=samples,
            optimizer_steps=max((int(s.get('step', 0)) for s in self.ppo.optimizer.state.values()), default=0) - before,
            rollout_seconds=rollout_seconds, ppo_seconds=time.perf_counter() - ppo_started)
        self.memory.clear_memory()
        return metrics

    def train(self, *, stop_after_updates=None):
        """Optional interruption boundary is for bounded CPU resume rehearsals."""
        if self.completed:
            self._write_completion()
            return self.outcome
        for i in range(self.next_update, self.max_updates):
            started = time.perf_counter()
            stage = self.scale_ctl.stage_idx
            metrics = self._collect_update(i)
            if any(not math.isfinite(float(metrics[k])) for k in ('value_loss', 'policy_loss')):
                raise FloatingPointError('Nonfinite PPO loss')
            decision = self.scale_ctl.observe(metrics['value_loss'], metrics.get('explained_variance'))
            self.next_update = i + 1
            self.stop_ctl.observe_gate(self.next_update, self.scale_ctl, decision=decision)
            terminal = decision in (DIVERGED, STUCK)
            validate = (i % self.update_per_validate == 0 or self.next_update == self.max_updates or terminal)
            val = None
            stop = False
            if validate:
                val = self._greedy_rollout(self.validation_data)
                if not math.isfinite(val):
                    raise FloatingPointError('Validation makespan must be finite')
                improved = val < self.best_val_makespan
                if improved:
                    self.best_val_makespan = val
                    self.selected_update_index = i
                    self.best_model_payload = deepcopy(self.save_model('best_' + self.model_name))
                stop = self.stop_ctl.observe_validation(self.next_update, val, improved=improved)
            if terminal:
                self.outcome = decision
            elif stop:
                self.outcome = 'early_stopped'
            elif self.next_update == self.max_updates:
                self.outcome = ('fixed_budget_completed' if self.stop_ctl.rule == 'fixed_budget' else 'budget_exhausted')
            elif decision in (ADVANCE, DESCEND):
                (self.scale_ctl.advance if decision == ADVANCE else self.scale_ctl.descend)()
                self._apply_scale_stage()
            self.completed = self.outcome is not None
            self.cumulative_training_seconds += time.perf_counter() - started
            row = dict(metrics, update_step=i, total_updates=self.next_update,
                seed=self.seed, training_events=False, training_size=list(self.stages[stage][:2]),
                stage_index=stage, stage_decision=decision, next_stage_index=self.scale_ctl.stage_idx,
                validation_makespan=val, validation_fixture_sha256=self.validation_sha256,
                best_validation_makespan=self.best_val_makespan,
                selected_update_index=self.selected_update_index,
                stopping=self.stop_ctl.state_dict(), outcome=self.outcome,
                cumulative_training_seconds=self.cumulative_training_seconds,
                scale=self.scale_ctl.gate_metrics())
            with self.journal_path.open('a') as f:
                f.write(json.dumps(_json_value(row), sort_keys=True, allow_nan=False) + '\n')
                f.flush(); os.fsync(f.fileno())
            self._save_resume()
            if self.wandb_enabled:
                wandb.log(_json_value(row), step=i)
            if self.completed or (stop_after_updates is not None and self.next_update >= stop_after_updates):
                break
        if self.completed:
            self._write_completion()
        if self.wandb_enabled:
            wandb.finish()
        return self.outcome

    def _write_completion(self):
        self.save_model()
        self._atomic_bytes(self.root / 'completion.json', (json.dumps({
            'outcome': self.outcome, 'actual_total_updates': self.next_update,
            'selected_update_index': self.selected_update_index,
            'best_validation_makespan': self.best_val_makespan,
            'validation_fixture_sha256': self.validation_sha256}, allow_nan=False) + '\n').encode())

    @staticmethod
    def _atomic_bytes(path, data):
        path = Path(path); temp = path.with_name(path.name + '.tmp')
        with temp.open('wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)

    @staticmethod
    def _atomic_torch(payload, path):
        path = Path(path); temp = path.with_name(path.name + '.tmp')
        with temp.open('wb') as f:
            torch.save(payload, f); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)

    def _verify_or_create(self, path, data):
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f'Existing fixture differs: {path}')
        else:
            self._atomic_bytes(path, data)

    def _save_resume(self):
        self._atomic_torch({
            'schema_version': self.SNAPSHOT_VERSION, 'next_update': self.next_update,
            'completed': self.completed, 'outcome': self.outcome,
            'resume_contract': self.resume_contract, 'source_identity': self.source_identity,
            'training_settings': getattr(self.config, 'training_settings', {}),
            'validation_fixture_sha256': self.validation_sha256,
            'best_val_makespan': self.best_val_makespan,
            'selected_update_index': self.selected_update_index,
            'best_model_payload': self.best_model_payload,
            'policy_state_dict': self.ppo.policy.state_dict(),
            'policy_old_state_dict': self.ppo.policy_old.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'popart_head_state': {name: getattr(getattr(self.ppo, name), '_popart_head_is_warm', True)
                                  for name in ('policy', 'policy_old')},
            'scale_curriculum_state': self.scale_ctl.state_dict(),
            'validation_stopping_state': self.stop_ctl.state_dict(),
            'cumulative_training_seconds': self.cumulative_training_seconds,
            'wandb_run_id': self.wandb_run_id,
            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.random.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }, self.root / 'resume_state.pth')

    def _load_resume(self, path):
        state = torch.load(path, map_location=self.ppo.device, weights_only=False)
        if state['schema_version'] != self.SNAPSHOT_VERSION:
            raise ValueError('Unsupported static resume schema')
        validate_resume_contract(state['resume_contract'], self.resume_contract,
                                 state['optimizer_state_dict'], self.ppo.lr)
        if state['source_identity'] != self.source_identity:
            raise ValueError('Resume source identity changed')
        if state['validation_fixture_sha256'] != self.validation_sha256:
            raise ValueError('Resume validation fixture changed')
        count = state['next_update']
        if type(count) is not int or not 0 <= count <= self.max_updates:
            raise ValueError('Invalid resume update boundary')
        if type(state['completed']) is not bool or state['completed'] != (state['outcome'] is not None):
            raise ValueError('Inconsistent terminal state')
        selected, payload = state['selected_update_index'], state['best_model_payload']
        if count == 0:
            if selected is not None or payload is not None:
                raise ValueError('Zero-update snapshot cannot have a selected model')
        elif (type(selected) is not int or not 0 <= selected < count or not isinstance(payload, dict)
              or not math.isfinite(float(state['best_val_makespan']))):
            raise ValueError('Missing or inconsistent validation selection')
        if payload is not None:
            hp = payload['hgt_hyperparams']
            if ((hp['n_j'], hp['n_m']) != (self.n_j, self.n_m)
                    or hp['checkpoint_update_index'] != selected):
                raise ValueError('Best payload identity does not match selected target checkpoint')
        if any(type(state['popart_head_state'][name]) is not bool for name in ('policy','policy_old')):
            raise ValueError('Invalid PopArt cold/warm state')
        self.scale_ctl.load_state_dict(state['scale_curriculum_state'])
        self.stop_ctl.load_state_dict(state['validation_stopping_state'])
        self._apply_scale_stage()
        for name, key in [('policy','policy_state_dict'), ('policy_old','policy_old_state_dict')]:
            policy = getattr(self.ppo, name)
            policy.load_state_dict(state[key])
            policy._popart_head_is_warm = state['popart_head_state'][name]
        self.ppo.policy_old.eval()
        self.ppo.optimizer.load_state_dict(state['optimizer_state_dict'])
        for key in ('next_update', 'completed', 'outcome', 'best_val_makespan',
                    'selected_update_index', 'best_model_payload',
                    'cumulative_training_seconds', 'wandb_run_id'):
            setattr(self, key, state[key])
        # Journal append may have committed before the snapshot. Roll back only
        # the uncommitted tail; never continue from a missing or discontinuous prefix.
        lines = self.journal_path.read_bytes().splitlines() if self.journal_path.exists() else []
        if len(lines) < self.next_update:
            raise ValueError('Journal is shorter than the saved update boundary')
        for i, line in enumerate(lines[:self.next_update]):
            row = json.loads(line)
            if (row['update_step'] != i or row['total_updates'] != i + 1
                    or row['seed'] != self.seed
                    or row['validation_fixture_sha256'] != self.validation_sha256):
                raise ValueError('Noncontiguous committed journal')
        self._verify_or_create(self.root / 'validation-fixture.json', self._fixture_bytes)
        self._atomic_bytes(self.journal_path, b''.join(line+b'\n' for line in lines[:self.next_update]))
        best_path = self.root / ('best_' + self.model_name + '.pth')
        if self.best_model_payload is not None:
            self._atomic_torch(self.best_model_payload, best_path)
        elif best_path.exists():
            best_path.unlink()
        if self.completed:
            self._write_completion()
        else:
            for name in (self.model_name + '.pth', 'completion.json'):
                (self.root / name).unlink(missing_ok=True)
        rng = state['rng']
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.random.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in rng['cuda']])
    def save_model(self, model_name=None):
        if model_name is None:
            model_name = self.model_name
        hgt_hyperparams = {
            'checkpoint_role': 'validation_selected' if model_name == 'best_' + self.model_name else 'final',
            'checkpoint_update_index': self.next_update - 1,
            'training_stage_size': [self.train_n_j, self.train_n_m],
            'selection_criterion': 'best_target_validation_makespan' if model_name == 'best_' + self.model_name else 'last_completed_update',
            'n_j': self.config.n_j,
            'n_m': self.config.n_m,
            'op_feature_dim': getattr(self.config, 'op_feature_dim', 9),
            'job_feature_dim': getattr(self.config, 'job_feature_dim', 4),
            'machine_feature_dim': getattr(self.config, 'machine_feature_dim', 6),
            'hidden_dim': getattr(self.config, 'hidden_dim', 128),
            'num_hgt_layers': getattr(self.config, 'num_hgt_layers', 4),
            'num_heads': getattr(self.config, 'num_heads', 1),
            'dropout': getattr(self.config, 'dropout', 0.1),
            'lr': getattr(self.config, 'lr', 3e-5),
            'gamma': getattr(self.config, 'gamma', 1.0),
            'gae_lambda': getattr(self.config, 'gae_lambda', 0.98),
            'eps_clip': getattr(self.config, 'eps_clip', 0.2),
            'k_epochs': getattr(self.config, 'k_epochs', 4),
            'max_grad_norm': getattr(self.config, 'max_grad_norm', 1.0),
            'vloss_coef': getattr(self.config, 'vloss_coef', 0.5),
            'ploss_coef': getattr(self.config, 'ploss_coef', 1.0),
            'objective_weights': {'makespan_weight': 1.0, 'twt_weight': 0.0},
            'seed': self.seed,
            'reproducibility': dict(getattr(
                self.config, 'reproducibility', self.reproducibility)),
            'actor_channels': self.ppo.policy.actor.channel_list,
            'critic_channels': self.ppo.policy.critic.channel_list,
            'single_objective': True,
            'use_pt_nodes': True,
            # Old/static design: no dynamic-event observability columns
            # (mch_fea_dim=6). Mirrors train_graph_rl_static.py. Informational
            # for checkpoint inspection; the network is rebuilt from the saved
            # *_feature_dim values and the SO env hardwires this design.
            'use_dynamic_event_features': False,
            'dp_lb_diff': {
                'enabled': self.use_lowerbound_differencing,
            },
            'lowerbound_features': {
                'enabled': True,
            },
            'use_popart': self.ppo.policy.use_popart,
            'popart_beta': self.ppo.policy.popart_beta,
            'training_settings': getattr(self.config, 'training_settings', {}),
            'scale_curriculum_state': self.scale_ctl.state_dict(),
            'validation_stopping_state': self.stop_ctl.state_dict(),
            'actual_total_updates': self.next_update,
            'selected_update_index': self.selected_update_index,
            'validation_fixture_sha256': self.validation_sha256,
            'source_identity': self.source_identity,
            'generator_type': getattr(self.config, 'generator_type', 'SD1'),
        }
        hgt_hyperparams['pt_feature_dim'] = getattr(
            self.config, 'pt_feature_dim', 2)
        save_obj = {
            'state_dict': self.ppo.policy.state_dict(),
            'hgt_hyperparams': checkpoint_safe_metadata(hgt_hyperparams),
            'config': checkpoint_safe_metadata(self.config),
        }
        self._atomic_torch(save_obj, self.root / f'{model_name}.pth')
        return save_obj
