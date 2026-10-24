"""
Configuration for graph-based RL experiments (HGT).

Split into two parts:
- TrainingConfig: everything needed to train a model
- ExperimentConfig: everything needed to run inference / benchmarks

Helper functions (create_hgt_config, create_inference_config, etc.)
live in src/utils/config_util.py.
"""

from pathlib import Path


# ------------------------------------------------------------------ #
# Training Configuration
# ------------------------------------------------------------------ #

class TrainingConfig:
    """Everything needed to train a model."""

    def __init__(self):
        # Problem parameters (base values; curriculum dynamic-event params
        # are merged in by create_hgt_config).
        self.problem_params = {
            'job_num': 20,
            'machine_num': 10,
            # Due Date Tightness factor k ~ U[ddt_min, ddt_max]
            'due_date_tightness': [0.5, 1.5],
            # Urgency Degree (Weight) ~ U_int[weight_min, weight_max]
            'urgency_weight': [1, 5],
            # Operations per job ~ U_int[ops_min, ops_max]
            'operation_per_job': [1, 10],
            # Compatible machines per operation ~ U_int[mch_min, mch_max]
            'machine_per_operation': [1, 10],
            # Process Time ~ U[pt_min, pt_max]
            'process_time': [1, 50],
        }

        # Objective weights (saved in checkpoint for critic pruning)
        self.objective_weights = {
            'makespan_weight': 0.5,
            'twt_weight': 0.5,
        }

        # Training hyperparameters
        self.training = {
            'lr': 3e-4,
            'gamma': 1.0,
            'k_epochs': 4,
            'eps_clip': 0.2,
            'vloss_coef': 0.5,
            'ploss_coef': 1.0,
            'gae_lambda': 0.98,
            'minibatch_size': 256,
            'max_grad_norm': 1.0,
            'entropy_coef': 0.0,
            'hidden_dim': 16,
            'num_hgt_layers': 3,
            'mlp_hidden_dims': [128, 64, 32],
            'num_heads': 1,
            'dropout': 0.1,
        }

        # Validation
        self.validation = {
            'seed': 42,
            'validation_set_size': 30,
        }

        self.curriculum_learning = {
            'enabled': True,
            'initial_due_date_tightness': [1.0, 2.5],
            'ddt_stage_ratio': 0.60,
            'ddt_min_substage_ratio': 0.5,
            'events_enabled': True,
            'dynamic_event_params': {
                'num_new_job_arrival_waves': [2, 2],
                'num_machine_breakdown_waves': [2, 2],
                'num_priority_change_waves': [0, 0],
                # machine breakdowns — match capacity_shock
                'mtbf': [90, 120],
                'machines_per_breakdown': [3, 5],
                'repair_duration': [120, 120],
                # new-job arrivals — match priority_shock
                'new_job_lambda_inv': [90, 120],
                'new_jobs_per_arrival': [3, 4],
                'new_job_weight': [5, 5],
                'new_job_due_date_tightness': [0.3, 0.8],
            },
        }

        self.static_design = {
            'use_dynamic_event_features': False,
            'use_anchored_segments': False,
            'ddt_stage_ratio': 1.0,
            'events_enabled': False,
        }


        # Common training settings
        self.device = 'cuda'
        self.seed = 42
        self.wandb_project = 'PEACH'
        # Run controls come from training/settings.json via build_overrides.
        self.reset_env_episode = 1

    @property
    def common_params(self):
        """DataGenerator compatibility."""
        n_j = self.problem_params['job_num']
        n_m = self.problem_params['machine_num']
        return {
            'device': self.device,
            'wandb_project': self.wandb_project,
            'model_name': f'{n_j}X{n_m}_static',
        }


# ------------------------------------------------------------------ #
# Experiment Configuration
# ------------------------------------------------------------------ #

class ExperimentConfig:
    """Everything needed to run inference / benchmarks."""

    def __init__(self):
        # Problem params shared by inference / benchmark scripts. Static only:
        # dynamic-event defaults are centralized in ``_EVENT_RANGE_DEFAULTS``
        # (src/utils/config_util.py) and applied via normalize_generation_ranges,
        # so they are not duplicated here. Scripts that need events set them
        # explicitly (e.g. exp_large_scale_dispatching via ``large_scale_params``).
        self.problem_params = {
            'job_num': 20,
            'machine_num': 10,
            'due_date_tightness': [0.5, 1.5],
            'urgency_weight': [1, 5],
            'operation_per_job': [1, 10],
            'machine_per_operation': [1, 10],
            'process_time': [1, 50],
        }

        # Large-scale dispatching-rule baseline experiment.
        # Expanded paper-aligned menu: original fixed rules plus tunable ATC
        # and COVERT broken out by k-value. Includes priority changes because
        # this is a multi-criteria objective with TWT.
        self.large_scale_params = {
            'job_num': 100,
            'machine_num': 20,

            # Two complementary disruption archetypes, swept together by
            # exp_large_scale_dispatching.py (no single default scenario). Each
            # is self-contained: it lists its own wave counts (which event
            # types fire) and the intensity of those events.
            #   priority_shock - demand-side: rush arrivals + reprioritization,
            #                    no capacity loss (stresses responsiveness/TWT).
            #   capacity_shock - supply-side: simultaneous multi-machine
            #                    breakdowns, no new demand (stresses rerouting).
            'scenarios': {
                'priority_shock': {
                    'num_new_job_arrival_waves': [2, 2],
                    'num_machine_breakdown_waves': [0, 0],
                    'num_priority_change_waves': [2, 2],
                    # Arrival times are cumulative exponential gaps (wave i mean
                    # ≈ i × mean_gap). The 100x20 greedy horizon is ≈740, so the
                    # 2 arrival waves land within the first ~60% (mean ~14% &
                    # 28%). exp_large_scale_dispatching couples each priority
                    # change to an arrival wave — they fire at the SAME time (one
                    # demand event = rush arrival + reprioritization) — so
                    # priority_change_interval is unused here (it stays only as
                    # the generator default). See project_event_timing_horizon.
                    'new_job_lambda_inv': [90, 120],    # 2 arrivals → ~mean 14% & 28% of H
                    'priority_change_interval': [120, 150],  # unused: priority coupled to arrivals in-exp
                    'new_jobs_per_arrival': [3, 4],
                    'new_job_weight': [5, 5],
                    'new_job_due_date_tightness': [0.3, 0.8],
                    'priority_jobs_per_change': [8, 12],
                    'priority_new_weight': [5, 5],
                },
                'capacity_shock': {
                    'num_new_job_arrival_waves': [0, 0],
                    'num_machine_breakdown_waves': [2, 2],
                    'num_priority_change_waves': [0, 0],
                    # 2 breakdown waves (cumulative exponential gaps, mean=mtbf).
                    # 100x20 greedy horizon ≈700; mtbf≈105 places the waves at
                    # ~mean 15% & 30% so both typically land within the first
                    # ~60% of the schedule (soft target via the parameter, not a
                    # hard cap). The old mtbf=475 fired waves 2–3 past makespan.
                    'mtbf': [90, 120],    # 2 breakdowns → ~mean 15% & 30% of H≈700
                    'machines_per_breakdown': [3, 5],
                    'repair_duration': [120, 120],
                },
            },
            'rules': ['FIFO', 'SPT', 'EDD', 'WSPT',
                      'ATC_1.0', 'ATC_2.0', 'ATC_3.0', 'ATC_4.0', 'MS',
                      'COVERT_1.0', 'COVERT_2.0', 'COVERT_3.0', 'COVERT_4.0',
                      'CR', 'MOD', 'AVPRO', 'NPT'],
        }


        # Solver parameters
        self.solver = {
            # CP-SAT (OR-Tools) reference baseline wall-clock budget.
            'cp_time_limit': 14400,
            # CP-SAT search workers; report this count with every result.
            # Use 1 for a reproducible sequential search.
            'cp_num_workers': 8,
            'cp_verbose': 0,
            # Integer ticks per time unit for CP-SAT's search grid. Processing
            # times and job weights are already integral, so feasibility is
            # exact at any scale; release/breakdown/fixed-op values are rounded
            # CONSERVATIVELY (outward) and the returned schedule is replayed in
            # exact float arithmetic (see src/benchmark/CP.py).
            #
            # This is a CORRECTNESS knob, not just a precision knob. Conservative
            # rounding shrinks the feasible region, so any true slack narrower
            # than one tick is erased — and that can flip a sequencing decision
            # while CP still reports status='Optimal' with relative_gap=0. The
            # objective error is NOT bounded by 1/scale even though the time
            # error is. Raised from 10 to 100 on 2026-08-02 on this evidence:
            #   - bench/difftest_cp_model.py: 4/4000 instances came out worse
            #     than the exhaustive optimum at scale 10 (worst +48%), 0/4000
            #     at scale 100.
            #   - a 15x10 benchmark instance returned a gridded-optimal tie
            #     0.009 worse than Gurobi's grid-free optimum at scale 10.
            #   - cost is nil: 20x10 total solve time 17.2s / 19.0s / 15.4s at
            #     scale 10 / 100 / 1000 (run-to-run noise exceeds the trend),
            #     while the grid distortion of the optimised objective falls
            #     0.039% -> 0.010% -> 0.000%.
            # Check performance['grid_exact'] before calling any CP result a
            # proven optimum of the float problem.
            'cp_time_scale': 100,
        }

        # Experiment settings
        self.model_path = str(Path(__file__).resolve().parent /
                              'models/checkpoints/generalization-scale-cl-20x10-s42.pth')
        self.num_runs = 5
        self.device = 'cuda'
        self.seed = 42

    @property
    def common_params(self):
        """DataGenerator compatibility."""
        return {'device': self.device}


# ------------------------------------------------------------------ #
# Single-Objective Training Configuration (makespan only, no TWT)
# ------------------------------------------------------------------ #

class TrainingConfigSingleObj:
    """Makespan-only training config (DANIEL SD1/SD2 replication).

    Standalone — does NOT inherit ``TrainingConfig``. The single-obj
    pipeline (``src/HGT_single_objective/``) is already makespan-only,
    has no curriculum, no dynamic events, no TWT weights; so carrying
    those knobs here just to disable them would be noise.

    Per-run knobs (``n_j``, ``n_m``, ``generator_type``) come from CLI via
    ``create_hgt_single_obj_config``; the SD1/SD2 distribution constants
    live in ``DataGeneratorSingleObj``, not here.
    """

    def __init__(self):
        # PPO / HGT hyperparameters
        self.training = {
            'lr': 3e-4,
            'gamma': 1.0,
            'k_epochs': 4,
            'eps_clip': 0.2,
            'vloss_coef': 0.5,
            'ploss_coef': 1.0,
            'gae_lambda': 0.98,
            'minibatch_size': 256,
            'max_grad_norm': 1.0,
            'entropy_coef': 0.0,
            'hidden_dim': 16,
            'num_hgt_layers': 3,
            'mlp_hidden_dims': [128, 64, 32],
            'num_heads': 1,
            'dropout': 0.1,
        }

        self.validation = {
            'seed': 42,
            'validation_set_size': 30,
        }

        # Schedule + misc
        self.device = 'cuda'
        self.seed = 42
        self.wandb_project = 'PEACH'
        self.reset_env_episode = 1


# ------------------------------------------------------------------ #
# Global singletons
# ------------------------------------------------------------------ #
training_config = TrainingConfig()
training_config_single_obj = TrainingConfigSingleObj()
experiment_config = ExperimentConfig()

# Backward-compat alias
config = experiment_config
