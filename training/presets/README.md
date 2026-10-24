# Recorded ablation training controls

These 35 presets contain the exact named controls from the five-seed ablation records. Each file corresponds to `runs/<preset name>/config.json` in [results/ablations.tar.gz](../../results/ablations.tar.gz). Reporting fields such as `selection` and `resolved_scale_stages` remain in that archive; presets contain only fields accepted by the training interface. The recorded device is `cuda:0`; an explicit `--device cpu` override changes the execution device.

Full checkpoint configurations and selected-update metadata remain in [experiments/configs](../../experiments/configs) and [models/manifest.json](../../models/manifest.json). These metadata documents are not `--settings` files.

## Inspect or start a new run

Run commands from the repository root. `--print_config` only resolves configuration; it does not train or allocate a policy.

```bash
uv run python training/ablation_training_stability.py \
  --ablation baseline --seed 42 \
  --settings training/presets/component_baseline_s42.json \
  --disable_wandb --print_config

uv run python training/train_graph_rl.py \
  --settings training/presets/cl_peach_s42.json \
  --experiment_name cl_peach_s42 --disable_wandb --print_config
```

To start a new run with the active implementation, remove `--print_config` and select a new output directory with `--result_dir result/<new-run-name>`. Keep the preset seed and the component launcher's `--seed` identical. The component launcher explicitly selects the condition; the filename alone does not select it.

| Preset prefix | Entrypoint | Required condition argument |
| :-- | :-- | :-- |
| `component_baseline` | `training/ablation_training_stability.py` | `--ablation baseline` |
| `component_chain_lb_reward` | `training/ablation_training_stability.py` | `--ablation chain_lb_reward` |
| `component_no_dp_lb_diff` | `training/ablation_training_stability.py` | `--ablation no_dp_lb_diff` |
| `component_no_pt_node` | `training/ablation_training_stability.py` | `--ablation no_pt_node` |
| `cl_peach` | `training/train_graph_rl.py` | Condition is recorded in the preset |
| `cl_no_training_events` | `training/train_graph_rl.py` | Condition is recorded in the preset |
| `cl_direct` | `training/train_graph_rl.py` | Condition is recorded in the preset |

Every prefix has `_s42.json` through `_s46.json`. Component runs use 800 updates, 20 environments, validation every 200 updates, and no scale CL. Both adaptive curriculum conditions use 20 environments, validation every 20 updates, readiness-gated patience four, and a 20,000-update safety limit. The direct condition uses its recorded fixed budget, events, and no scale CL or PopArt.

## Full adaptive runs and matched prefixes

| Training seed | Matched budget |
| :-- | --: |
| 42 | 281 |
| 43 | 221 |
| 44 | 401 |
| 45 | 341 |
| 46 | 401 |

The two adaptive presets reproduce their full stopping configuration. Do not replace their safety limit with the matched budget: the comparison takes a prefix of the completed adaptive run. The direct preset trains for the corresponding matched budget. Checkpoint selection uses the best validation objective within that prefix, retaining the earlier checkpoint on ties. The combined figure uses the common range through update 221. Full journals, actual stopping updates, and prefix-selected model identities remain in the ablation archive and model manifest.

## Use the recorded scientific source

All 35 runs identify source `ab8e6e471a5e0913929450d29fd04ff547cb313d`. The active implementation is a later release, so matching these controls does not establish identical historical stopping or trajectories. The [historical source manifest](../../experiments/historical_sources.json) identifies the original trainer and its file hashes.

```bash
mkdir -p outputs/historical
tar -xzf experiments/historical_sources.tar.gz -C outputs/historical
```

The following example constructs the recorded configuration using the extracted source. It only prints configuration. Change the preset filename to select any of the 35 runs. The explicit component setup also supports seeds 44 through 46, which the historical component launcher's command-line parser did not accept.

```bash
uv run python - <<'PY'
import json
import sys
from pathlib import Path

root = Path.cwd()
source = root / "outputs/historical/versions/ab8e6e471a5e0913929450d29fd04ff547cb313d"
preset = root / "training/presets/component_no_pt_node_s44.json"
sys.path.insert(0, str(source))

from src.utils.training_settings import load_settings, build_overrides
from src.utils.config_util import create_hgt_config

settings = load_settings(preset)
overrides = build_overrides(settings)
overrides.update(validation_seed=42, validation_set_size=30,
                 checkpoint_every=1, wandb_enabled=False)
condition = preset.stem.rsplit("_s", 1)[0]
if condition.startswith("component_"):
    variant = condition.removeprefix("component_")
    overrides.update(use_pt_nodes=variant != "no_pt_node",
                     component_ablation=variant,
                     dp_lb_diff={"enabled": variant != "no_dp_lb_diff",
                                 "measure": "chain" if variant == "chain_lb_reward" else "dp"})
    overrides["training_settings"]["component_ablation"] = variant
config = create_hgt_config(preset.stem, create_result_dir=False, **overrides)
dynamic = config.use_dynamic_event_features
config.op_feature_dim = 9
config.job_feature_dim = 8 if dynamic else 7
config.machine_feature_dim = (7 if dynamic else 6) + (not config.use_pt_nodes)
config.pt_feature_dim = 2 if config.use_pt_nodes else 0
print(json.dumps(vars(config), indent=2, default=str))
PY
```

For a new run with that historical trainer, set a new `result_dir` in `overrides`, use `create_result_dir=True`, and replace the final print with `from src.HGT.hgt_trainer import HGTTrainer` followed by `HGTTrainer(config).train()`. Compare the resulting validation fixture with the retained `validation_fixture.json` before interpreting results as a replay. Software/runtime and hardware differences can affect execution; the retained observations remain the evidence for the paper.
