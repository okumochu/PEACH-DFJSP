#!/usr/bin/env python3
"""Train PEACH using the controls in training/settings.json."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.training_settings import (
    FIELDS,
    INTEGERS,
    BOOLEANS,
    FLOATS,
    build_overrides,
    load_settings,
    save_settings,
)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).with_name("settings.json"),
        help="Training settings JSON (default: training/settings.json)",
    )
    controls = parser.add_argument_group(
        "training controls", "Explicit values override the settings file"
    )
    descriptions = {
        "scale_cl": "Enable or disable scale curriculum learning",
        "initial_size": "First curriculum size, e.g. 20x10",
        "training_size": "Target training and validation size, e.g. 50x10",
        "stopping_rule": "Target readiness then patience, or a fixed update budget",
        "early_stopping_patience": "Consecutive validations without a lower objective",
        "validation_interval": "PPO cycles between validations; each cycle collects a rollout and runs all PPO epochs",
        "maximum_updates": "Total PPO-cycle safety limit; reaching it is not convergence",
        "value_normalization": "Critic normalization, independent of the CL switch",
        "micro_batch_size": "Graphs processed together to limit GPU memory",
        "scale_growth_factor": "Problem-area growth factor between generated stages",
        "stage_minimum_updates": "Minimum updates before the critic can advance a scale stage (at least 20)",
        "stage_maximum_updates": "Abort a non-final stage that cannot advance within this limit",
        "stage_explained_variance": "Critic explained variance required to advance a stage",
        "num_envs": "Parallel training environments per update",
        "seed": "Training random seed",
        "device": "Training device, e.g. cuda:0 or cpu",
    }
    for name in FIELDS:
        options = {
            "default": argparse.SUPPRESS,
            "help": descriptions.get(name, name.replace("_", " ")),
        }
        if name in BOOLEANS:
            options["choices"] = ("on", "off")
        elif name == "mlp_hidden_dims":
            options["type"] = lambda text: [int(x) for x in text.split(",")]
        elif name == "stopping_rule":
            options["choices"] = ("readiness_patience", "fixed_budget")
        elif name == "value_normalization":
            options["choices"] = ("popart", "none")
        else:
            options["type"] = (
                int
                if name in INTEGERS or name == "seed"
                else (
                    float
                    if name in FLOATS
                    else str
                )
            )
        controls.add_argument("--" + name, **options)
    parser.add_argument(
        "--experiment_name", default="peach", help="Result directory name under result/"
    )
    parser.add_argument("--result_dir", type=Path, help="Explicit run directory under result/")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--resume",
        type=Path,
        help="Continue a full snapshot in its existing result directory",
    )
    source.add_argument(
        "--init_from",
        type=Path,
        help="Initialize a new run from policy weights with a fresh optimizer",
    )
    parser.add_argument("--wandb_run_name")
    parser.add_argument("--wandb_project")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print settings and resolved overrides without training",
    )
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    explicit = {key: value for key, value in vars(args).items() if key in FIELDS}
    for key in BOOLEANS & explicit.keys():
        explicit[key] = explicit[key] == "on"
    try:
        settings = load_settings(args.settings, explicit)
        overrides = build_overrides(settings)
        for name, key in (("resume", "resume_path"), ("init_from", "init_from")):
            path = getattr(args, name)
            if path is not None:
                if not path.is_file():
                    raise ValueError(f"--{name} file does not exist: {path}")
                overrides[key] = str(path.resolve())
        if args.result_dir is not None:
            overrides['result_dir'] = str(args.result_dir.resolve())
        if args.resume is not None:
            if args.result_dir is not None and args.result_dir.resolve() != args.resume.resolve().parent:
                raise ValueError('--resume must use its existing result directory')
            overrides['result_dir'] = str(args.resume.resolve().parent)
        for key in ("wandb_run_name", "wandb_project"):
            if getattr(args, key) is not None:
                overrides[key] = getattr(args, key)
        if args.disable_wandb:
            overrides["wandb_enabled"] = False
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.print_config:
        print(
            json.dumps(
                {
                    "training_settings": overrides["training_settings"],
                    "resolved_overrides": overrides,
                },
                indent=2,
            )
        )
        return

    # Import the ML runtime only when a run is actually requested.
    from src.utils.config_util import create_hgt_config
    from src.HGT.hgt_trainer import HGTTrainer

    config = create_hgt_config(
        args.experiment_name, create_result_dir=True, use_pt_nodes=True, **overrides
    )
    record = save_settings(
        config.result_dir,
        overrides["training_settings"],
        getattr(config, "scale_curriculum", None),
    )
    print("Effective training settings:\n" + json.dumps(record, indent=2), flush=True)
    trainer = HGTTrainer(config)
    trainer.train()
    completion = json.loads((Path(config.result_dir) / 'completion.json').read_text())
    if completion['outcome'] not in ('early_stopped', 'fixed_budget_completed'):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
