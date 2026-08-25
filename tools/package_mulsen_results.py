"""Export compact, portable MulSen result provenance from ignored run folders.

Checkpoints and full prediction arrays remain local.  This tool validates their
hashes/configs and writes only compact summaries, portable config snapshots,
diagnostic reports, thermal statistics, and explicitly selected figures.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Mapping, Sequence


_LOSS_PATTERN = re.compile(r"epoch\s+(\d+)\s+mean loss\s+([0-9.eE+-]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _without_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_values(item)
            for key, item in value.items()
            if key not in {
                "values",
                "combined_scores",
                "detection_scores",
                "patch_max_scores_raw",
                "patch_max_scores_normalized",
            }
        }
    if isinstance(value, list):
        return [_without_values(item) for item in value]
    return value


def _portable_config(config: Mapping[str, Any], stage: str) -> Dict[str, Any]:
    portable = deepcopy(dict(config))
    portable["data_root"] = "${MULSEN_DATA_ROOT}"
    thermal = portable.get("thermal_normalization")
    if isinstance(thermal, dict):
        thermal["file"] = f"results/mulsen/{stage}/thermal_stats_{stage}.json"
    return portable


def _losses(path: Path) -> Dict[int, float]:
    losses: Dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LOSS_PATTERN.search(line)
        if match:
            losses[int(match.group(1))] = float(match.group(2))
    if not losses:
        raise ValueError(f"training log has no epoch losses: {path}")
    return losses


def _macro_detection_ap(categories: Mapping[str, Mapping[str, Any]]) -> float:
    values = [
        float(result["image_detection_only"]["average_precision"])
        for result in categories.values()
    ]
    if not values:
        raise ValueError("cannot compute macro detection AP without categories")
    return sum(values) / len(values)


def _selected_evaluation(report: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluations = list(report["evaluations"])
    stage = str(report["protocol_stage"])
    if stage == "development":
        selected = report.get("selected_checkpoint")
        if not isinstance(selected, Mapping):
            raise ValueError("development report has no selected checkpoint")
        epoch = int(selected["epoch"])
        matches = [item for item in evaluations if int(item["epoch"]) == epoch]
        if len(matches) != 1:
            raise ValueError("development selection does not identify one evaluation")
        return matches[0]
    if stage == "final":
        if len(evaluations) != 1 or report.get("selected_checkpoint") is not None:
            raise ValueError("final report must contain one unselected evaluation")
        return evaluations[0]
    raise ValueError(f"unsupported protocol stage: {stage}")


def _run_summary(
    label: str,
    run_dir: Path,
    evaluation_path: Path,
    run_source_commit: str,
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    config_path = run_dir / "experiment_config.json"
    log_path = run_dir / "train.log"
    config = _read_json(config_path)
    report = _read_json(evaluation_path)
    if report["experiment_config"] != config:
        raise ValueError(f"{label}: evaluation config differs from run config")
    selected = _selected_evaluation(report)
    epoch = int(selected["epoch"])
    checkpoint_path = run_dir / f"mulsen_epoch_{epoch:03d}.pth"
    actual_checkpoint_hash = _sha256(checkpoint_path)
    if actual_checkpoint_hash != str(selected["checkpoint_sha256"]).lower():
        raise ValueError(f"{label}: selected checkpoint hash mismatch")
    losses = _losses(log_path)
    macro = dict(selected["metrics"]["macro"])
    macro.setdefault(
        "macro_image_detection_ap",
        _macro_detection_ap(selected["metrics"]["categories"]),
    )
    stage = str(report["protocol_stage"])
    summary = {
        "label": label,
        "protocol_version": report["protocol_version"],
        "protocol_stage": stage,
        "run_source_commit": run_source_commit,
        "run_source_commit_note": (
            "Audited repository state for the run; the checkpoint format did not "
            "embed a Git commit."
        ),
        "seed": int(config["seed"]),
        "variant": config["variant"],
        "train_categories": config["train_categories"],
        "evaluation_categories": config["evaluation_categories"],
        "evaluation_sample_count": int(report["evaluation_sample_count"]),
        "checkpoint_selection": (
            {
                "rule": report["metric_policy"]["selection"],
                "tie_break": report["metric_policy"]["tie_break"],
                "selected_epoch": epoch,
                "selection_score": float(macro["selection_score"]),
            }
            if stage == "development"
            else {
                "rule": "fixed development-selected epoch; no final checkpoint scan",
                "selected_epoch": epoch,
            }
        ),
        "metrics": {
            "macro": macro,
            "pooled_diagnostics": _without_values(
                selected["metrics"]["diagnostics"]
            ),
            "per_category": _without_values(selected["metrics"]["categories"]),
        },
        "training_loss": {
            "selected_epoch": float(losses[epoch]),
            "last_epoch": max(losses),
            "last_epoch_mean": float(losses[max(losses)]),
        },
        "artifacts": {
            "selected_checkpoint": checkpoint_path.name,
            "selected_checkpoint_sha256": actual_checkpoint_hash,
            "selected_checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_binary_tracked": False,
            "source_evaluation": evaluation_path.name,
            "source_evaluation_sha256": _sha256(evaluation_path),
            "source_evaluation_size_bytes": evaluation_path.stat().st_size,
            "source_config": config_path.name,
            "source_config_sha256": _sha256(config_path),
            "source_train_log_sha256": _sha256(log_path),
        },
    }
    portable_config = {
        "path_policy": (
            "Hyperparameters are exact; only data_root and thermal-stat file are "
            "replaced with portable repository placeholders."
        ),
        "source_config_sha256": _sha256(config_path),
        "config": _portable_config(config, stage),
    }
    return summary, portable_config, stage


def _parse_labeled_paths(
    values: Sequence[Sequence[str]], argument: str
) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for label, path in values:
        if label in parsed:
            raise ValueError(f"duplicate {argument} label: {label}")
        parsed[label] = Path(path).expanduser().resolve()
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("LABEL", "RUN_DIR", "EVALUATION_JSON"),
        required=True,
    )
    parser.add_argument(
        "--routing",
        nargs=2,
        action="append",
        default=[],
        metavar=("LABEL", "ROUTING_JSON"),
    )
    parser.add_argument("--thermal-stats", type=Path)
    parser.add_argument("--qualitative-dir", type=Path)
    parser.add_argument("--run-source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite result package: {output_dir}")
    runs = {label: (Path(run_dir).resolve(), Path(evaluation).resolve())
            for label, run_dir, evaluation in args.run}
    if len(runs) != len(args.run):
        raise ValueError("run labels must be unique")
    routing_paths = _parse_labeled_paths(args.routing, "routing")

    stages = set()
    for label, (run_dir, evaluation_path) in runs.items():
        summary, portable_config, stage = _run_summary(
            label, run_dir, evaluation_path, args.run_source_commit
        )
        stages.add(stage)
        _write_json(output_dir / f"{label}_summary.json", summary)
        _write_json(
            output_dir / "configs" / f"{label}_config.json", portable_config
        )
    if len(stages) != 1:
        raise ValueError("one result package cannot mix protocol stages")
    stage = stages.pop()

    for label, path in routing_paths.items():
        report = _read_json(path)
        report["checkpoint"] = Path(report["checkpoint"]).name
        report["source_report_sha256"] = _sha256(path)
        _write_json(output_dir / f"{label}_routing.json", report)

    if args.thermal_stats is not None:
        stats_path = args.thermal_stats.expanduser().resolve()
        shutil.copy2(stats_path, output_dir / f"thermal_stats_{stage}.json")

    if args.qualitative_dir is not None:
        source_dir = args.qualitative_dir.expanduser().resolve()
        report = _read_json(source_dir / "report.json")
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for sample in report["samples"]:
            source = Path(sample["figure"])
            destination = figure_dir / source.name
            shutil.copy2(source, destination)
            samples.append(
                {
                    "sample_key": sample["sample_key"],
                    "category": sample["category"],
                    "anomaly_type": sample["anomaly_type"],
                    "label_rgb": sample["label_rgb"],
                    "label_thermal": sample["label_thermal"],
                    "label_rgbt": sample["label_rgbt"],
                    "selection": "lexicographically first sample in its modality subgroup",
                    "figure": f"figures/{destination.name}",
                    "figure_sha256": _sha256(destination),
                }
            )
        _write_json(
            output_dir / "qualitative_manifest.json",
            {
                "checkpoint_sha256": report["checkpoint_sha256"],
                "protocol_stage": report["protocol_stage"],
                "selection_used_model_performance": False,
                "samples": samples,
            },
        )
    print(json.dumps({"stage": stage, "runs": sorted(runs), "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
