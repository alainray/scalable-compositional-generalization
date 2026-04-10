import argparse
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd


FLOAT_PATTERN = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

# Métricas de representación registradas durante el entrenamiento.
REPR_METRICS = [
    "val_4cases_twonn_id",
    "val_4cases_n_components_90pct",
    "val_4cases_topsim",
    "val_4cases_pscore_mean",
    "val_4cases_sv_auc",
    "val_4cases_hoyer_sparsity",
    "val_4cases_embedding_dim",
]


def find_wandb_log_path(path: str) -> str | None:
    """Find output.log in common wandb layouts.

    Supports:
    - <run>/wandb/latest-run/files/output.log
    - <run>/wandb/run-*/files/output.log
    - Any nested output.log under <run> (fallback for custom layouts)
    """
    direct_candidates = [
        os.path.join(path, "wandb", "latest-run", "files", "output.log"),
        os.path.join(path, "output.log"),
        os.path.join(path, "wandb", "output.log"),
    ]
    for candidate in direct_candidates:
        if os.path.exists(candidate):
            return candidate

    wandb_root = os.path.join(path, "wandb")
    search_roots = [wandb_root] if os.path.isdir(wandb_root) else [path]

    for search_root in search_roots:
        found_logs: list[str] = []
        for root, _, files in os.walk(search_root):
            if "output.log" in files:
                found_logs.append(os.path.join(root, "output.log"))
        if found_logs:
            # Prefer the most recently modified output.log.
            return max(found_logs, key=os.path.getmtime)

    return None


def extract_epoch_metrics(log_data: str, metrics: List[str]) -> Dict[int, Dict[str, float]]:
    epoch_data: Dict[int, Dict[str, float]] = {}
    epoch_pattern = re.compile(r"Epoch \[(\d+)\]")
    chunks = epoch_pattern.split(log_data)[1:]

    for i in range(0, len(chunks), 2):
        epoch_num = int(chunks[i].strip())
        epoch_content = chunks[i + 1]
        parsed_metrics: Dict[str, float] = {}
        for metric in metrics:
            parsed = re.search(rf"{re.escape(metric)}:\s*{FLOAT_PATTERN}", epoch_content)
            parsed_metrics[metric] = float(parsed.group(1)) if parsed else np.nan
        epoch_data[epoch_num] = parsed_metrics

    return epoch_data


def parse_run_repr_by_epoch(run_path: str, metrics: List[str]) -> pd.DataFrame:
    log_path = find_wandb_log_path(run_path)
    if log_path is None:
        raise FileNotFoundError(f"No output.log found for run: {run_path}")

    with open(log_path, "r") as file:
        log_data = file.read()

    epoch_data = extract_epoch_metrics(log_data, metrics=metrics)
    rows = []
    for epoch, values in sorted(epoch_data.items()):
        row = {"epoch": epoch}
        row.update(values)
        rows.append(row)

    return pd.DataFrame(rows)


def build_repr_curves_dataframe(path: str, experiment: str, dataset: str, split: str, metrics: List[str]) -> pd.DataFrame:
    base_path = os.path.join(path, experiment, dataset)
    if split:
        base_path = os.path.join(base_path, split)

    if not os.path.isdir(base_path):
        raise FileNotFoundError(
            f"Base path does not exist: {base_path}. "
            "Check --path/--experiment/--dataset/--split."
        )

    c_dirs = [f.path for f in os.scandir(base_path) if f.is_dir()]
    all_rows = []

    for c_path in c_dirs:
        c_name = os.path.basename(c_path)
        composition_match = re.search(r"composition_(.+)$", c_name)
        c_value = composition_match.group(1) if composition_match else c_name

        model_dirs = [f.path for f in os.scandir(c_path) if f.is_dir()]
        for model_path in model_dirs:
            model_name = os.path.basename(model_path).split(".")[0]
            comb_dirs = [f.path for f in os.scandir(model_path) if f.is_dir()]

            for comb_path in comb_dirs:
                comb_name = os.path.basename(comb_path)
                run_dirs = [f.path for f in os.scandir(comb_path) if f.is_dir()]

                for run_path in run_dirs:
                    seed = os.path.basename(run_path)
                    if find_wandb_log_path(run_path) is None:
                        continue
                    try:
                        run_df = parse_run_repr_by_epoch(run_path, metrics=metrics)
                    except Exception as exc:
                        print(f"[WARN] Could not parse {run_path}: {exc}")
                        continue

                    run_df["dataset"] = dataset
                    run_df["experiment"] = experiment
                    run_df["split"] = split if split else ""
                    run_df["c"] = c_value
                    run_df["arch"] = model_name
                    run_df["combination"] = comb_name
                    run_df["seed"] = seed
                    run_df["run_path"] = run_path
                    all_rows.append(run_df)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "experiment",
                "split",
                "c",
                "arch",
                "combination",
                "seed",
                "epoch",
                *metrics,
                "run_path",
            ]
        )

    return pd.concat(all_rows, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parsea output.log de wandb y construye un dataframe con métricas de "
            "representación por época para todos los runs."
        )
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path", type=str, default="out/")
    parser.add_argument(
        "--experiment",
        type=str,
        default="metrics",
        help="Carpeta del experimento dentro de --path (ej.: metrics, orthotopic).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help=(
            "Subdirectorio opcional entre dataset y composición. Déjalo vacío "
            "si tu estructura es out/<experiment>/<dataset>/composition_*/..."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=REPR_METRICS,
        help="Lista de métricas a extraer por época.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta del archivo de salida. Si no se pasa, usa <dataset>_repr_epochs.pkl",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pkl", "csv", "parquet"],
        default="pkl",
        help="Formato de salida del dataframe.",
    )
    return parser.parse_args()


def save_dataframe(df: pd.DataFrame, output: str, fmt: str) -> None:
    if fmt == "pkl":
        df.to_pickle(output)
    elif fmt == "csv":
        df.to_csv(output, index=False)
    elif fmt == "parquet":
        df.to_parquet(output, index=False)


def main():
    args = parse_args()
    df = build_repr_curves_dataframe(
        path=args.path,
        experiment=args.experiment,
        dataset=args.dataset,
        split=args.split,
        metrics=args.metrics,
    )

    output = args.output or f"{args.dataset}_repr_epochs.{args.format}"
    save_dataframe(df, output, args.format)

    print(f"Saved {len(df)} rows to {output}")
    if not df.empty:
        print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
