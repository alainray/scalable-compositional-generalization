import argparse
import json
import os
import pandas as pd
import yaml
import re
import numpy as np
from collections import defaultdict

CFG_TO_COL = {
    "model.arch": "arch",
    "model.iso": "iso",
    "training.n_epoch": "n_epoch",
    "seed": "seed",
    "data.training.targets": "combination"
}

METRICS = [
    "train_acc",
    "val_acc",
    "ood_val_0_acc",
    "test_acc",
    "val_4cases_twonn_id",
    "val_4cases_n_components_90pct",
    "val_4cases_topsim",
    "val_4cases_pscore_mean",
    "val_4cases_sv_auc",
    "val_4cases_hoyer_sparsity",
    "val_4cases_embedding_dim",
]

PARSE_LOG = True
CORE_METRICS = ["train_acc", "val_acc", "ood_val_0_acc", "test_acc"]
FLOAT_PATTERN = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def find_wandb_log_path(path):
    latest_run = os.path.join(path, "wandb", "latest-run", "files", "output.log")
    if os.path.exists(latest_run):
        return latest_run
    wandb_root = os.path.join(path, "wandb")
    if not os.path.isdir(wandb_root):
        return None
    run_dirs = sorted(
        [entry.path for entry in os.scandir(wandb_root) if entry.is_dir()]
    )
    for run_dir in reversed(run_dirs):
        candidate = os.path.join(run_dir, "files", "output.log")
        if os.path.exists(candidate):
            return candidate
    return None


def select_best(train_data):
    def wio_metric(id_val, ood_val, test, l):
        return id_val + (ood_val - 100)/l
    def val_metric(id_val, ood_val, test, l):
        return id_val
    def ood_metric(id_val, ood_val, test, l):
        return ood_val
    def oracle_metric(id_val, ood_val, test, l):
        return test
    metrics = {
        "id": val_metric,
        "ood": ood_metric,
        "wio": wio_metric,
        "oracle": oracle_metric
    }
    bests = defaultdict(tuple)
    all = []
    # k: n_epoch, v: metrics
    for v in train_data.values():
        met_vals = []
        for k, m in metrics.items():
            curr = m(v["val_acc"], v["ood_val_0_acc"], v["test_acc"], 10)
            met_vals.append(curr)
            if bests[k] == tuple() or curr >= bests[k][0]:
                bests[k] = (curr, v)
        met_vals.append(v["test_acc"])
        all.append(met_vals)
    bests = {k: v[1] for k,v in bests.items()}
    return bests, np.array(all)



def parse_training_metrics(path):
    log_path = find_wandb_log_path(path)
    if log_path is None:
        raise FileNotFoundError(
            f"No output.log found under {os.path.join(path, 'wandb')}"
        )
    with open(log_path, "r") as file:
        log_data = file.read()
    epoch_data = extract_epoch_metrics(log_data, extra_metrics=METRICS)
    best_epoch_results, curves = select_best(epoch_data)
    for metric in best_epoch_results.keys():
        with open(os.path.join(path, f"results_{metric}.json"), "w") as json_file:
            json.dump(dict(best_epoch_results[metric]), json_file, indent=4)
    return curves

def _safe_nested_get(data, dotted_key):
    if not isinstance(data, dict):
        return np.nan
    parsed_att = data
    for key in dotted_key.split("."):
        if not isinstance(parsed_att, dict):
            return np.nan
        parsed_att = parsed_att.get(key, np.nan)
    return parsed_att


def process_experiment(path):
    """
    Read experiment files and extract results
    """
    extracted = dict()
    cfg_file_path = os.path.join(path, "cfg.yml")
    cfg = {}
    if os.path.exists(cfg_file_path):
        with open(cfg_file_path, 'r') as file:
            cfg = yaml.safe_load(file) or {}

    for eval in ["id", "ood", "wio", "oracle"]:
        # read files
        res_file_path = os.path.join(path, f"results_{eval}.json")
        with open(res_file_path, 'r') as file:
            metrics = json.load(file)

        tmp = dict()
        for k, v in CFG_TO_COL.items():
            tmp[v] = _safe_nested_get(cfg, k)
        for m in METRICS:
            tmp[m] = metrics.get(m, np.nan)
        extracted[eval] = tmp
    return extracted

def select_best_id(train_data):
    best_epoch = None
    for v in train_data.values():
        if best_epoch is None or v["val_acc"] >= best_epoch["val_acc"]:
            best_epoch = dict(v)
    return best_epoch or dict()


def extract_epoch_metrics(log_data, extra_metrics=None):
    metrics_to_parse = list(dict.fromkeys(CORE_METRICS + (extra_metrics or [])))
    epoch_data = {}
    epoch_pattern = re.compile(r"Epoch \[(\d+)\]")
    epochs = epoch_pattern.split(log_data)
    epochs = epochs[1:]
    for i in range(0, len(epochs), 2):
        epoch_num = int(epochs[i].strip())
        epoch_content = epochs[i + 1]
        parsed_metrics = {}
        for metric in metrics_to_parse:
            parsed = re.search(rf"{re.escape(metric)}:\s*{FLOAT_PATTERN}", epoch_content)
            parsed_metrics[metric] = float(parsed.group(1)) if parsed else 0
        epoch_data[epoch_num] = parsed_metrics
    return epoch_data

def parse_id(path):
    log_path = find_wandb_log_path(path)
    if log_path is None:
        raise FileNotFoundError(
            f"No output.log found under {os.path.join(path, 'wandb')}"
        )
    with open(log_path, "r") as file:
        log_data = file.read()
    epoch_data = extract_epoch_metrics(log_data, extra_metrics=METRICS)
    best_epoch_result = select_best_id(epoch_data)
    return best_epoch_result




def elaborate_results(datasets, cfg_to_col, metrics, path, experiment, split, selection):
    df = pd.DataFrame(columns=list(cfg_to_col.values())+metrics)
    for data in datasets:
        base_path = os.path.join(path, experiment, data, split)
        models = [ f.path for f in os.scandir(base_path) if f.is_dir() ]
        curves = []
        for model_path in models:
            model_name = os.path.basename(model_path).split(".")[0]
            if model_name in ["resnet18_leaky", "resnet50_leaky", "ed_prelu", "densenet121_old"]: continue
            try:
                combinations = [ f.path for f in os.scandir(model_path) if f.is_dir() ]
            except:
                combinations = []
            for c in combinations:
                runs = [ f.path for f in os.scandir(c) if f.is_dir() ]
                for r in runs:
                    try:
                        # parse training logs
                        curves.append(parse_training_metrics(r))
                        # process experiment and log the results in the dataframe
                        res = process_experiment(r)[selection]
                        res["arch"] = model_name
                        res["dataset"] = data
                        # append results
                        df = df._append(res, ignore_index = True) if not df.empty else pd.DataFrame([res])
                    except Exception as e:
                        print(r)
                        print(e)
    return df, curves




def parse_args():
    """Parse CLI arguments.

    Returns:
        (argparse.Namespace, list): returns known and unknown parsed args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cars3d")
    parser.add_argument("--path", type=str, default="out/")
    parser.add_argument("--experiment", type=str, default="orthotopic")
    parser.add_argument("--split", type=str, default="composition_0.1")
    parser.add_argument(
        "--selection",
        type=str,
        default="id",
        choices=["id", "ood", "wio", "oracle"],
        help="Model selection criterion used to build the output dataframe",
    )
    return parser.parse_known_args()


def read_training_out(datasets, cfg_to_col, metrics, path, experiment, split):
    df = pd.DataFrame(columns=list(cfg_to_col.values())+metrics)
    for data in datasets:
        base_path = os.path.join(path, experiment, data, split)
        models = [ f.path for f in os.scandir(base_path) if f.is_dir() ]
        curves = []
        for model_path in models:
            model_name = os.path.basename(model_path).split(".")[0]
            if model_name in ["resnet18_leaky", "resnet50_leaky", "ed_prelu", "densenet121_old"]: continue
            try:
                combinations = [ f.path for f in os.scandir(model_path) if f.is_dir() ]
            except:
                combinations = []
            for c in combinations:
                runs = [ f.path for f in os.scandir(c) if f.is_dir() ]
                for r in runs:
                    try:
                        with open(os.path.join(r, "checkpoints", "results.json")) as f:
                            res = dict(json.load(f))
                        res["arch"] = model_name
                        res["dataset"] = data
                        # append results
                        df = df._append(res, ignore_index = True) if not df.empty else pd.DataFrame([res])
                    except Exception as e:
                        print(r)
                        print(e)   
    return df, None


ARCHS = [
    'convnext_base',
    'convnext_small',
    'convnext_tiny',
    'densenet121',
    'densenet121_pretrained',
    'densenet161',
    'densenet201',
    'ed',
    'mlp',
    'resnet101',
    'resnet101_pretrained',
    'resnet152',
    'resnet152_pretrained',
    'resnet18',
    'resnet50',
    'swin_base',
    'swin_tiny',
    'wideresnet',
]

def check_arch_counts(df, max_c):
    print("\nArchitecture check")
    print("=" * 60)
    if df.empty or not {"arch", "c", "seed"}.issubset(df.columns):
        print("❌ No parsed runs found; skipping architecture count checks.")
        print("=" * 60)
        return
    arch_counts = df['arch'].value_counts()
    all_ok = True
    exp_runs = 3 * (max_c+1)
    for arch in ARCHS:
        count = arch_counts.get(arch, 0)
        status = "OK" if count == exp_runs else f"NO ({count}, expected {exp_runs})"
        dots = '.' * (50 - len(arch))
        print(f"{arch}{dots}{status}")
        if count != exp_runs:
            all_ok = False
    print("=" * 60)
    if all_ok:
        print("✅ All architectures are fine.")
    else:
        print("❌ Some architectures have incorrect counts.")

    expected_combinations = set(
        (arch, c, seed)
        for arch in ARCHS
        for c in range(max_c + 1)
        for seed in [1, 2, 3]
    )
    present_combinations = set(df[['arch', 'c', 'seed']].itertuples(index=False, name=None))
    present_combinations = set([(t[0], int(t[1]), int(t[2])) for t in present_combinations])

    missing = expected_combinations - present_combinations
    if missing:
        print(f"❌ Missing {len(missing)} (arch, c, seed) combinations:\n")
        print(" ".join(f'"{arch} {c} {seed}"' for arch, c, seed in sorted(missing)))

def main():
    args, uknw = parse_args()
    selection_to_fn = {
        "id": parse_id,
    }
    parse_result_fn = selection_to_fn.get(args.selection)
    df = pd.DataFrame(columns=list(CFG_TO_COL.values())+METRICS)
    base_path = os.path.join(args.path, args.experiment, args.dataset)
    c_list = [ f.path for f in os.scandir(base_path) if f.is_dir() ]
    print("Loading data...")
    parsed_int_cs = []
    for c_path in c_list:
        c = os.path.basename(c_path).split("_")[-1]
        parsed_int_cs.append(int(c))
        models = [ f.path for f in os.scandir(c_path) if f.is_dir() ]
        for model_path in models:
            model_name = os.path.basename(model_path).split(".")[0]
            try:
                combinations = [ f.path for f in os.scandir(model_path) if f.is_dir() ]
            except:
                combinations = []
            for comb in combinations:
                runs = [ f.path for f in os.scandir(comb) if f.is_dir() ]
                for r in runs:
                    try:
                        id = os.path.basename(r).split("/")[-1]
                        # parse training logs
                        if PARSE_LOG:
                            parse_training_metrics(r)
                        # process experiment and log the results in the dataframe
                        if parse_result_fn is not None:
                            res = parse_result_fn(r)
                        else:
                            res = process_experiment(r)[args.selection]
                        res["arch"] = model_name
                        res["c"] = c
                        res["seed"] = id
                        # append results
                        df = df._append(res, ignore_index = True) if not df.empty else pd.DataFrame([res])
                    except Exception as e:
                        print(r)
                        print(e)    
    print("Data loaded.")
    check_arch_counts(df, max(parsed_int_cs))
    df.to_pickle(f"{args.dataset}_{args.selection}.pkl")



if __name__ == "__main__":
    main()
