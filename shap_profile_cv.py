from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold


def load_table(path: Path, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    table = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    if target not in table or not table.columns.is_unique:
        raise ValueError("The target column must exist and all column names must be unique")
    features = table.drop(columns=target)
    if len(features) < 10 or features.shape[1] < 8:
        raise ValueError("At least 10 samples and 8 features are required")
    features = features.apply(pd.to_numeric, errors="raise")
    response = pd.to_numeric(table[target], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(response).all():
        raise ValueError("The target contains missing or non-finite values")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, response


def prepare_training(features: pd.DataFrame) -> pd.DataFrame:
    medians = features.median(axis=0).fillna(0.0)
    return features.fillna(medians).astype(float)


def fit_shap(features: pd.DataFrame, response: np.ndarray, seed: int) -> np.ndarray:
    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        iterations=150,
        loss_function="RMSE",
        random_seed=seed,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(features, response)
    values = np.asarray(shap.TreeExplainer(model).shap_values(features), dtype=float)
    if values.shape != features.shape:
        raise ValueError("Unexpected SHAP matrix shape")
    return values


def rank_features(importance: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(len(importance)), -importance))


def original_shap(features: pd.DataFrame, response: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = fit_shap(features, response, seed)
    importance = np.abs(values).mean(axis=0)
    return importance, rank_features(importance)


def improved_shap(
    features: pd.DataFrame,
    response: np.ndarray,
    seed: int,
    bootstrap_count: int = 10,
    candidate_ratio: float = 0.30,
    selected_count: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bootstrap_count < 1 or not 0 < candidate_ratio <= 1:
        raise ValueError("Invalid bootstrap count or candidate ratio")
    if not 1 <= selected_count <= features.shape[1]:
        raise ValueError("Invalid selected feature count")
    generator = np.random.default_rng(seed)
    bootstrap_importance = []
    for bootstrap in range(bootstrap_count):
        sampled = generator.integers(0, len(features), len(features))
        model = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            iterations=150,
            loss_function="RMSE",
            random_seed=seed + bootstrap,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(features.iloc[sampled], response[sampled])
        values = np.asarray(shap.TreeExplainer(model).shap_values(features), dtype=float)
        if values.shape != features.shape:
            raise ValueError("Unexpected SHAP matrix shape")
        bootstrap_importance.append(np.abs(values).mean(axis=0))
    importance_by_bootstrap = np.stack(bootstrap_importance)
    importance = importance_by_bootstrap.mean(axis=0)
    order = rank_features(importance)
    candidate_count = max(selected_count, int(np.floor(candidate_ratio * features.shape[1])))
    candidates = order[:candidate_count]
    profiles = importance[candidates].astype(float).reshape(-1, 1)
    spread = profiles.std(axis=0)
    profiles = np.divide(
        profiles - profiles.mean(axis=0),
        spread,
        out=np.zeros_like(profiles),
        where=spread > 0,
    )
    distinct = np.unique(profiles, axis=0).shape[0]
    cluster_count = min(selected_count, distinct)
    labels = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit_predict(profiles)
    representatives = []
    for cluster in range(cluster_count):
        members = candidates[labels == cluster]
        representatives.append(members[rank_features(importance[members])[0]])
    selected = np.asarray(representatives, dtype=int)
    if len(selected) < selected_count:
        remaining = [index for index in order if index not in selected]
        selected = np.concatenate([selected, remaining[:selected_count - len(selected)]])
    selected = selected[rank_features(importance[selected])]
    top_rate = np.zeros(features.shape[1], dtype=float)
    for row in importance_by_bootstrap:
        top_rate[rank_features(row)[:candidate_count]] += 1
    top_rate /= bootstrap_count
    cluster_by_feature = np.full(features.shape[1], -1, dtype=int)
    cluster_by_feature[candidates] = labels
    return importance, order, selected, top_rate, cluster_by_feature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="BIOMASS")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    features, response = load_table(args.input, args.target)
    args.output.mkdir(parents=True, exist_ok=True)
    ranking_rows = []
    selected_rows = []
    splitter = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(splitter.split(features), start=1):
        training = prepare_training(features.iloc[train])
        target = response[train]
        original_importance, original_order = original_shap(training, target, args.seed + fold)
        improved_importance, improved_order, improved_selected, top_rate, clusters = improved_shap(
            training, target, args.seed + 1000 * fold
        )
        original_selected = original_order[:min(34, features.shape[1])]
        for method, importance, order in (
            ("Original SHAP", original_importance, original_order),
            ("Improved SHAP", improved_importance, improved_order),
        ):
            for rank, index in enumerate(order, start=1):
                ranking_rows.append({
                    "fold": fold,
                    "method": method,
                    "rank": rank,
                    "feature": features.columns[index],
                    "mean_abs_shap": importance[index],
                    "bootstrap_top_rate": top_rate[index] if method == "Improved SHAP" else np.nan,
                    "profile_cluster": clusters[index] if method == "Improved SHAP" else np.nan,
                })
        for method, indices in (("Original SHAP", original_selected), ("Improved SHAP", improved_selected)):
            for rank, index in enumerate(indices, start=1):
                selected_rows.append({
                    "fold": fold,
                    "method": method,
                    "selected_rank": rank,
                    "feature": features.columns[index],
                    "train_n": len(train),
                    "test_n": len(test),
                })
        pd.DataFrame(ranking_rows).to_csv(args.output / "fold_shap_rankings.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(args.output / "fold_selected_features.csv", index=False)
        print(f"Fold {fold}/10 completed", flush=True)
    selected_table = pd.DataFrame(selected_rows)
    frequency = selected_table.groupby(["method", "feature"]).size().rename("selected_folds").reset_index()
    frequency["selection_rate"] = frequency["selected_folds"] / 10
    frequency.sort_values(["method", "selected_folds", "feature"], ascending=[True, False, True]).to_csv(
        args.output / "selection_frequency.csv", index=False
    )
    rankings = pd.DataFrame(ranking_rows)
    consensus = rankings.groupby(["method", "feature"], as_index=False).agg(
        mean_rank=("rank", "mean"),
        mean_abs_shap=("mean_abs_shap", "mean"),
        mean_bootstrap_top_rate=("bootstrap_top_rate", "mean"),
    )
    consensus = consensus.merge(frequency, on=["method", "feature"], how="left")
    consensus["selected_folds"] = consensus["selected_folds"].fillna(0).astype(int)
    consensus["selection_rate"] = consensus["selection_rate"].fillna(0.0)
    consensus = consensus.sort_values(
        ["method", "mean_rank", "mean_abs_shap", "feature"],
        ascending=[True, True, False, True],
    )
    consensus["consensus_rank"] = consensus.groupby("method").cumcount() + 1
    consensus.to_csv(args.output / "consensus_rankings.csv", index=False)
    summary = {
        "samples": len(features),
        "features": features.shape[1],
        "folds": 10,
        "split_seed": args.seed,
        "selection_uses_training_rows_only": True,
        "original_bootstrap_count": 0,
        "original_selected_count": min(34, features.shape[1]),
        "improved_bootstrap_count_per_fold": 10,
        "improved_candidate_ratio": 0.30,
        "improved_selected_count": 8,
        "profile_basis": "standardized mean absolute SHAP importance of candidate features (34 x 1)",
        "cluster_method": "KMeans on one-dimensional standardized mean |SHAP| values",
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
