import argparse
import math
import os
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


def mean_absolute_error(values: pd.DataFrame) -> float:
    return float((values["true_score"] - values["predicted_score"]).abs().mean())


def bootstrap_ci(errors: np.ndarray, iterations: int = 1000, confidence: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    if len(errors) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    bootstrap_means = []
    for _ in range(iterations):
        sample = rng.choice(errors, size=len(errors), replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    alpha = 1.0 - confidence
    lower = float(np.quantile(bootstrap_means, alpha / 2.0))
    upper = float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0))
    return lower, upper


def paired_t_test(left_errors: np.ndarray, right_errors: np.ndarray) -> Dict[str, float]:
    if len(left_errors) != len(right_errors) or len(left_errors) < 2:
        return {"t_statistic": float("nan"), "p_value": float("nan")}

    differences = left_errors - right_errors
    mean_difference = float(np.mean(differences))
    std_difference = float(np.std(differences, ddof=1))
    if std_difference == 0:
        return {"t_statistic": 0.0, "p_value": 1.0}

    t_statistic = mean_difference / (std_difference / math.sqrt(len(differences)))
    if stats is not None:
        p_value = float(stats.t.sf(abs(t_statistic), df=len(differences) - 1) * 2)
    else:
        p_value = float(math.erfc(abs(t_statistic) / math.sqrt(2)))

    return {"t_statistic": float(t_statistic), "p_value": p_value}


def compute_process_metrics(group: pd.DataFrame) -> Dict[str, float]:
    valid = group.dropna(subset=["predicted_score"])
    if len(valid) == 0:
        return {
            "audit_correction_rate": float("nan"),
            "hallucination_rate": float("nan"),
            "knowledge_grounding_consistency": float("nan"),
            "interpretability_quality": float("nan"),
            "reasoning_stability": float("nan"),
        }

    audit_correction_rate = float(valid["audit_corrected"].astype(bool).mean())
    hallucination_rate = float(valid["hallucination_flag"].astype(bool).mean())

    has_knowledge = valid["knowledge_reference_ids"].fillna("").astype(str).str.len() > 0
    has_rationale = valid["rationale"].fillna("").astype(str).str.len() > 20
    knowledge_grounding = float((has_knowledge & has_rationale).mean())

    prediction_variance = valid.groupby("participant_id")["predicted_score"].std().fillna(0)
    reasoning_stability = float(1.0 / (1.0 + prediction_variance.mean()))

    evidence_terms = [
        "evidence",
        "transcript",
        "sleep",
        "fatigue",
        "mood",
        "interest",
        "appetite",
        "concentration",
        "self-esteem",
        "psychomotor",
    ]
    rationale = valid["rationale"].fillna("").astype(str).str.lower()
    interpretability = float(
        rationale.apply(lambda text: any(term in text for term in evidence_terms) and len(text) > 30).mean()
    )

    return {
        "audit_correction_rate": audit_correction_rate,
        "hallucination_rate": hallucination_rate,
        "knowledge_grounding_consistency": knowledge_grounding,
        "interpretability_quality": interpretability,
        "reasoning_stability": reasoning_stability,
    }


def summarize_results(results: pd.DataFrame, bootstrap_iterations: int) -> pd.DataFrame:
    summaries = []

    for configuration, group in results.groupby("configuration"):
        valid = group.dropna(subset=["predicted_score"]).copy()
        seed_mae = valid.groupby("seed")[["true_score", "predicted_score"]].apply(mean_absolute_error)
        errors = (valid["true_score"] - valid["predicted_score"]).abs().to_numpy(dtype=float)
        ci_lower, ci_upper = bootstrap_ci(errors, iterations=bootstrap_iterations)
        process_metrics = compute_process_metrics(valid)

        summaries.append(
            {
                "configuration": configuration,
                "cases": int(len(valid)),
                "seeds": int(valid["seed"].nunique()),
                "mae_mean": float(seed_mae.mean()) if len(seed_mae) else float("nan"),
                "mae_std": float(seed_mae.std(ddof=1)) if len(seed_mae) > 1 else 0.0,
                "mae_bootstrap_ci_lower": ci_lower,
                "mae_bootstrap_ci_upper": ci_upper,
                **process_metrics,
            }
        )

    return pd.DataFrame(summaries).sort_values("configuration")


def pairwise_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = results.dropna(subset=["predicted_score"]).copy()
    valid["absolute_error"] = (valid["true_score"] - valid["predicted_score"]).abs()

    configurations = sorted(valid["configuration"].unique())
    for left, right in combinations(configurations, 2):
        left_df = valid[valid["configuration"] == left]
        right_df = valid[valid["configuration"] == right]
        merged = left_df.merge(
            right_df,
            on=["participant_id", "seed"],
            suffixes=("_left", "_right"),
        )

        test_result = paired_t_test(
            merged["absolute_error_left"].to_numpy(dtype=float),
            merged["absolute_error_right"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "left_configuration": left,
                "right_configuration": right,
                "paired_cases": int(len(merged)),
                "left_mae": float(merged["absolute_error_left"].mean()) if len(merged) else float("nan"),
                "right_mae": float(merged["absolute_error_right"].mean()) if len(merged) else float("nan"),
                **test_result,
            }
        )

    return pd.DataFrame(rows)


def load_result_files(input_paths: List[str]) -> pd.DataFrame:
    frames = []
    for path in input_paths:
        if os.path.isdir(path):
            csv_path = os.path.join(path, "case_results.csv")
            if os.path.exists(csv_path):
                frames.append(pd.read_csv(csv_path))
        else:
            frames.append(pd.read_csv(path))

    if not frames:
        raise ValueError("No case result files were found.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PHQ-8 experiment results.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Result directories or case_results.csv files.")
    parser.add_argument("--output-dir", default="results_summary", help="Directory for summary CSV files.")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_result_files(args.inputs)
    summary = summarize_results(results, args.bootstrap_iterations)
    comparisons = pairwise_comparisons(results)

    summary_path = os.path.join(args.output_dir, "metric_summary.csv")
    comparisons_path = os.path.join(args.output_dir, "pairwise_t_tests.csv")
    summary.to_csv(summary_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)

    print(f"Saved metric summary to {summary_path}")
    print(f"Saved pairwise t-tests to {comparisons_path}")


if __name__ == "__main__":
    main()
