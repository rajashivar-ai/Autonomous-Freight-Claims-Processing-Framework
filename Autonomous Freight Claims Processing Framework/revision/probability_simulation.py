"""
probability_simulation.py
=========================
Standalone, auditable implementation of the 50,000-record probability-proxy
simulation and prediction-confidence diagnostic described in Section 6.10
(Tables 12 & 13) of the manuscript.

IMPORTANT EVIDENTIARY DISCLOSURE:
  - This script generates synthetic probability proxies (P1-P4) from shared +
    proxy-specific Gaussian logits plus a class-conditioned signal followed by
    softmax. It does NOT train or execute B1-B4 extraction/XGBoost/neural
    pipelines, does NOT map placeholder classes (class_0, class_1, class_2) to
    real freight claims, and does NOT validate private-data results or establish
    comparative model superiority.
  - Its purpose is purely to exercise auditable metric calculations:
      * 50,000 claim IDs, stratified 35,000 / 7,500 / 7,500 split
      * Placeholder class proportions: 60% / 25% / 15%
      * Seeds: data=20261031, split=4242, search=9917, proxies=1101..1104,
               bootstrap=20261101 (2,000 multinomial confusion-cell resamples)
      * 20 illustrative validation trials per proxy (reference setting receives
        the strongest signal by construction)
      * Validation macro F1, test macro F1 [95% bootstrap CI], accuracy, natural
        log loss (Table 12 format)
      * 6 pairwise exact two-sided McNemar tests + Holm-Bonferroni correction
        (Table 13 format)
      * Common prediction-confidence diagnostic (all 4 proxies max_prob >= 0.80)
        with exact two-sided Clopper-Pearson binomial CI for conditional accuracy
        and retained claim-ID audit list + SHA-256 hashes.
  - Also supports `--evaluate-csv-dir <dir>` to audit any pre-existing retained
    P1..P4 probability CSV files directly without regenerating them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest


SEEDS = {
    "data": 20261031,
    "split": 4242,
    "search": 9917,
    "proxies": {"P1": 1101, "P2": 1102, "P3": 1103, "P4": 1104},
    "bootstrap": 20261101,
}

CLASS_NAMES = ["class_0", "class_1", "class_2"]
CLASS_PROPORTIONS = [0.60, 0.25, 0.15]

# Reference logit signal/noise parameters (illustrative simulation defaults)
PROXY_SPECS = {
    "P1": {"ref_label": "B1", "signal_ref": 1.32, "noise_sd": 0.96, "shared_weight": 0.55},
    "P2": {"ref_label": "B2", "signal_ref": 1.74, "noise_sd": 0.92, "shared_weight": 0.60},
    "P3": {"ref_label": "B3", "signal_ref": 1.56, "noise_sd": 0.94, "shared_weight": 0.58},
    "P4": {"ref_label": "B4", "signal_ref": 1.84, "noise_sd": 0.90, "shared_weight": 0.62},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def macro_f1_from_cm(cm: np.ndarray) -> float:
    f1s = []
    for c in range(3):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0.0 else 0.0)
    return float(np.mean(f1s))


def confusion_matrix_3x3(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def log_loss_natural(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    p_clip = np.clip(probs, eps, 1.0 - eps)
    p_true = p_clip[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(p_true)))


def multinomial_bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 20261101,
) -> Tuple[float, float]:
    """
    2,000-resample percentile bootstrap CI via multinomial resampling of the
    9 confusion-matrix cell counts (mathematically equivalent to claim-level
    bootstrap resampling for macro F1).
    """
    n = len(y_true)
    rng = np.random.default_rng(seed)
    cells = y_true * 3 + y_pred
    counts = np.bincount(cells, minlength=9)
    draws = rng.multinomial(n, counts / float(n), size=n_resamples).reshape(n_resamples, 3, 3)

    f1_sum = np.zeros(n_resamples, dtype=float)
    for c in range(3):
        tp = draws[:, c, c].astype(float)
        fp = draws[:, :, c].sum(axis=1).astype(float) - tp
        fn = draws[:, c, :].sum(axis=1).astype(float) - tp
        denom = 2.0 * tp + fp + fn
        f1_sum += np.where(denom > 0.0, 2.0 * tp / denom, 0.0)
    lo, hi = np.quantile(f1_sum / 3.0, [0.025, 0.975])
    return float(lo), float(hi)


def holm_bonferroni(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * float(pvals[idx])
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return [float(x) for x in adj]


def generate_probability_files(out_dir: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    rng_data = np.random.default_rng(SEEDS["data"])
    n_total = 50_000
    # Exact 60% / 25% / 15% counts: 30,000 / 12,500 / 7,500
    labels = np.array([0] * 30_000 + [1] * 12_500 + [2] * 7_500, dtype=int)
    rng_data.shuffle(labels)
    claim_ids = np.array([f"SIM{i + 1:05d}" for i in range(n_total)])

    # Stratified random partition: 35,000 train / 7,500 val / 7,500 test
    rng_split = np.random.default_rng(SEEDS["split"])
    split_arr = np.empty(n_total, dtype=object)
    for c in (0, 1, 2):
        idx_c = np.where(labels == c)[0]
        rng_split.shuffle(idx_c)
        n_c = len(idx_c)
        n_tr = int(round(n_c * 0.70))
        n_va = int(round(n_c * 0.15))
        split_arr[idx_c[:n_tr]] = "train"
        split_arr[idx_c[n_tr : n_tr + n_va]] = "val"
        split_arr[idx_c[n_tr + n_va :]] = "test"

    meta_df = pd.DataFrame(
        {
            "claim_id": claim_ids,
            "true_label_id": labels,
            "true_label": [CLASS_NAMES[y] for y in labels],
            "split": split_arr,
        }
    )
    val_df = meta_df[meta_df["split"] == "val"].reset_index(drop=True)
    test_df = meta_df[meta_df["split"] == "test"].reset_index(drop=True)
    assert len(val_df) == 7500 and len(test_df) == 7500

    y_val = val_df["true_label_id"].to_numpy(dtype=int)
    y_test = test_df["true_label_id"].to_numpy(dtype=int)

    # Shared Gaussian latent logits (creates realistic cross-proxy correlation)
    shared_val = rng_data.normal(0.0, 1.0, size=(len(y_val), 3))
    shared_test = rng_data.normal(0.0, 1.0, size=(len(y_test), 3))

    # Class-prior logit offsets reflecting 60/25/15 base rates
    prior_offset = np.log(np.array(CLASS_PROPORTIONS)) * 0.25

    rng_search = np.random.default_rng(SEEDS["search"])
    val_probs_by_proxy: Dict[str, pd.DataFrame] = {}
    test_probs_by_proxy: Dict[str, pd.DataFrame] = {}
    trial_records: List[Dict[str, Any]] = []

    for pname in ("P1", "P2", "P3", "P4"):
        spec = PROXY_SPECS[pname]
        rng_p = np.random.default_rng(SEEDS["proxies"][pname])

        # Base proxy-specific Gaussian logits
        eps_val = rng_p.normal(0.0, spec["noise_sd"], size=(len(y_val), 3))
        eps_test = rng_p.normal(0.0, spec["noise_sd"], size=(len(y_test), 3))
        sw = spec["shared_weight"]
        base_val = sw * shared_val + np.sqrt(max(0.0, 1.0 - sw * sw)) * eps_val + prior_offset
        base_test = sw * shared_test + np.sqrt(max(0.0, 1.0 - sw * sw)) * eps_test + prior_offset

        # 20 illustrative validation trials (trial 1 = reference config with strongest signal by construction)
        signal_factors = [1.0] + list(rng_search.uniform(0.72, 0.96, size=19))
        best_f1 = -1.0
        best_val_probs = None
        for t_idx, fac in enumerate(signal_factors, start=1):
            sig = spec["signal_ref"] * float(fac)
            logits_v = base_val.copy()
            logits_v[np.arange(len(y_val)), y_val] += sig
            probs_v = softmax(logits_v)
            vf1 = macro_f1_from_cm(confusion_matrix_3x3(y_val, np.argmax(probs_v, axis=1)))
            trial_records.append(
                {
                    "proxy": pname,
                    "reference_baseline": spec["ref_label"],
                    "trial": t_idx,
                    "is_reference_setting": int(t_idx == 1),
                    "imposed_signal_strength": round(sig, 6),
                    "val_macro_f1": round(vf1, 6),
                }
            )
            if vf1 > best_f1:
                best_f1 = vf1
                best_val_probs = probs_v

        assert best_val_probs is not None
        logits_t = base_test.copy()
        logits_t[np.arange(len(y_test)), y_test] += spec["signal_ref"]
        probs_t = softmax(logits_t)

        def _pack(df_part: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
            yp = np.argmax(probs, axis=1)
            return pd.DataFrame(
                {
                    "claim_id": df_part["claim_id"].to_numpy(),
                    "true_label_id": df_part["true_label_id"].to_numpy(dtype=int),
                    "true_label": df_part["true_label"].to_numpy(),
                    "p_class_0": np.round(probs[:, 0], 8),
                    "p_class_1": np.round(probs[:, 1], 8),
                    "p_class_2": np.round(probs[:, 2], 8),
                    "pred_label_id": yp,
                    "max_prob": np.round(np.max(probs, axis=1), 8),
                }
            )

        v_df = _pack(val_df, best_val_probs)
        t_df = _pack(test_df, probs_t)
        v_df.to_csv(out_dir / f"{pname}_val_probabilities.csv", index=False)
        t_df.to_csv(out_dir / f"{pname}_test_probabilities.csv", index=False)
        val_probs_by_proxy[pname] = v_df
        test_probs_by_proxy[pname] = t_df

    pd.DataFrame(trial_records).to_csv(out_dir / "proxy_validation_trials_20x4.csv", index=False)
    meta_df.to_csv(out_dir / "simulated_50000_claim_splits.csv", index=False)
    return meta_df, val_probs_by_proxy, test_probs_by_proxy, trial_records


def audit_probability_files(csv_dir: Path, out_dir: Path, conf_threshold: float = 0.80) -> Dict[str, Any]:
    proxies = ["P1", "P2", "P3", "P4"]
    val_dfs: Dict[str, pd.DataFrame] = {}
    test_dfs: Dict[str, pd.DataFrame] = {}
    source_hashes: Dict[str, str] = {}

    for p in proxies:
        vp = csv_dir / f"{p}_val_probabilities.csv"
        tp = csv_dir / f"{p}_test_probabilities.csv"
        if vp.exists():
            val_dfs[p] = pd.read_csv(vp)
            source_hashes[vp.name] = sha256_file(vp)
        test_dfs[p] = pd.read_csv(tp)
        source_hashes[tp.name] = sha256_file(tp)

    ref_ids = test_dfs["P1"]["claim_id"].tolist()
    y_test = test_dfs["P1"]["true_label_id"].to_numpy(dtype=int)
    for p in proxies:
        assert test_dfs[p]["claim_id"].tolist() == ref_ids, f"Claim ID mismatch in {p}"
        assert np.array_equal(test_dfs[p]["true_label_id"].to_numpy(dtype=int), y_test)

    # Table 12 metrics
    table12_rows = []
    preds_by_p: Dict[str, np.ndarray] = {}
    maxp_by_p: Dict[str, np.ndarray] = {}
    for idx, p in enumerate(proxies):
        df_t = test_dfs[p]
        probs_t = df_t[["p_class_0", "p_class_1", "p_class_2"]].to_numpy(dtype=float)
        y_pred = np.argmax(probs_t, axis=1)
        preds_by_p[p] = y_pred
        maxp_by_p[p] = np.max(probs_t, axis=1)

        cm_t = confusion_matrix_3x3(y_test, y_pred)
        te_f1 = macro_f1_from_cm(cm_t)
        lo, hi = multinomial_bootstrap_macro_f1_ci(
            y_test, y_pred, n_resamples=2000, seed=SEEDS["bootstrap"] + idx
        )
        te_acc = float(np.mean(y_pred == y_test) * 100.0)
        te_ll = log_loss_natural(y_test, probs_t)

        val_f1 = None
        if p in val_dfs:
            df_v = val_dfs[p]
            yv = df_v["true_label_id"].to_numpy(dtype=int)
            pv = df_v[["p_class_0", "p_class_1", "p_class_2"]].to_numpy(dtype=float)
            val_f1 = round(macro_f1_from_cm(confusion_matrix_3x3(yv, np.argmax(pv, axis=1))), 6)

        table12_rows.append(
            {
                "proxy": p,
                "reference_baseline": PROXY_SPECS[p]["ref_label"],
                "val_macro_f1": val_f1,
                "test_macro_f1": round(te_f1, 6),
                "test_macro_f1_ci95": [round(lo, 6), round(hi, 6)],
                "test_accuracy_pct": round(te_acc, 2),
                "test_log_loss": round(te_ll, 6),
            }
        )

    # Table 13: 6 pairwise exact two-sided McNemar tests + Holm correction
    pairs = [("P1", "P2"), ("P1", "P3"), ("P1", "P4"), ("P2", "P3"), ("P2", "P4"), ("P3", "P4")]
    table13_rows = []
    for a, b in pairs:
        ca = preds_by_p[a] == y_test
        cb = preds_by_p[b] == y_test
        n_b = int(np.sum(ca & (~cb)))
        n_c = int(np.sum((~ca) & cb))
        n_disc = n_b + n_c
        if n_disc == 0:
            exact_p = 1.0
            chi2_cc = 0.0
        else:
            exact_p = float(binomtest(min(n_b, n_c), n=n_disc, p=0.5, alternative="two-sided").pvalue)
            chi2_cc = float(((max(0.0, abs(n_b - n_c) - 1.0)) ** 2) / n_disc)
        table13_rows.append(
            {
                "pair": f"{a} / {b}",
                "b_A_correct_B_wrong": n_b,
                "c_A_wrong_B_correct": n_c,
                "chi2_cc": round(chi2_cc, 4),
                "exact_p": exact_p,
            }
        )
    adj_p = holm_bonferroni([r["exact_p"] for r in table13_rows])
    for r, ap in zip(table13_rows, adj_p):
        r["holm_adjusted_p"] = ap

    # Prediction-confidence diagnostic: common mask requiring ALL 4 proxies max_prob >= conf_threshold
    common_mask = np.ones(len(y_test), dtype=bool)
    for p in proxies:
        common_mask &= maxp_by_p[p] >= conf_threshold

    n_sub = int(common_mask.sum())
    sub_ids = [ref_ids[i] for i in range(len(ref_ids)) if common_mask[i]]
    pd.DataFrame({"claim_id": sub_ids, "true_label_id": y_test[common_mask]}).to_csv(
        out_dir / "common_subset_claim_ids.csv", index=False
    )

    y_sub = y_test[common_mask]
    class_counts_sub = {CLASS_NAMES[c]: int(np.sum(y_sub == c)) for c in (0, 1, 2)}
    sub_metrics: Dict[str, Any] = {}
    for p in proxies:
        yp_sub = preds_by_p[p][common_mask]
        k_corr = int(np.sum(yp_sub == y_sub))
        f1_sub = macro_f1_from_cm(confusion_matrix_3x3(y_sub, yp_sub)) if n_sub > 0 else None
        acc_sub = (k_corr / float(n_sub)) if n_sub > 0 else None
        if n_sub > 0:
            ci_bin = binomtest(k_corr, n_sub).proportion_ci(confidence_level=0.95, method="exact")
            bin_ci = [round(float(ci_bin.low) * 100.0, 2), round(float(ci_bin.high) * 100.0, 2)]
            b_lo, b_hi = multinomial_bootstrap_macro_f1_ci(y_sub, yp_sub, n_resamples=2000, seed=SEEDS["bootstrap"] + 99)
            boot_ci = [round(b_lo, 4), round(b_hi, 4)]
        else:
            bin_ci, boot_ci = None, None
        sub_metrics[p] = {
            "correct": k_corr,
            "macro_f1": round(f1_sub, 4) if f1_sub is not None else None,
            "bootstrap_macro_f1_ci95": boot_ci,
            "conditional_accuracy_pct": round(acc_sub * 100.0, 2) if acc_sub is not None else None,
            "exact_binomial_accuracy_ci95_pct": bin_ci,
        }

    # Discordant pairs on common subset
    sub_discordant = {}
    for a, b in pairs:
        ca = preds_by_p[a][common_mask] == y_sub
        cb = preds_by_p[b][common_mask] == y_sub
        sub_discordant[f"{a}/{b}"] = {"b": int(np.sum(ca & (~cb))), "c": int(np.sum((~ca) & cb))}

    audit = {
        "disclaimer": (
            "Evaluates synthetic Gaussian-logit probability proxies (P1-P4) only. "
            "Does not implement B1-B4 pipelines or validate private-data empirical claims."
        ),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "seeds": SEEDS,
        "test_cases_n": int(len(y_test)),
        "source_csv_sha256": source_hashes,
        "table12_proxy_scores": table12_rows,
        "table13_paired_mcnemar_holm": table13_rows,
        "prediction_confidence_diagnostic": {
            "threshold": conf_threshold,
            "retained_n": n_sub,
            "retained_pct": round(n_sub / float(len(y_test)) * 100.0, 2),
            "retained_class_counts": class_counts_sub,
            "per_proxy": sub_metrics,
            "discordant_pairs_on_subset": sub_discordant,
        },
    }

    (out_dir / "probability_simulation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame(table12_rows).to_csv(out_dir / "table12_proxy_scores.csv", index=False)
    pd.DataFrame(table13_rows).to_csv(out_dir / "table13_mcnemar_holm.csv", index=False)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="50,000-record probability simulation and audit.")
    parser.add_argument("--evaluate-csv-dir", type=Path, default=None, help="Audit existing P1..P4 CSVs instead of generating new ones.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs_prob_sim")
    parser.add_argument("--conf-threshold", type=float, default=0.80)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_csv_dir is None:
        generate_probability_files(out_dir)
        csv_dir = out_dir
    else:
        csv_dir = args.evaluate_csv_dir

    audit = audit_probability_files(csv_dir, out_dir, conf_threshold=args.conf_threshold)

    print("=== Probability Simulation Scores (Table 12 format, n = %d test cases) ===" % audit["test_cases_n"])
    for r in audit["table12_proxy_scores"]:
        ci = r["test_macro_f1_ci95"]
        print(
            f"  {r['proxy']}/{r['reference_baseline']} | Val F1={r['val_macro_f1']:.3f} | "
            f"Test F1={r['test_macro_f1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"Acc={r['test_accuracy_pct']:.2f}% | LogLoss={r['test_log_loss']:.3f}"
        )
    print("\n=== Paired Exact McNemar + Holm Adjustment (Table 13 format) ===")
    for r in audit["table13_paired_mcnemar_holm"]:
        print(
            f"  {r['pair']:8s} | b={r['b_A_correct_B_wrong']:4d}, c={r['c_A_wrong_B_correct']:4d} | "
            f"exact p={r['exact_p']:.2e} | Holm p={r['holm_adjusted_p']:.2e}"
        )
    diag = audit["prediction_confidence_diagnostic"]
    print(
        f"\n=== Common Confidence Filter (all 4 proxies max_prob >= {diag['threshold']:.2f}) ===\n"
        f"  Retained: {diag['retained_n']} / {audit['test_cases_n']} ({diag['retained_pct']:.2f}%) | "
        f"Class counts: {diag['retained_class_counts']}"
    )


if __name__ == "__main__":
    main()
