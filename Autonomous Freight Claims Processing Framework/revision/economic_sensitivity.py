"""
economic_sensitivity.py
=======================
Standalone, self-contained script reproducing the disclosed economic model
(Table 10 / Table 11) and the 200,000-draw Monte Carlo sensitivity scenario
(Section 6.7 / Fig. 6 of the manuscript).

Exact Specification Reproduced:
  - Annual claim volume N = 50,000 (fixed)
  - Implementation cost amortised straight-line over 3 years
  - 200,000 independent triangular draws, seed = 20261029 (NumPy PCG64)
  - Triangular (min, mode, max) bounds:
      * Manual unit cost C_m:        ($38.00,  $45.00,  $52.00)
      * Automated unit cost C_a:     ($9.50,   $12.00,  $15.50)
      * Recovery-rate uplift:        (3.00%,   5.85%,   8.20%)
      * Mean claim value:            ($340,    $410,    $470)
      * Recurring annual cost:       ($0.380M, $0.493M, $0.760M)
      * One-time implementation:     ($1.80M,  $2.40M,  $3.60M)
  - Verified Outputs (NumPy 2.3.5, PCG64):
      * P10    = $969,158  ($0.97M)
      * Median = $1,380,023 ($1.38M)
      * P90    = $1,786,353 ($1.79M)
      * Adverse support corner = -$325,000 annual loss

Usage:
    python economic_sensitivity.py
    python economic_sensitivity.py --out-dir ./outputs_economic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Disclosed Table 10 / 11 nominal inputs
NOMINAL = {
    "annual_claims_N": 50000,
    "manual_handling_hours": 0.62,
    "fully_loaded_labour_rate_usd_hr": 58.00,
    "manual_other_cost_usd": 9.04,
    "compute_inference_cost_usd": 1.90,
    "rpa_runtime_cost_usd": 0.76,
    "escalation_unit_labour_usd": 62.27,
    "nominal_stp_rate": 0.85,
    "mean_recoverable_claim_usd": 410.0,
    "recovery_rate_before": 0.614,
    "recovery_rate_after": 0.6725,
    "implementation_cost_usd": 2_400_000.0,
    "amortisation_years": 3,
    "licensing_annual_usd": 265_000.0,
    "infrastructure_annual_usd": 88_000.0,
    "maintenance_annual_usd": 140_000.0,
    "discount_rate": 0.10,
}

# Exact Monte Carlo triangular parameters (Section 6.7)
MC_SPEC = {
    "n_draws": 200_000,
    "seed": 20261029,
    "manual_unit_cost_usd": (38.0, 45.0, 52.0),
    "automated_unit_cost_usd": (9.5, 12.0, 15.5),
    "recovery_uplift_frac": (0.030, 0.0585, 0.082),
    "mean_claim_value_usd": (340.0, 410.0, 470.0),
    "recurring_annual_cost_usd": (380_000.0, 493_000.0, 760_000.0),
    "implementation_cost_usd": (1_800_000.0, 2_400_000.0, 3_600_000.0),
}


def compute_economic_model() -> Tuple_Result:
    N = NOMINAL["annual_claims_N"]
    manual_labour = NOMINAL["manual_handling_hours"] * NOMINAL["fully_loaded_labour_rate_usd_hr"]
    c_m = manual_labour + NOMINAL["manual_other_cost_usd"]
    esc_labour = (1.0 - NOMINAL["nominal_stp_rate"]) * NOMINAL["escalation_unit_labour_usd"]
    c_a = NOMINAL["compute_inference_cost_usd"] + NOMINAL["rpa_runtime_cost_usd"] + esc_labour

    cost_saving = (c_m - c_a) * N
    rec_uplift = NOMINAL["recovery_rate_after"] - NOMINAL["recovery_rate_before"]
    recovery_gain = N * NOMINAL["mean_recoverable_claim_usd"] * rec_uplift
    gross_annual = cost_saving + recovery_gain

    impl = NOMINAL["implementation_cost_usd"]
    amort = impl / NOMINAL["amortisation_years"]
    recurring = (
        NOMINAL["licensing_annual_usd"]
        + NOMINAL["infrastructure_annual_usd"]
        + NOMINAL["maintenance_annual_usd"]
    )
    net_annual = gross_annual - recurring - amort
    cash_annual = gross_annual - recurring
    payback_months = (impl / cash_annual) * 12.0

    r = NOMINAL["discount_rate"]
    pv_factor = sum(1.0 / ((1.0 + r) ** t) for t in range(1, NOMINAL["amortisation_years"] + 1))
    npv_3yr = -impl + cash_annual * pv_factor

    # STP sensitivity schedule (90% down to 70%)
    stp_rows = []
    for stp in (0.90, 0.85, 0.80, 0.75, 0.70):
        ca_s = (
            NOMINAL["compute_inference_cost_usd"]
            + NOMINAL["rpa_runtime_cost_usd"]
            + (1.0 - stp) * NOMINAL["escalation_unit_labour_usd"]
        )
        cs_s = (c_m - ca_s) * N
        gross_s = cs_s + recovery_gain
        net_s = gross_s - recurring - amort
        stp_rows.append(
            {
                "stp_pct": round(stp * 100.0, 1),
                "automated_unit_cost_usd": round(ca_s, 2),
                "annual_cost_saving_usd": round(cs_s, 2),
                "gross_annual_benefit_usd": round(gross_s, 2),
                "net_annual_benefit_usd": round(net_s, 2),
            }
        )

    # 200,000-draw Monte Carlo triangular simulation (exact draw order & seed)
    rng = np.random.default_rng(MC_SPEC["seed"])
    n_draws = MC_SPEC["n_draws"]
    d_cm = rng.triangular(*MC_SPEC["manual_unit_cost_usd"], size=n_draws)
    d_ca = rng.triangular(*MC_SPEC["automated_unit_cost_usd"], size=n_draws)
    d_up = rng.triangular(*MC_SPEC["recovery_uplift_frac"], size=n_draws)
    d_val = rng.triangular(*MC_SPEC["mean_claim_value_usd"], size=n_draws)
    d_rec = rng.triangular(*MC_SPEC["recurring_annual_cost_usd"], size=n_draws)
    d_imp = rng.triangular(*MC_SPEC["implementation_cost_usd"], size=n_draws)

    d_gross = (d_cm - d_ca) * N + N * d_val * d_up
    d_net = d_gross - d_rec - (d_imp / NOMINAL["amortisation_years"])

    q05, q10, q25, q50, q75, q90, q95 = np.quantile(d_net, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

    # Analytical adverse support corner
    adverse_gross = (
        (MC_SPEC["manual_unit_cost_usd"][0] - MC_SPEC["automated_unit_cost_usd"][2]) * N
        + N * MC_SPEC["mean_claim_value_usd"][0] * MC_SPEC["recovery_uplift_frac"][0]
    )
    adverse_net = (
        adverse_gross
        - MC_SPEC["recurring_annual_cost_usd"][2]
        - (MC_SPEC["implementation_cost_usd"][2] / NOMINAL["amortisation_years"])
    )

    summary = {
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "bit_generator": "PCG64",
        },
        "deterministic_table": {
            "manual_labour_cost_usd": round(manual_labour, 2),
            "manual_unit_cost_usd": round(c_m, 2),
            "escalation_labour_usd": round(esc_labour, 4),
            "automated_unit_cost_usd": round(c_a, 4),
            "cost_saving_annual_usd": round(cost_saving, 2),
            "recovery_gain_annual_usd": round(recovery_gain, 2),
            "gross_annual_benefit_usd": round(gross_annual, 2),
            "recurring_annual_cost_usd": round(recurring, 2),
            "amortisation_annual_usd": round(amort, 2),
            "net_annual_benefit_usd": round(net_annual, 2),
            "annual_cash_benefit_usd": round(cash_annual, 2),
            "payback_months": round(payback_months, 2),
            "npv_3yr_10pct_usd": round(npv_3yr, 2),
        },
        "stp_sensitivity": stp_rows,
        "monte_carlo": {
            "n_draws": n_draws,
            "seed": MC_SPEC["seed"],
            "mean_usd": round(float(np.mean(d_net)), 2),
            "sd_usd": round(float(np.std(d_net, ddof=1)), 2),
            "min_draw_usd": round(float(np.min(d_net)), 2),
            "max_draw_usd": round(float(np.max(d_net)), 2),
            "p05_usd": round(float(q05), 2),
            "p10_usd": round(float(q10), 2),
            "p25_usd": round(float(q25), 2),
            "p50_median_usd": round(float(q50), 2),
            "p75_usd": round(float(q75), 2),
            "p90_usd": round(float(q90), 2),
            "p95_usd": round(float(q95), 2),
            "pct_draws_positive": round(float(np.mean(d_net > 0.0) * 100.0), 4),
            "adverse_support_corner_usd": round(float(adverse_net), 2),
        },
    }
    return summary, d_net


Tuple_Result = Any


def plot_fig6(d_net: np.ndarray, mc: Dict[str, Any], out_path: Path) -> None:
    plt.rcParams.update(
        {"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200}
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    counts, edges = np.histogram(d_net / 1e6, bins=50, range=(0.0, 3.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.90, color="#5c7d9c", edgecolor="white", linewidth=0.4)

    p10 = mc["p10_usd"] / 1e6
    p50 = mc["p50_median_usd"] / 1e6
    p90 = mc["p90_usd"] / 1e6
    ymax = float(np.max(counts)) * 1.15
    for val, lbl, col in ((p10, "P10", "#333333"), (p50, "P50", "#8b1e1e"), (p90, "P90", "#333333")):
        ax.axvline(val, color=col, linestyle="--", linewidth=1.3, ymax=0.88)
        ax.text(val, ymax * 0.91, f"{lbl}\n${val:.2f}M", ha="center", va="bottom", color=col, fontsize=8.5, fontweight="bold")

    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, ymax * 1.08)
    ax.set_xlabel("Net annual benefit after ownership costs (USD millions)")
    ax.set_ylabel("Draws")
    ax.set_title(f"Monte Carlo Sensitivity ({mc['n_draws']:,} triangular draws, seed {mc['seed']})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Economic model and 200,000-draw Monte Carlo sensitivity.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs_economic")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, d_net = compute_economic_model()
    (out_dir / "economic_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["stp_sensitivity"]).to_csv(out_dir / "economic_stp_sensitivity.csv", index=False)
    plot_fig6(d_net, summary["monte_carlo"], out_dir / "fig6_economic_monte_carlo.png")

    d = summary["deterministic_table"]
    mc = summary["monte_carlo"]
    print("=== Economic Model & Monte Carlo Sensitivity Summary ===")
    print(
        f"Deterministic: Gross=${d['gross_annual_benefit_usd']:,.0f} | Net=${d['net_annual_benefit_usd']:,.0f} "
        f"| Cash=${d['annual_cash_benefit_usd']:,.0f} | Payback={d['payback_months']:.1f} mo | 3yr NPV=${d['npv_3yr_10pct_usd']:,.0f}"
    )
    print(
        f"Monte Carlo ({mc['n_draws']:,} draws, seed {mc['seed']}): "
        f"P10=${mc['p10_usd']:,.0f} (${mc['p10_usd']/1e6:.2f}M) | "
        f"Median=${mc['p50_median_usd']:,.0f} (${mc['p50_median_usd']/1e6:.2f}M) | "
        f"P90=${mc['p90_usd']:,.0f} (${mc['p90_usd']/1e6:.2f}M)"
    )
    print(
        f"Positive draws: {mc['pct_draws_positive']:.2f}% | "
        f"Analytical adverse support corner: ${mc['adverse_support_corner_usd']:,.0f}"
    )


if __name__ == "__main__":
    main()
