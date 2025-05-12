import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm import tqdm

class RFLMQuantileModel:
    def __init__(self, θ):
        self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ = θ
        self.EPS = 1e-300

    def μ(self, v, x):
        return self.β0 + self.β1 * np.log(np.exp(x) - np.exp(v))

    def f_W_integrand(self, v, w, x):
        return norm.pdf(w, loc=self.μ(v, x), scale=self.σ) * norm.pdf(v, loc=self.μ_γ, scale=self.σ_γ)

    def F_W_integrand(self, v, w, x):
        return norm.cdf(w, loc=self.μ(v, x), scale=self.σ) * norm.pdf(v, loc=self.μ_γ, scale=self.σ_γ)

    def F_W(self, w, x):
        return quad(self.F_W_integrand, -np.inf, x, args=(w, x), limit=100)[0]

    def quantile_function(self, q, x):
        return lambda w: self.F_W(w, x) - q

    def compute_quantile_curves(self, ΔS_min=None, ΔS_max=None, npts=40, quantiles=None, df_exp=None):
        if quantiles is None:
            raise ValueError("Quantiles must be provided.")

        # Automatically determine stress range from experimental data if not given
        if (ΔS_min is None or ΔS_max is None) and df_exp is not None:
            stress_values = df_exp["Stress"]
            if ΔS_min is None:
                ΔS_min = stress_values.min()
            if ΔS_max is None:
                ΔS_max = stress_values.max()

        # Fallback defaults if nothing is provided
        if ΔS_min is None:
            ΔS_min = 40
        if ΔS_max is None:
            ΔS_max = 300

        x_vals = np.linspace(np.log(ΔS_min), np.log(ΔS_max), npts)
        results = []

        print(f"Computing quantile SN curves for ΔS ∈ [{ΔS_min:.1f}, {ΔS_max:.1f}]...")
        for q in tqdm(quantiles, desc="Quantiles"):
            for x in tqdm(x_vals, desc=f"q={q:.3f}", leave=False):
                try:
                    w = brentq(self.quantile_function(q, x), -100, 1000)
                    N = np.exp(w)
                    ΔS = np.exp(x)
                    results.append({
                        "Quantile": q,
                        "Stress Range": ΔS,
                        "Cycles to Failure": N
                    })
                except Exception:
                    continue
        print("✅ Done computing SN curves.")
        return pd.DataFrame(results)

    def plot(self, df_quantiles, df_exp=None, filename=None, nlim=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        for q in sorted(df_quantiles["Quantile"].unique()):
            df_q = df_quantiles[df_quantiles["Quantile"] == q]
            ax.plot(df_q["Cycles to Failure"], df_q["Stress Range"], label=f"q = {q:.3f}", lw=2)

        if df_exp is not None:
            failures = df_exp[df_exp["Runout"] == 0]
            runouts = df_exp[df_exp["Runout"] == 1]
            ax.scatter(failures["Cycles"], failures["Stress"], label="Failures", marker="^", color="black")
            ax.scatter(runouts["Cycles"], runouts["Stress"], label="Runouts", facecolors='none', edgecolors='red', marker="o")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cycles to Failure (N)")
        ax.set_ylabel("Stress Range (ΔS)")
        ax.set_title("Extended Quantile SN-Curves with Experimental Data (RFLM)")
        ax.legend(title="Quantile / Data", loc="best", fontsize='small')
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        if nlim is not None:
            ax.set_xlim(nlim)
        plt.tight_layout()

        if filename:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()
