import pandas as pd
import argparse
import sys
from rflm import RFLM5  # Ensure your RFLM5 class is available as rflm5.py

def main():
    parser = argparse.ArgumentParser(description="Compute negative log-likelihood from fatigue test data and fitted RFLM parameters")
    parser.add_argument('-i', '--input', type=str, required=True, help='Excel file with fatigue test data')
    parser.add_argument('-p', '--params', type=str, required=True, help='Excel file with fitted parameters')
    args = parser.parse_args()

    try:
        # === Load fatigue data from columns ===
        df = pd.read_excel(args.input)
        if df.shape[1] < 3:
            raise ValueError("Input data must have at least three columns: Stress, Cycles, Runout")

        ΔS = df.iloc[:, 0].values  # Column 1: Stress
        N = df.iloc[:, 1].values   # Column 2: Cycles to failure
        runout = df.iloc[:, 2].values  # Column 3: Runout (1 = runout, 0 = failure)
        δ = 1 - runout  # Convert to δ: 1 = failure, 0 = runout

        # === Load fitted parameters ===
        params_df = pd.read_excel(args.params)
        params_df.columns = params_df.columns.str.strip()
        params_df['Parameter'] = params_df['Parameter'].str.strip()
        param_map = params_df.set_index("Parameter")["Value"].to_dict()

        # Support both 'mu_gamma' and 'mean_gamma'
        if "mu_gamma" not in param_map and "mean_gamma" in param_map:
            param_map["mu_gamma"] = param_map["mean_gamma"]

        required_keys = ['beta0', 'beta1', 'sigma', 'mu_gamma', 'sigma_gamma']
        missing = [k for k in required_keys if k not in param_map]
        if missing:
            raise ValueError(f"Missing required parameter(s): {missing}")

        # === Initialize model and compute log-likelihood ===
        model = RFLM5(ΔS, 1 - δ, N)
        model.set_params(
            param_map['beta0'], param_map['beta1'], param_map['sigma'],
            param_map['mu_gamma'], param_map['sigma_gamma']
        )
        neg_logL = model.neg_log_likelihood_value(ΔS, N, δ)
        print(f"✅ Negative Log-Likelihood = {neg_logL:.6f}")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
