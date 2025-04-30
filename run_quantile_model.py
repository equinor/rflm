import argparse
import os
import pandas as pd
from rflm import RFLMQuantileModel

def main():
    parser = argparse.ArgumentParser(description="Compute and plot quantile SN-curves using RFLM.")
    parser.add_argument('-i', '--input', required=True, help='Path to experimental Excel file')
    parser.add_argument('-p', '--params', required=True, help='Path to fitted parameters Excel file')
    parser.add_argument('--quantiles', type=float, nargs='+', required=True, help='List of quantiles to compute, e.g., --quantiles 0.01 0.5 0.99')
    parser.add_argument('--nlim', type=float, nargs=2, help='Limits for x-axis (Cycles to Failure), e.g., --nlim 1e4 1e8')
    args = parser.parse_args()

    # === Load input data ===
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Experimental data file not found: {args.input}")
    if not os.path.exists(args.params):
        raise FileNotFoundError(f"Fitted parameter file not found: {args.params}")

    # Load fitted parameters
    df_params = pd.read_excel(args.params)
    θ = df_params['Value'].values[:5]

    # Load experimental fatigue data
    df_exp = pd.read_excel(args.input, header=None)
    df_exp = df_exp.iloc[:, :3]
    df_exp.columns = ["Stress", "Cycles", "Runout"]
    df_exp = df_exp.dropna()
    df_exp["Stress"] = pd.to_numeric(df_exp["Stress"], errors='coerce')
    df_exp["Cycles"] = pd.to_numeric(df_exp["Cycles"], errors='coerce')
    df_exp["Runout"] = pd.to_numeric(df_exp["Runout"], errors='coerce')

    # === Run model ===
    model = RFLMQuantileModel(θ)
    df_quantiles = model.compute_quantile_curves(quantiles=args.quantiles)

    # === Output filenames ===
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_excel = f"{base_name}_Quantiles.xlsx"
    output_plot = f"{base_name}_Quantiles.png"

    # === Save results ===
    df_quantiles.to_excel(output_excel, index=False)
    model.plot(df_quantiles, df_exp=df_exp, filename=output_plot, nlim=args.nlim)

    print(f"✅ Quantile curves saved to '{output_excel}'")
    print(f"📊 Plot saved to '{output_plot}'")

if __name__ == "__main__":
    main()
