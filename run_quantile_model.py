import argparse
import os
import pandas as pd
from rflm import RFLMQuantileModel, RFLMGeneralModel

def main():
    parser = argparse.ArgumentParser(description="Compute and plot quantile SN-curves using RFLM.")
    parser.add_argument('-i', '--input', required=True, help='Path to experimental Excel file')
    parser.add_argument('-p', '--params', required=True, help='Path to fitted parameters Excel file')
    parser.add_argument('--quantiles', type=float, nargs='+', required=True, help='List of quantiles to compute, e.g., --quantiles 0.01 0.5 0.99')
    parser.add_argument('--nlim', type=float, nargs=2, help='Limits for x-axis (Cycles to Failure), e.g., --nlim 1e4 1e8')
    parser.add_argument('--slim', type=float, nargs=2, help='Stress range limits (min max), e.g., --slim 50 300')
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

    # === Run quantile model ===
    model = RFLMQuantileModel(θ)
    if args.slim:
        ΔS_min, ΔS_max = args.slim
    else:
        ΔS_min = ΔS_max = None  # auto-detect from df_exp

    df_quantiles = model.compute_quantile_curves(
        ΔS_min=ΔS_min,
        ΔS_max=ΔS_max,
        quantiles=args.quantiles,
        df_exp=df_exp
    )

    # === Compute general SN model curve ===
    general_model = RFLMGeneralModel(θ)
    N_gen, S_gen = general_model.compute_sn_curve(ΔS_max=args.slim[1] if args.slim else 450)
    df_general = pd.DataFrame({"Quantile": ["RFLM"]*len(N_gen), "Stress Range": S_gen, "Cycles to Failure": N_gen})

    # === Merge and Save ===
    df_combined = pd.concat([df_quantiles, df_general], ignore_index=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_excel = f"{base_name}_Quantiles+General.xlsx"
    output_plot = f"{base_name}_Quantiles+General.png"
    df_combined.to_excel(output_excel, index=False)

    # === Plot ===
    model.plot(df_combined, df_exp=df_exp, filename=output_plot, nlim=args.nlim)

    print(f"✅ Quantile + general curves saved to '{output_excel}'")
    print(f"📊 Plot saved to '{output_plot}'")

if __name__ == "__main__":
    main()
