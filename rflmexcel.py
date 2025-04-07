import pandas as pd
import scipy.io
import argparse
import warnings
from rflm import RFLM5

def convert_excel_to_mat(excel_file, mat_file):
    # Read the Excel file
    df = pd.read_excel(excel_file)

    # Check the columns in the DataFrame
    print("Original Columns:", df.columns)

    # Ensure there are at least 3 columns to swap
    if df.shape[1] < 3:
        raise ValueError("The Excel file must contain at least three columns.")

    # Swap columns 2 and 3 (assuming 0-based indexing)
    df.iloc[:, [1, 2]] = df.iloc[:, [2, 1]].values

    # Prepare data for .mat file
    stress_range = df.iloc[:, 0].values  # Column 1
    runout_parameter = df.iloc[:, 1].values  # Column 3 (originally column 2)
    cycles_to_failure = df.iloc[:, 2].values  # Column 2 (originally column 3)

    # Create a dictionary to save in .mat format
    mat_data = {
        'X': stress_range,
        'runout_bool': runout_parameter,
        'y': cycles_to_failure
    }

    # Save to .mat file
    scipy.io.savemat(mat_file, mat_data)
    print(f"Data has been successfully converted to {mat_file}.")

def read_cli_arguments():
    """ Returns the command line arguments """
    description = """
    Random fatigue limit code 
    
    Reads fatigue data from an Excel file and converts it to a .mat file.
    
    Excel data file:
    =================
    The Excel file shall have column data, where the columns represent:
    
    Column 1:       Stress range
    Column 2:       Runout parameter (1 for runout, 0 for no runout)
    Column 3:       Number of cycles to failure
    
    The filename of the Excel file should end with .xlsx or .xls."""
    
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', metavar="filename", type=str, required=True, help='Input Excel file name (must end with .xlsx or .xls)')
    parser.add_argument('-o', metavar="identifier", type=str, default="output", help='Optional identifier for the outputted data (default: %(default)s)')
    parser.add_argument('-v', metavar="level", type=int, default=0, help="verbosity level for printing (0=no print, 1=print), (default: %(default)s)")
    parser.add_argument('-m', metavar="value", type=float, default=None, help="Fixed value for slope of SN-curve")
    parser.add_argument('--nlim', metavar=('min(N)','max(N)'), type=float, nargs=2, default=None, 
                        help="maximum and minimum cycles to failure to show on a plot")
    parser.add_argument('--slim', metavar=('min(ΔS)','max(ΔS)'), type=float, nargs=2, default=None, 
                        help="maximum and minimum stress range to show on a plot")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Get command line arguments
    args = read_cli_arguments()

    # Convert the provided Excel file to .mat format
    mat_file = args.o + '.mat'
    convert_excel_to_mat(args.i, mat_file)

    # Set verbosity level
    verbosity = args.v
    if verbosity == 0: warnings.filterwarnings("ignore")
    RFLM5.set_verbosity(verbosity)

    # Instantiate an RFLM5 class instance using the converted .mat file
    rflm = RFLM5(filename=mat_file, m=args.m)

    # Fit the model
    rflm.fit()
    
    # Plot the SN-curve
    rflm.plot_SN_curve(filename=args.o + '.png', nlim=args.nlim)

    # Prepare output parameters for Excel
    output_parameters = {
        'Parameter Symbol': ['β0', 'β1', 'σ', 'μ_γ', 'σ_γ', 'a', 'm'],  # Parameter symbols
        'Value': [rflm.β0, rflm.β1, rflm.σ, rflm.μ_γ, rflm.σ_γ, rflm.a_lsq, rflm.m_lsq]  # Actual attributes from the RFLM5 instance
    }
    
    # Create a DataFrame
    output_df = pd.DataFrame(output_parameters)

    # Save to Excel file
    output_excel_file = args.o + '_parameters.xlsx'
    output_df.to_excel(output_excel_file, index=False)
    print(f"Output parameters have been saved to {output_excel_file}.")
