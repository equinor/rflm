def read_cli_arguments():
    """ Returns the command line arguments """
    import argparse
    description = """
    Random fatigue limit code 
    
    Reads fatigue data from an Excel file. 
    
    Excel data file:
    =================
    The Excel file shall have column data, where the columns are represented as:
    
    Column 1:       Stress range
    Column 2:       Number of cycles to failure
    Column 3:       Runout parameter (1 for runout, 0 for no runout)
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', metavar="filename", type=str, default="data.xlsx", help='Input data file name (default: %(default)s)')
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
    
    from rflm import RFLM5
    
    # Get command line arguments
    args = read_cli_arguments()

    # Set verbosity level
    import warnings
    verbosity = args.v
    if verbosity == 0: warnings.filterwarnings("ignore")
    RFLM5.set_verbosity(verbosity)

    # Instantiate an RFLM5 class instance using the inputted data
    rflm = RFLM5(filename=args.i, m=args.m)

    # Fit the model
    rflm.fit()
    
    # Plot the SN-curve
    rflm.plot_SN_curve(filename=args.o + '.png', nlim=args.nlim)

    # Now export the results to an Excel file
    rflm.save(args.o + '.xlsx')  # Save as Excel file
    print(f"Data has been successfully converted to {args.o}.xlsx.")
