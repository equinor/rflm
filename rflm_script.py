def read_cli_arguments():
    """ Returns the command line arguments """
    import argparse
    description = """
    Random fatigue limit code 
    
    Reads fatigue data from a matlab or text file. 
    
    Text data file:
    ===============
    The text file shall have column data, where the columns are separated 
    by spaces and represent:
    
    Column 1:       Stress range
    Column 2:       Runout parameter (1 for runout, 0 for no runout)
    Column 3:       Number of cycles to failure
    
    The filename of the text file shall NOT end in .mat
    
    MATLAB data file:
    =================
    The matlab file shall have the following array names. 
    
    "X":            Stress range
    "runout_bool":  Runout parameter (1 for runout, 0 for no runout)
    "y":            Number of cycles to failure, N
    
    The filename of the matlab file shall end in .mat"""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter)
    #parser.add_argument('data_filename', metavar="filename", type=str, help='The input data file')
    parser.add_argument('-i', metavar="filename", type=str, default="data.txt", help='Input data file name (default: %(default)s)')
    parser.add_argument('-o', metavar="identifier", type=str, default="output", help='Optional identifier for the outputted data (default: %(default)s)')
    parser.add_argument('-v', metavar="level", type=int, default=0, help="verbosity level for printing (0=no print, 1=print), (default: %(default)s)")
    parser.add_argument('-m', metavar="value", type=float, default=None, help="Fixed value for slope of SN-curve")
    parser.add_argument('--nlim', metavar=('min(N)','max(N)'), type=float, nargs=2, default=None, 
                        help="maximum and minimum cycles to failure to show on a plot")
    parser.add_argument('--slim', metavar=('min(ΔS)','max(ΔS)'), type=float, nargs=2, default=None, 
                        help="maximum and minimum stress range to show on a plot")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    
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
    #(ΔS, δ, N) = RFLM5.read_test_data(filename="sharepoint/data_Mikulski2022.mat", m=args.m)
    #(ΔS, δ, N) = RFLM5.read_test_data(filename="sharepoint/data_Stojkovic2018.mat", m=args.m)
    #(ΔS, δ, N) = RFLM5.read_test_data(filename="sharepoint/data_dnv_correspondence231026.mat", m=args.m)

    # Fit the model
    rflm.fit()
    
    # plot the SN-curve
    rflm.plot_SN_curve(filename=args.o + '.png', nlim=args.nlim)

    # Now export the results
    rflm.save(args.o + '.txt')



