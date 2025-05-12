# Random Fatigue Limit Model (RFLM)

The repository provides classes for handling the RFLM. Currently only one class is implemented which is the 5 parameter RFLM model of 
    
    Pascal and Meeker 1999 - Estimating Fatigue Curves with the
    Random Fatigue-Limit Model, Sixth Annual Spring Research Conference
    Minneapolis, Minnesota, June 2–4, 1999

# Installation

Requires python 3.X or greater to be installed.


Create and activate a python virtual environment (exemplified for powershell)

```powershell
python -m venv <replace-with-venv-name>
<replace-with-venv-name>/Scripts/Activate.ps1 # or equivalent
```

The recommended install for general users is:

```powershell
# Install either by HTTPS
pip install git+https://github.com/equinor/rflm.git
# or by SSH
pip install git+ssh://git@github.com/equinor/rflm.git
```

The recommended install for developers is:

```powershell
# Clone using either HTTPS
git clone https://github.com/equinor/rflm.git
# or by SSH
git clone git@github.com:equinor/rflm.git
# Then perform a local install
python -m pip install -e .
```

## Example

A simple example of use can be seen in the provided rflm_script.py, which is a command line tool for fitting the 5-parameter RFLM. 

```unix
usage: rflm_script.py [-h] [-i filename] [-o identifier] [-v level] [-m value]

    Random fatigue limit code

    Reads fatigue data from a matlab or text file.

    Excel data file:
    ===============
    The excel file shall have column data, where the columns are separated
    by spaces and represent:

    Column 1:       Stress range
    Column 2:       Number of cycles to failure or runout
    Column 3:       Runout parameter (1 for runout, 0 for no runout)

    The filename of the text file shall NOT end in .mat

    RFLM model and Quantile SN Curve Computation:
    =================
    The `RFLM` supports user-defined quantiles and stress range limits.
```

You can run the first run rflm model following command:

```powershell
python .\rflm_script.py -i my_data.xlsx -o output_name -v 1
```
This will generate an excel file containing the fitted parameters, which you can use later to generate desired quantiles by using the following command:
```powershell
python ./run_quantile_model.py -i my_data.xlsx -p my_fitted_parameters.xlsx --quantiles 0.025 0.5 0.99 --slim 40 450 --nlim 1e4 1e7
```