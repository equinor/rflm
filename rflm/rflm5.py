import sys
import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import least_squares, minimize
from .opt import optimize_wrapper, adjust_constraints

# integrands for pdf and cdf
μ = lambda v, x, β0, β1: β0+β1*np.log(np.exp(x)-np.exp(v))
f_W_integrand2 = lambda v, w, x, β0, β1, σ, μ_γ, σ_γ: \
    1/(σ*σ_γ)*norm.pdf((w-μ(v, x, β0, β1))/σ)*norm.pdf((v-μ_γ)/σ_γ)
F_W_integrand2 = lambda v, w, x, β0, β1, σ, μ_γ, σ_γ: \
    1/σ_γ*norm.cdf((w-μ(v, x, β0, β1))/σ)*norm.pdf((v-μ_γ)/σ_γ)
f_W_integrand = lambda v, w, x, β0, β1, σ, μ_γ, σ_γ: \
    norm.pdf(w, loc=μ(v, x, β0, β1), scale=σ)*norm.pdf(v, loc=μ_γ, scale=σ_γ)
F_W_integrand = lambda v, w, x, β0, β1, σ, μ_γ, σ_γ: \
    norm.cdf(w, loc=μ(v, x, β0, β1), scale=σ)*norm.pdf(v, loc=μ_γ, scale=σ_γ)

# pdf and cdf (generated using gauss quadrature)
# Note that f_V is not used but is correct as per the paper
# The paper refers to the standard normal distribution. To represent f_V as in the paper
# we could have written f_V = lambda v, μ_γ, σ_γ: 1/σ_γ*norm.pdf((v-μ_γ)/σ_γ, loc=0, scale=1) 
f_V = lambda v, μ_γ, σ_γ: norm.pdf(v, loc=μ_γ, scale=σ_γ) 
f_W = lambda w, x, β0, β1, σ, μ_γ, σ_γ: quad(f_W_integrand, -np.inf, x, args=(w, x, β0, β1, σ, μ_γ, σ_γ))[0]                      
F_W = lambda w, x, β0, β1, σ, μ_γ, σ_γ: quad(F_W_integrand, -np.inf, x, args=(w, x, β0, β1, σ, μ_γ, σ_γ))[0]
f_W2 = lambda w, x, β0, β1, σ, μ_γ, σ_γ: quad(f_W_integrand2, -np.inf, x, args=(w, x, β0, β1, σ, μ_γ, σ_γ))[0]                      
F_W2 = lambda w, x, β0, β1, σ, μ_γ, σ_γ: quad(F_W_integrand2, -np.inf, x, args=(w, x, β0, β1, σ, μ_γ, σ_γ))[0]  

def quantile_function(q, x, θ): 
    β0, β1, σ, μ_γ, σ_γ = θ
    return lambda w: F_W(w, x, β0, β1, σ, μ_γ, σ_γ) - q


class RFLM5():
    
    __verbosity:int = 1
    
    @classmethod
    def set_verbosity(cls, value): cls.__verbosity = value

    def __init__(self, ΔS:np.ndarray=None, runout_bool:np.ndarray=None, N:np.ndarray=None, m:float=None, filename:str=None):
        """class to handle the 5-parameter random fatigue limit model of:

            Pascal and Meeker 1999 - Estimating Fatigue Curves with the
            Random Fatigue-Limit Model, Sixth Annual Spring Research Conference
            Minneapolis, Minnesota, June 2–4, 1999

            The model is 
            
                log(N) = β0 + β1.log(ΔS-γ)
                
            where the N and ΔS and the number of cycles to failure and the stress range, and γ is the fatigue limit.
            The parameters of the model are  
            
                β0: y-intercept of in the log-space
                β1: slope of SN-curve in the log-space
                σ: standard deviation of the log(N) data 
                μ_γ: expected fatigue limit in the log-space
                σ_γ: standard deviation of the fatigue limit in the log-space           

        Args:
            ΔS  (np.ndarray, optional): Stress ranges. Defaults to None
            runout_bool (np.ndarray, optional): Runout parameter (1 for runout, 0 for no runout). Defaults to None.
            N (np.ndarray, optional): Number of cycles to failure. Defaults to None.
            m (float, optional): Fixed slope of SN-curve. Defaults to None.
            filename (str, optional): Data file. If specified read_data_file is called. Defaults to None.
        """

        # variables for the data
        self.ΔS = None
        self.δ = None
        self.N = None
        self.m = m
        self.m_fixed = True if m is not None else False

        # parameters of the model
        self.β0 = None
        self.β1 = None
        self.σ = None
        self.μ_γ = None
        self.σ_γ = None
        
        # parameters of 2 parameter standard model
        self.a_lsq = None  
        self.m_lsq = m 
        
        # storage for computed quantiles
        self.quantile = {}
        
        if filename is not None:
            self.read_test_data(filename)
            self.data_set = True

        if np.all([ΔS, runout_bool, N]):            
            self.ΔS = ΔS
            self.δ = 1 - runout_bool
            self.N = N    
            self.data_set = True
            
          
    @property
    def params(self):
        """ 
        Returns a list of the parameters for the model log(N) = β0 + β1.log(ΔS-γ)
        The parameters returned are:
            β0: y-intercept of in the log-space
            β1: slope of SN-curve in the log-space
            σ: standard deviation of the log(N) data 
            μ_γ: expected fatigue limit in the log-space
            σ_γ: standard deviation of the fatigue limit in the log-space
            a: from least squared fitted N=a(ΔS)^(-m)
            m: from least squared fitted N=a(ΔS)^(-m)
        """
        return [self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ, self.a_lsq, self.m_lsq]
        
    
    @params.setter
    def params(self, values):
        """ 
        Sets the parameters for the model log(N) = β0 + β1.log(ΔS-γ)
        The parameters should be provided in list of values with entries:
            β0: y-intercept of in the log-space
            β1: slope of SN-curve in the log-space
            σ: standard deviation of the log(N) data 
            μ_γ: expected fatigue limit in the log-space
            σ_γ: standard deviation of the fatigue limit in the log-space
            a: from least squared fitted N=a(ΔS)^(-m)
            m: from least squared fitted N=a(ΔS)^(-m)
        The last two entries are optional. 
        """
        if len(values) == 7:
            self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ, self.a_lsq, self.m_lsq = values
        elif len(values) == 5:
            self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ = values
        else:
            raise IndexError("Expected a list of 5 or 7 entries")
        
    def read_test_data(self, filepath: str, cols:list=[0, 1, 2]):
        """
        Reads fatigue data from a matlab or text file. 
    
        Text data file:
        ===============
        The text file shall have column data, where the columns are separated 
        by spaces and represent:
        
        Column 1:       Stress range
        Column 2:       Runout parameter (1 for runout, 0 for no runout)
        Column 3:       Number of cycles to failure
        
        If a different column order is required use the cols keyword argument to 
        specify a list of column numbers e.g. cols=[5,4,1] corresponding to the
        column numbers of the stress range, runout and number of cycles in that
        order. Note that the cols=[5,4,1] corresponds to columns 6, 5, 2 respectively 
        
        The filename of the text file shall NOT end in .mat
        
        MATLAB data file:
        =================
        The matlab file shall have the following array names. 
        
        "X":            Stress range
        "runout_bool":  Runout parameter (1 for runout, 0 for no runout)
        "y":            Number of cycles to failure, N
        
        The filename of the matlab file shall end in .mat

        Args:
            filepath (str): filepath

        Raises:
            NotImplementedError: Raised if the data file cannot be read

        """
        from scipy.io import loadmat
        if filepath.endswith('.mat'):
            data = loadmat(filepath)
            try:
                ΔS = np.r_[[float(v) for v in np.squeeze(data["X"])]] # stress range
                δ = 1 - np.r_[[int(v) for v in np.squeeze(data["runout_bool"])]]
                N = np.r_[[float(v) for v in np.squeeze(data["y"])]] # number of cycles to failure        
            except:
                raise ValueError(f"Issue reading matlab file {filepath}. "
                                "Make sure it has array names 'X', 'y', 'runout_bool'")
        else:
            try:
                ΔS, runout_bool, N = np.loadtxt(filepath, usecols=cols).T
                δ = 1 - runout_bool
            except:
                raise NotImplementedError(f"data file format for {filepath} not recognized. "
                                        "Expecting a text file with three columns of data")
        self.ΔS = ΔS
        self.δ = δ
        self.N = N
        return
    
        
    def fit(self):
        """Performs a SLSQP fit to the 5 parameter RFLM model
        """
        if self.ΔS is None: 
            print("Need to set the data. Set data during construction or using read_test_data function")
            return 
        θ = RFLM5.__regression(self.ΔS, self.δ, self.N, self.m)
        self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ, self.a_lsq, self.m_lsq = θ
     
    
    def get_stress(self, q:float, N, params=None):
        """ A fast (evidently) approximate method to compute the stress from N """
        if params is None:
            θ = [β0, β1, σ, μ_γ, σ_γ] = self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ
        else:
            θ = [β0, β1, σ, μ_γ, σ_γ] = params
        γ = np.exp(norm.ppf(q, loc=μ_γ, scale=σ_γ))
        ΔS = np.exp(1/β1*(np.log(N)-β0)) + γ  #+  norm.ppf(q, loc=0, scale=σ)
        return ΔS
    
    
    def compute_quantile(self, q:float, ΔSmax:float=300, force:bool=False, npts=40, params=None, ΔS=None):
        """Computes the q-fractile curve of the RFLM fit

        Args:
            q (float): fractile (0 < q < 1)
            ΔSmax (float): upper stress range to compute curve. Defaults to 300. Units are those of data.
            force (bool, optional): if False, and q-fractile curve already computed, this
                routine will not recompute the curve. Default to False. Set to True to force recomputation.

        """
        if params is None:
            θ = [β0, β1, σ, μ_γ, σ_γ] = self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ
        else:
            θ = [β0, β1, σ, μ_γ, σ_γ] = params
      
        if None in θ: #not np.all(θ):
            print("Need to fit the data using the fit function")
            return 
        
        # Establish the curves at the relavent probability level
        # Option added for the user to specify his own stress ranges
        x_min = norm.ppf(q*1.001, loc=μ_γ, scale=σ_γ) 
        x = np.log(ΔS) if ΔS is not None else np.linspace(x_min, np.log(ΔSmax), npts)
        x = x[x>=x_min]  # ensure stress ranges are valid 
        n = np.zeros(x.size)

        if params is None:
            if q in self.quantile:
                if not force: 
                    print(f"quantile q={q} already computed. To force a recompute set force=True")
                    return # quantile already computed. 
            
        for j, xj in enumerate(x):
            try:
                wj = opt.brentq(quantile_function(q, xj, θ), -100, 1000) 
                n[j] = np.exp(wj)
                # check that is sufficiently converged
                if np.abs(F_W(wj, xj, β0, β1, σ, μ_γ, σ_γ) - q) > q*0.05:
                    wj = np.nan #np.inf 
                    n[j] = np.nan #np.inf             
            except:
                wj = np.nan #np.inf 
                n[j] = np.nan #np.inf 
            finally:
                if RFLM5.__verbosity == 1:
                    print(f"ΔS: {np.exp(xj):.1f}:  N: {n[j]:.0f}, F_W: {F_W(wj, xj, β0, β1, σ, μ_γ, σ_γ)}")  
                    
        # Store results
        if ΔS is not None: return [np.exp(x), n]
        if params is None:
            self.quantile[q] = [np.exp(x), n] # SN-data
        else:
            return [np.exp(x), n]
       
    
    
    def plot_SN_curve(self, ΔSmax:float=300, q:list=[0.025,0.5,0.975], 
                      nlim:list=None, slim:list=None,
                      npts=40, filename:str=None, show_lsq:bool=False, label=None, fast=False):
        """Plots the fitted RFLM model and the data

        Args:
            q (list, optional): q-fractiles to plot. Defaults to [0.025,0.5,0.975].
            nlim (list, optional): [min,max] for cycles to failure on x-axis e.g. [1E4, 1E8]. Defaults to None.
            slim (list, optional): [min,max] for stress range on y-axis. Defaults to None.
            npts (int, optional): number of points along the SN-curve. Defaults to 40.
            filename (str, optional): output filename. Defaults to None. 
            show_lsq (bool, optional): if True, the least-squared fitted 1-slope SN-curve is shown. 
                Defaults to False.

        Returns:
            fig, ax: if filename is None, the function returns the matplotlib figure and axes.
        """
                        
        θ = [β0, β1, σ, μ_γ, σ_γ] = self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ
        a, m = self.a_lsq, self.m_lsq
        if label is None: label = "RFLM"
        
        if None in θ: #not np.all(θ):
            print("Need to fit the data using the fit function")
            return 
                                              
        # Now plot the curves
        fig, ax = plt.subplots()
                                
        if q is not None:
            for i, qi in enumerate(q):
                if qi in self.quantile.keys():
                    [s, n] = self.quantile[qi]                
                else: 
                    if self.__verbosity == 1: print(f"q={qi} not computed.", end=" ")
                    if fast:
                        if self.__verbosity == 1: print("Computing using fast method.")
                        n = 10**np.linspace(1,12, npts)
                        self.quantile[qi] = [self.get_stress(qi, n), n] 
                    else:
                        if self.__verbosity == 1: print("Computing using compute_quantile.")                                    
                        self.compute_quantile(qi, ΔSmax=ΔSmax, npts=npts)
                    
                [s, n] = self.quantile[qi]                
                ax.plot(n, s, label=f"{label} (q={qi})", lw=2)

        else:
            for qi in sorted(self.quantile):
                [s, n] = self.quantile[qi]
                ax.plot(n, s, label=f"{label} (q={qi})", lw=2)
                
        if show_lsq and np.all([a,m]): # add the 2 parameter model curve
            if self.ΔS is not None: ax.plot(a*self.ΔS**-m, self.ΔS, label="LSQ (γ=0)", ls='--', color='k')
        # Add the data points
        if self.ΔS is not None:
            ax.scatter(self.N[self.δ==1], self.ΔS[self.δ==1], label="failures", marker="^")
            ax.scatter(self.N[self.δ==0], self.ΔS[self.δ==0], label="runouts", marker='o', facecolor="none", edgecolor="orange")
        # format plot
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="both")    
        ax.legend(loc=0)
        if nlim is not None: ax.set_xlim(*nlim)
        if slim is not None: ax.set_ylim(*slim)
        ax.legend(loc=0)
        ax.set_title(f"β0={β0:.3f}; β1={β1:.3f}; σ={σ:.3f}; μ_γ={μ_γ:.3f}; σ_γ={σ_γ:.3f}")
        if filename is not None:
            plt.savefig(filename)
            plt.close(fig)
        else:        
            return fig, ax
    
    def add_SN_curve(self, ax, params=None, q:list=[0.025,0.5,0.975], ΔSmax:float=300, npts=40, label=None, fast=False, **kwargs):
        if params is None:
            params = [β0, β1, σ, μ_γ, σ_γ] = self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ
            
        if len(params) == 5: # add a 5 parameter curve
            θ = [β0, β1, σ, μ_γ, σ_γ] = params
            if label is None: label = f"β0={β0:.3f}; β1={β1:.3f}; σ={σ:.3f};\nμ_γ={μ_γ:.3f}; σ_γ={σ_γ:.3f}" 
        else:
            raise IndexError("params should have 5 parameters")
        
        for i, qi in enumerate(q):
            if fast:
                n = 10**np.linspace(1,12, npts)
                s = self.get_stress(qi, n, params=params)
                self.quantile[qi] = [s, n] 
            else:
                [s, n] = self.compute_quantile(qi, ΔSmax=ΔSmax, npts=npts, params=params)
            ax.plot(n, s, label=f"{label} (q={qi})", lw=2, **kwargs)
        ax.legend(loc=0) # update legend
    
    def save(self, filename:str):
        """Saves the model parameters to file

        Args:
            filename (str): name of output file
        """
        import os
        from datetime import datetime
        if filename.endswith('xlsx'):
            params = [self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ, self.a_lsq, self.m_lsq]
            var_names = 'beta0 beta1 sigma mean_gamma sigma_gamma a_lsq m_lsq'.split()
            df = pd.DataFrame({'Parameter': var_names, 'Value': params})
            df.to_excel(filename, index=False)
        else:        
            with open(filename, 'w', encoding='utf-8') as fh:
                fh.write(f'# This file contains results from an SLSQP fit of the RLFM 5\n')
                fh.write(f'# parameter model to the test data. Fit performed\n')
                fh.write(f'# using a python code at https://github.com/equinor/rflm.\n#\n')
                fh.write(f'# Code run by {os.getlogin()} on {datetime.now().strftime("%Y-%m-%d %H:%M")}\n#\n')
                fh.write(f'# Parameters listed below are in accordance with the formulation in\n')        
                fh.write(f'# Pascal and Meeker 1999 - Estimating Fatigue Curves with the\n')
                fh.write(f'# Random Fatigue-Limit Model, Sixth Annual Spring Research Conference,\n')
                fh.write(f'# Minneapolis, Minnesota, June 2–4, 1999\n#\n')
                fh.write(f'# β0, β1, σ, μ_γ, σ_γ - parameters from the RFLM fit\n')
                fh.write(f"# a, m - parameters from a LS fit of a standard SN-curve\n")
                if self.m_fixed: fh.write(f'# β1 (m) specified as fixed by user\n#\n')
                fh.write(f'# β0, β1, σ, μ_γ, σ_γ, a, m\n')
                fh.write(f'{self.β0} {self.β1} {self.σ} {self.μ_γ} {self.σ_γ} {self.a_lsq} {self.m_lsq}\n')


    def load(self, filename:str):
        if filename.endswith('xlsx'):
            df = pd.read_excel(filename)
            self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ, self.a_lsq, self.m_lsq = df["Value"].values
        else:
            self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ, self.a_lsq, self.m_lsq = np.loadtxt(filename)


    def set_params(self, β0, β1, σ, μ_γ, σ_γ, a_lsq=None, m_lsq=None):
        self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ = β0, β1, σ, μ_γ, σ_γ
        if a_lsq is not None: self.a_lsq = a_lsq
        if m_lsq is not None: self.m_lsq = m_lsq


    def __model_fitted(self)->bool:
        θ = self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ
        if None in θ: #not np.all(θ):
            print("Need to fit the data using the fit function or manually set it using say set_params")
            return False
        return True


    def neg_log_likelihood_value(self, ΔS=None, N=None, δ=None):
        if not self.__model_fitted(): return
        θ = [β0, β1, σ, μ_γ, σ_γ] = self.β0, self.β1, self.σ, self.μ_γ, self.σ_γ
        if ΔS is None and N is None:        
            if not self.data_set: 
                print("Need to read data first")
                return
            else:
                ΔS = self.ΔS
                δ = self.δ
                N = self.N
        
        x = np.log(ΔS)
        w = np.log(N)
        ε = 1E-300
        return -np.sum([δi*np.log(max(f_W(wi, xi, β0, β1, σ, μ_γ, σ_γ),ε)) + 
                            (1-δi)*np.log(max(1-F_W(wi, xi, β0, β1, σ, μ_γ, σ_γ),ε)) 
                            for wi, xi, δi in zip(w,x,δ)])
        
    
    @staticmethod
    def __regression(ΔS:np.ndarray, δ:np.ndarray, N:np.ndarray, m:float=None):
        """Perform an SLSQP fit of the 5 parameter RFLM model to stress-failure data,
        taking into account the runouts.

        Args:
            ΔS (np.ndarray): stress range
            N (np.ndarray): number of cycles to failure
            δ (np.ndarray): 0 for runout, 1 for no runout
            m (float, optional): fixed slope of SN-curve. Defaults to None.

        Returns:
            list: [β0, β1, σ, μ_γ, σ_γ, a, m] parameters of the model 
                    including a and m from a LS fit of 1 slope SN-curve.
        """

        # data to fit
        x_data = np.log(ΔS)
        w_data = np.log(N)

        # establish an initial guess
        β_guess = np.array([10, -3 if m is None else -m])

        # Set a array for the fixed parameters. In the current code
        # only m (the slope of the SN-curve) can be fixed. This is 
        # always at index 1 in the optimization routine. 
        fixed_indices = [] if m is None else [1]
        fixed_values = [] if m is None else [-m]
        free_values = np.delete(β_guess, fixed_indices)

        # make an initial guess of the parameters based on an LSQ fit
        res = least_squares(optimize_wrapper, free_values, 
                       args=(fixed_values, fixed_indices, lambda β, x, w: w - β[0] - β[1]*x, x_data, w_data))
        result = res.x if m is None else np.insert(res.x, fixed_indices, fixed_values)   
        β0, β1 = result
        a, m = np.exp(β0), -β1
        σ = np.mean((w_data - β0 - β1*x_data)**2)
        μ_γ = 1.0 
        σ_γ = 0.2 
        θ_guess = [β0, β1, σ, μ_γ, σ_γ]    
        print(f"Initial guess for θ params: β0={β0:.2f}, β1={β1:.2f}, σ={σ:.2f}, μ_γ={μ_γ:.2f}, σ_γ={σ_γ:.2f}")
    
        # (negative) log-likelihood function for minimization
        def negative_log_likelihood_function(θ, w, x, δ):
            β0, β1, σ, μ_γ, σ_γ = θ
            ε = 1E-300
            if RFLM5.__verbosity == 1: print(θ)      
            # Both these forms should be identical
            return -np.sum([δi*np.log(max(f_W(wi, xi, β0, β1, σ, μ_γ, σ_γ),ε)) + 
                            (1-δi)*np.log(max(1-F_W(wi, xi, β0, β1, σ, μ_γ, σ_γ),ε)) 
                            for wi, xi, δi in zip(w,x,δ)])
            #return -np.sum([δi*np.log(max(f_W2(wi, xi, β0, β1, σ, μ_γ, σ_γ),ε)) + 
            #                (1-δi)*np.log(max(1-F_W2(wi, xi, β0, β1, σ, μ_γ, σ_γ),ε)) 
            #                for wi, xi, δi in zip(w,x,δ)])

        # define some constraints: expression >= 0
        constraints = [
            {'type': 'ineq', 'fun': lambda θ:  θ[0]}, #  β0 >= 0  | log(a) >= 0
            {'type': 'ineq', 'fun': lambda θ: -θ[1]}, #  β1 <= 0  | m >= 0
            {'type': 'ineq', 'fun': lambda θ:  θ[2]}, #   σ >= 0
            {'type': 'ineq', 'fun': lambda θ:  θ[3]}, # μ_γ >= 0 ?? Unsure about this one
            {'type': 'ineq', 'fun': lambda θ:  θ[4]}, # σ_γ >= 0
        ]    

        # define bounds: not used but can used for other optimization methods
        bounds = [(0,np.inf), (-np.inf,0), (None,None), (0, np.inf), (0, np.inf)]

        # Now handle fixed parameters for the main optimization
        free_values = np.delete(θ_guess, fixed_indices)
        adjusted_constraints = adjust_constraints(constraints, fixed_indices, fixed_values)   

        # Now minimize the negative log likelihood function to get RFLM parameters
        if RFLM5.__verbosity == 1: print("Printing parameters β0  β1  σ  μ_γ  σ_γ during optimization")
        res = minimize(optimize_wrapper, free_values, 
                       args=(fixed_values, fixed_indices, negative_log_likelihood_function, w_data, x_data, δ),
                       method='SLSQP', constraints=adjusted_constraints)
        result = res.x if m is None else np.insert(res.x, fixed_indices, fixed_values)     
        if res.success: print("Solution found")
        [β0, β1, σ, μ_γ, σ_γ] = result
        print(f"Optimized θ params: β0={β0:.3f}; β1={β1:.3f}; σ={σ:.3f}; μ_γ={μ_γ:.3f}; σ_γ={σ_γ:.3f}")

        # Now let's compute the quantiles for plotting
        θ = [β0, β1, σ, μ_γ, σ_γ, a, m]  

        return θ

