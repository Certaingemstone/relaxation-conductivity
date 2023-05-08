# Jade Chongsathapornpong for MCSC Scholars, 2023

import numpy as np
import lmfit
from scipy.special import erf, erfc

import re

kB = 8.617333262e-5 # eV/K

############################### UTILITIES ###############################
def stage_to_furnace(T_stage):
    """From Jade's Jan. 2023 calibration with a Pt RTD on alumina substrate,
    return a blue furnace sample temperature given a Huber stage setpoint (in C)"""
    return (-4.36189550e-05 * T_stage**2 + 1.12177203*T_stage - 1.41614837e+01)
    
def int_from_filename(filename):
    """Given a filename, returns an integer, e.g. "furnace575C" -> 575 (!assumes no other numbers!)"""
    non_decimal = re.compile(r'[^\d]+') # gets the not-decimals
    return int(non_decimal.sub('', filename)) # replace the not-decimals with empty

############################### MODEL FUNCTIONS ###############################

### Impedance Spectra ###

def impedance_2d(x: np.ndarray, R1: float, R2: float, Q: float, n: float) -> np.ndarray:
    """Given x frequencies in rad/s, resistances R1 and R2, CPE parameters Q and n,
    return a concatenated array of length 2*len(x) of the real, then imaginary components
    of the corresponding equivalent circuit (R1 in series, R2 and CPE in parallel)."""
    # calculate common denominator for each omega
    theta = 0.5*n*np.pi
    denom1 = R2**2 * Q**2 * (np.sin(theta))**2 * np.power(x, 2*n)
    denom2 = ((R2 * Q * np.cos(theta) * np.power(x, n)) + 1)**2
    denom = denom1 + denom2
    # calculate the real and imaginary part for each omega
    Re = R1 + (R2 + R2**2 * Q * np.cos(theta) * np.power(x, n))/denom
    Im = (-R2**2 * Q * np.sin(theta) * np.power(x, n))/denom
    # return as one concatenated array of length 2*len(x)
    return np.concatenate([Re, Im])
    
def make_impedance_model(R1_guess: float, R2_guess: float, Q_guess: float):
    """Return an lmfit Model and Params object corresponding to impedance_2d, with the initial guesses given."""
    model = lmfit.Model(impedance_2d, independent_vars=['x'])
    params = lmfit.Parameters()
    params.add('R1', value=R1_guess, min=0)
    params.add('R2', value=R2_guess, min=0)
    params.add('Q', value=Q_guess)
    params.add('n', value=0.9, min=0.5, max=1.0)
    return model, params
    
### Relaxation Curves ###
def oxidation(x, r0, driftslope, offsetX, offsetY):
    '''
    Models a single relaxation to higher impedance.
    x : usually time, independent variable
    A : Exponential scaling/translation parameter
    exp_lim : Top value for x -> infinity
    driftslope : slope of underlying drift
    '''
    return (1 - np.exp(-r0*(x - offsetX))) + ((x - offsetX) * driftslope / 1000) + offsetY

def reduction(x, r0, driftslope, offsetX, offsetY):
    '''
    Models a single relaxation to lower impedance.
    '''
    return (np.exp(-r0*(x - offsetX))) + ((x - offsetX) * driftslope / 1000) + offsetY

def make_relax_exp_model(r0_guess, driftslope_guess, offX, offY, expfunc, vary=True):
    model = lmfit.Model(expfunc, independent_vars=['x'])
    params = lmfit.Parameters()
    params.add('r0', value=r0_guess)
    slope_param = ((driftslope_guess > 1e-3) or (driftslope_guess < -1e-3))
    params.add('driftslope', value=driftslope_guess, vary=slope_param)
    params.add('offsetX', value=offX, vary=not slope_param)
    params.add('offsetY', value=offY, vary=vary)
    return model, params

def relaxation_curve(x, x0, x1, r0, r1, exp_lim1, exp_lim2, driftslope, dx):
    '''
    Models a pair of relaxations (graph looks like a capacitor charging and discharging)
    x0 : first PO2 step time
    x1 : second PO2 step time
    r0 : first exponential time constant
    r1 : second exponential time constant
    exp_lim1 : limiting value of first relaxation without drift
    exp_lim2 : limiting value of second relaxation without drift
    driftslope : slope of underlying drift
    '''
    # weighting factors
    # weight0 = 1-(erf((x-x0)/dx) + 1)/2.0
    weight2 = 1-(erfc((x-x1)/dx))/2.0
    weight1 = (erf((x-x0)/dx) + 1 + erfc((x-x1)/dx)) / 2 - 1
    # functions
    return weight1 * (exp_lim1-np.exp(-r0*(x-x0))) + weight2 * (np.exp(-r1*(x-x1)) + exp_lim2) + x * driftslope

def make_relaxation_curve_model(t_step1, t_step2, r0_guess, r1_guess, exp_lim1_guess=1, exp_lim2_guess=0, driftslope_guess=0, dx=0.1):
    model = lmfit.Model(relaxation_curve, independent_vars=['x'])
    params = lmfit.Parameters()
    params.add('x0', value=t_step1, min=0)
    params.add('x1', value=t_step2, min=0)
    params.add('r0', value=r0_guess, min=0, max=2)
    params.add('r1', value=r1_guess, min=0, max=2)
    params.add('exp_lim1', value=exp_lim1_guess)
    params.add('exp_lim2', value=exp_lim2_guess)
    params.add('driftslope', value=driftslope_guess, vary=driftslope_guess != 0)
    params.add('dx', value=dx, vary=False);
    return model, params

### Activation Energies ###
def conductivity(x, E, A):
    """Return conductivity per simple exponential suppression by activation energy E, x is temperature"""
    return A * np.exp(-E / (kB * x))
    
def conductivity_corrected(x, E, A):
    """From Thomas' thesis"""
    return (A / x) * np.exp(-E / (kB * x))
    
def make_conductivity_model(E_guess, E_min=1e-19, A_guess=100, temperature_correction=True):
    model = lmfit.Model(conductivity_corrected, independent_vars=['x']) if temperature_correction else lmfit.Model(conductivity, independent_vars=['x'])
    params = lmfit.Parameters()
    params.add('E', value=E_guess, min=E_min)
    params.add('A', value=A_guess)
    return model, params


if __name__ == "__main__":
    pass