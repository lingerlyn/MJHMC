"""
This module contains the objective function used by Spearmint
  when searching for good hyperparameters.
In short, the objective function is the Re(r) where
  r is a complex number such that exp(r * t) is the
  best fit to the autocorrelation
"""

import numpy as np
from scipy.optimize import curve_fit
from mjhmc.misc.autocor import calculate_autocorrelation
from mjhmc.misc.plotting import plot_fit, plot_search_ac
from mjhmc.samplers.markov_jump_hmc import ContinuousTimeHMC

grad_evals = {
    'Gaussian' : int(5E4),
    'RoughWell' : int(5E4),
    'MultimodalGaussian' : int(2E5),
    'ProductOfT' : int(1E5)
}

debug = True


def obj_func(sampler, distr, job_id, **kwargs):
    """ Scores the performance of sampler on distribution given parameters

    :param sampler: sampler being tested. instance of mjhmc.samplers.markov_jump_hmc.HMCBase
    :param distr: distribution being used. instance of mjhmc.misc.distributions.Distribution
    :param job_id: integer label for job being run
    :returns: the score
    :rtype: float
    """
    cos_coef, normed_n_grad_evals, exp_coef, autocor, kwargs = obj_func_helper(sampler, distr, True, kwargs)
    if debug:
        plot_fit(normed_n_grad_evals, autocor, exp_coef, cos_coef, job_id, kwargs)
    return exp_coef

def obj_func_helper(sampler, distr, unpack, kwargs):
    """ Helper function for the objective function

    :param sampler: sampler being tested. instance of mjhmc.samplers.markov_jump_hmc.HMCBase
    :param distr: distribution being used. instance of mjhmc.misc.distributions.Distribution
    :param unpack: boolean flag of whether to unpack params or not.
    :param kwargs: dictionary of kwargs passed from parent function
    :returns: parameters of fitted curves, graph data for computed autocorrelation, kwargs
    :rtype: tuple

    """
    num_target_grad_evals = grad_evals[type(distr).__name__]
    default_args = {
        "num_grad_steps": num_target_grad_evals,
        "sample_steps": 1,
        "num_steps": None,
        "half_window": True,
        "use_cached_var": True
    }
    if unpack:
        kwargs = unpack_params(kwargs)
    if sampler.__name__ == 'MarkovJumpHMC':
        default_args["resample"] = False
    kwargs.update(default_args)

    print "Calculating autocorrelation for {} grad evals".format(num_target_grad_evals)
    ac_df = calculate_autocorrelation(sampler, distr, **kwargs)
    n_grad_evals = ac_df['num grad'].values.astype(int)
    # necessary to keep curve_fit from borking: THIS IS VERY IMPORTANT
    normed_n_grad_evals = n_grad_evals / (0.5 * num_target_grad_evals)
    autocor = ac_df['autocorrelation'].values
    print "Fitting curve"
    exp_coef, cos_coef = fit(normed_n_grad_evals.copy(), autocor.copy())
    return cos_coef, normed_n_grad_evals, exp_coef, autocor, kwargs


def min_idx(ac_df, target):
    ac_df.index = ac_df['num grad']
    ac_trunc = ac_df.loc[:, 'autocorrelation'] < target
    small_ac = ac_trunc[ac_trunc]
    if len(small_ac) != 0:
        return small_ac.index[0]
    else:
        return None

def unpack_params(params):
    """
    Spearmint passes params as 1x1 arrays.
    returns unpacked dict params
    probably not a problem, but just in case
    """
    unpacked_params = {}
    for key, item in params.iteritems():
        unpacked_params[key] = item[0]
    return unpacked_params

def fit(t_data, y_data):
    """ Fit a complex exponential to y_data

    :param t_data: array of values for t-axis (x-axis)
    :param y_data: array of values for y-axis. of the same shape as t-data
    :returns: fitted parameters: (exp_coef, cos_coef)
    :rtype: tuple
    """
    # very fast way to check for nan
    if not np.isnan(np.sum(y_data)):
        # p_0 = estimate_params(t_data, y_data)
        p_0 = None
        opt_params = curve_fit(curve, t_data, y_data, p0=p_0, maxfev=1000)[0]
        return opt_params
    else:
        return 1E3, 0

def estimate_params(t_data, y_data):
    """ Estimate the parameters to the complex exponential fit
    Would be exact if derivatives were

    :param t_data: array of values for t-axis (x-axis)
    :param y_data: array of values for y-axis. of the same shape as t-data
    :returns: estimated parameters
    :rtype: tuple
    """
    dydt_0 = (y_data[0] - y_data[1]) / float(t_data[0] - t_data[1])
    exp_coef = dydt_0

    zero_idx = np.where(y_data == 0)[0]
    # candidate for improvement: does not handle arrays that contain 0 properly
    zero_crossings = np.where(np.diff(np.sign(y_data)) != 0)[0]
    if len(zero_idx) != 0 and len(zero_crossings) != 0:
        first_zero_idx = min(zero_crossings[0], zero_idx[0])
    elif len(zero_crossings) != 0:
        first_zero_idx = zero_crossings[0]
    elif len(zero_idx) != 0:
        raise ValueError("zeros should be found if and only if a zero crossing is found")
    else:
        return exp_coef, 0

    dydt_zero_cross = (y_data[first_zero_idx] - y_data[first_zero_idx + 1]) / float(t_data[first_zero_idx] - t_data[first_zero_idx + 1])
    cos_coef = - (dydt_zero_cross / np.exp(exp_coef * (first_zero_idx + 0.5)))
    return exp_coef, cos_coef



def curve(n, a, b):
    return np.exp(a * n) * np.cos(b * n)
