"""
This file contains helpful base functions used in Monte Carlo sampling/simulation,
including functions for fitting and sampling from various bounded distributions.
"""
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd

import re
from copy import deepcopy

#Viz
import matplotlib.pyplot as plt
import seaborn as sns



class MonteCarlo:
    """
    Stores methods for distribution fitting and sampling from distributions in Monte Carlo sampling.
    """
    def __init__(self):
        """
        Simulation options are initialized (e.g. strategies, scenarios vars).
        """
        # Examples - are basements constructed? Where in the forecast is the growth?
        self.basements = True
        self.growth = 0.95



    def pert(self, a, b, c, size=1, l=4):
        """Returns realizations from a PERT distribution with
        minimum a, most likely b, and maximum c. Used for material
        GHG factors/intensities."""
        r = c - a

        # If r = 0 (only 1 datapoint, a=b=c), then return a np array of all that value.
        if (a >= c):
            return np.full(size, a)

        else:
            alpha = 1 + l * (b - a) / r
            beta = 1 + l * (c - b) / r

            # Scale beta distribution to min and max.
            pert_r = a + np.random.beta(alpha, beta, size=size) * r

            return pert_r



    def pert_fit(self, a, b, c, l=4, expand=False):
        """Part 1 of the overall pert function. Fits a PERT
         distribution to minimum a, most likely b, and
        maximum c. Returns distribution parameters alpha and beta"""
        r = c - a

        # If r = 0 (only 1 datapoint, a=b=c), then return foobar alpha beta values
        if (a >= c):
            if expand == True:
                print('Info: cannot fit beta, returned a, b = -1')

            alpha = -1
            beta = -1
            
            return (alpha, beta)

        else:
            alpha = 1 + l * (b - a) / r
            beta = 1 + l * (c - b) / r

            return (alpha, beta)
        


    def pert_sample(self, a, c, alpha, beta, size):
        """Part 2 of the overall pert function. Returns n
        samples from a PERT distribution given size=n,
        beta parameters, max and min."""
        r = c - a

        if (a >= c):
            return np.full(size, a)
        
        else:
            pert_r = a + np.random.beta(alpha, beta, size=size) * r

            return pert_r



    def fit_bounded_distribution(self, data, default_uniform = True, low_bound=5):
        """
        Given a set of data, fits the data to multiple, bounded
        distributions and returns the name + parameters of the distribution
        which maximizes the K-S test p-value.
        low_bound -> if the len of data is less than this value and default_uniform = True, 
        the method fits a uniform distribution regardless of the fit test.
        """
        # Def candidates and fit. stats.norm is not bounded so not considered
        distributions = (stats.triang, stats.uniform, stats.beta) # stats.truncnorm was not working!
        dis_names = ('triang','uniform', 'beta')

        # Have a fallback for "FitError"
        try:
            parameters = [d.fit(data) for d in distributions]

            # Calculate KS statistic
            ks_res = [stats.kstest(rvs=data, cdf=dis_names[i], args=parameters[i],
                                ) for i in (range(len(dis_names)))]

            # Return max index
            ks_p = [k[1] for k in ks_res]
            i_max = max(range(len(ks_p)), key=ks_p.__getitem__) 

        except:
            print('FIT ERROR EXCEPTION: DEFAULTING TO UNIFORM')
            # default to uniform
            print('Info: fitting error, defaulting to uniform distribution.')
            return {'best_distribution': dis_names[1], 'd_object': distributions[1], 'params': stats.uniform.fit(data), 'p': 1}

        # Default to uniform distribution for low data option.
        if (default_uniform == True) & (len(data) < low_bound):
            print('Info: sample size below threshold, defaulting to uniform distribution.')
            return {'best_distribution': dis_names[1], 'd_object': distributions[1], 'params': parameters[1], 'p': ks_p[1]}
        else:
            return {'best_distribution': dis_names[i_max], 'd_object': distributions[i_max], 'params': parameters[i_max], 'p': ks_p[i_max]}



    def bounded_distribution(self, d_object, params, size):
        """
        Returns realizations from a fit bounded distribution with
        params, location, and scale. Used for material
        GHG factors/intensities. Similar to PERT method.
        d_object = d_object from fit_bounded_distribution function.
        """
        fitted_realizations = d_object.rvs(*params, size=size)

        return fitted_realizations
    





# Call class
#simulation = MonteCarlo()
