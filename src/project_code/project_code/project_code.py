#!/usr/bin/env python

"""project_code.py: Main file of the project."""

__author__ = "Thomas SELECK, Alexis ROSUEL, Gaston BIZEL"
__credits__ = ["Thomas SELECK", "Alexis ROSUEL", "Gaston BIZEL"]
__license__ = "GPL"
__version__ = "1.0.0"
__status__ = "Production"

from numpy.random import binomial
from numpy.random import uniform
from numpy.random import beta
from numpy.random import chisquare
from numpy.random import normal
from scipy import stats
import numpy as np
import seaborn as sns
import time

def BasicAlgorithmLinkageExample(iterations, m):
    """
    This function implements the basic algorithm presented in the paper (Section 2). 
    It's basically an EM algorithm to compute the posterior distribution of a parameter theta.

    Parameters
    ----------
    iterations : positive integer
            This number is the number of iterations we want for our algorithm.

    m : positive integer
            This represents the number of sample we'll use.

    Returns
    -------
    theta : float
            This is the posterior estimation of theta we've computed.
    """

    y = [125, 18, 20, 34]
    #y = [13, 2, 2, 3]
    #y = [14, 0, 1, 5]

    # Step 1: Generate a sample z_1, ...z_m from the current approximation of theta
    ## Step 1.1. Generate theta from g_i(theta)
    g_i = uniform(size = m)

    for i in range(iterations):
        ## Step 1.2. Generate z from p(z|phi, y) where phi is the value obtained in Step 1.1.
        z = binomial(y[0], g_i / (g_i + 2), m)

        # Step 2: Update the current approximation of p(theta|y)
        nu_1 = z + y[3] + 1
        nu_2 = y[1] + y[2] + 1

        # Should be:
        #g_i = np.array([np.mean(stats.beta.pdf(g_i[i], nu_1, nu_2)) for i in range(m)])
        # but doesn't work
        g_i = np.array([np.mean(beta(nu_1[i], nu_2, m)) for i in range(m)])

    # Compute the true posterior distribution
    truePosterior = (((2 + g_i) ** y[0]) * ((1 - g_i) ** (y[1] + y[2])) * (g_i ** y[3]))
    # Scale the true posterior distribution
    truePosterior *= (np.mean(g_i) / np.mean(truePosterior))
    
    sns.kdeplot(g_i)
    sns.kdeplot(truePosterior)
    sns.kdeplot(beta(nu_1, nu_2, m))
    sns.plt.show()

    return g_i

def BasicAlgorithmMultivariateCovarianceMatrix(iterations, m):
    """
    This function implements the basic algorithm presented in the paper (Section 2). 
    It's basically an EM algorithm to compute the posterior distribution of a parameter theta.

    Parameters
    ----------
    iterations : positive integer
            This number is the number of iterations we want for our algorithm.

    m : positive integer
            This represents the number of sample we'll use.

    Returns
    -------
    theta : float
            This is the posterior estimation of theta we've computed.
    """

    # x1 and x2 contain the original censored data while x contains both, duplicated m times
    x1 = np.array([1, 1, -1, -1, 2, 2, -2, -2, np.nan, np.nan, np.nan, np.nan])
    x2 = np.array([1, -1, 1, -1, np.nan, np.nan, np.nan, np.nan, 2, 2, -2, 2])
    x = np.array([[x1.astype(np.float), x2.astype(np.float)] for i in range(m)])

    # We initialize the the parameters in the same way of the paper
    rho = uniform(-1, 1, m)
    sigma1 = np.sqrt(chisquare(7, m))
    sigma2 = np.sqrt(chisquare(7, m))
    Sigma = np.array([[[sigma1[i] ** 2, rho[i] * sigma1[i] * sigma2[i]], [rho[i] * sigma1[i] * sigma2[i], sigma2[i] ** 2]] for i in range(m)])

    for iter in range(iterations):
        # Step 1: We impute the missing data by drawing samples from a normal distribution. Its variance depends on which series have missing values.
        for obs_idx in range(x.shape[2]):
            if not np.isnan(x1[obs_idx]) and np.isnan(x2[obs_idx]):
                for i in range(m):
                    x[i, 1, obs_idx] = normal(rho[i] * (sigma2[i] / sigma1[i]) * x1[obs_idx], (sigma2[i] ** 2) * (1 - (rho[i] ** 2)))
            elif not np.isnan(x2[obs_idx]) and np.isnan(x1[obs_idx]):
                for i in range(m):
                    x[i, 0, obs_idx] = normal(rho[i] * (sigma1[i] / sigma2[i]) * x2[obs_idx], (sigma1[i] ** 2) * (1 - (rho[i] ** 2)))

        # Compute the covariance matrix
        covarianceMatrix = np.array([np.cov(x[i]) / (np.std(x[i, 0]) * np.std(x[i, 1])) for i in range(m)])

        # Step 2: Update the current approximation of p(Sigma|y)
        ## Generate m observations from the mixture of inverted Wishart distribution
        Sigma = np.array([stats.invwishart.rvs(df = 3, scale = covarianceMatrix[i]) for i in range(m)])

        ## We update the values of sigma1, sigma2 and rho
        sigma1 = np.sqrt(Sigma[:, 0, 0])
        sigma2 = np.sqrt(Sigma[:, 1, 1])
        ### Compute the associated correlation coefficient for each observation
        rho = Sigma[:, 1, 0] / np.sqrt(Sigma[:, 0, 0] * Sigma[:, 1, 1])

    # Compute the true posterior distribution
    truePosterior = ((1 - (rho ** 2)) ** 4.5) / ((1.25 - (rho ** 2)) ** 8)
    # Scale the true posterior distribution
    truePosterior *= 3

    sns.kdeplot(rho)
    sns.kdeplot(truePosterior)
    sns.plt.show()

    return rho, truePosterior
    

if __name__ == "__main__":
    # Start the timer
    startTime = time.time()

    iterations = 10
    m = 6400
    #res = BasicAlgorithmLinkageExample(iterations, m)
    iterations = 10
    m = 6400
    rho, truePosterior = BasicAlgorithmMultivariateCovarianceMatrix(iterations, m)

    # Stop the timer and print the exectution time
    print("Exec: --- %s seconds ---" % (time.time() - startTime))