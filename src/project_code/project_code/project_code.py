#!/usr/bin/env python

"""project_code.py: Main file of the project."""

__author__ = "Thomas SELECK, Alexis ROSUEL, Gaston BIZEL"
__credits__ = ["Thomas SELECK", "Alexis ROSUEL", "Gaston BIZEL"]
__license__ = "GPL"
__version__ = "1.0.0"
__status__ = "Production"

from numpy.random import binomial
from numpy.random import uniform
from numpy.random import chisquare
from numpy.random import normal
from numpy.random import randint
from scipy import stats
import numpy as np
import seaborn as sns
import time
from operator import itemgetter
from scipy.stats import gaussian_kde

def BasicAlgorithmLinkageExample(iterations, m, y, displayPlot):
    """
    This function implements the basic algorithm presented in the paper (Section 2). 
    It's basically an EM algorithm to compute the posterior distribution of a parameter theta.

    Parameters
    ----------
    iterations : positive integer
            This number is the number of iterations we want for our algorithm.

    m : positive integer
            This represents the number of sample we'll use.

    y : list
            This represents the observed data.

    displayPlot : boolean
            If true, the plot is displayed. Otherwise, it is save as a png file in the current directory.

    Returns
    -------
    theta : float
            This is the posterior estimation of theta we've computed.
    """

    print("Computing posterior distribution for the Genetic Linkage example with y = " + str(y) + " ...")

    # Step 1: Generate a sample z_1, ...z_m from the current approximation of theta
    ## Step 1.1. Generate theta from g_i(theta)
    theta = uniform(size = m)

    for i in range(iterations):
        ## Step 1.2. Generate z from p(z|phi, y) where phi is the value obtained in Step 1.1.
        z = binomial(y[0], theta / (theta + 2), m)

        # Step 2: Update the current approximation of p(theta|y)
        nu_1 = z + y[3] + 1
        nu_2 = y[1] + y[2] + 1

        # Select a distribution from the mixture of beta distributions and do it m times to get m samples
        idx = randint(0, m, size = m)

        # Draw a sample for theta from the mixture of beta and do it m times
        theta = stats.beta.rvs(nu_1[idx], nu_2, size = m)

    # Compute the true posterior distribution
    x = uniform(size = m) # Silent variable to plot the true posterior.
    truePosterior = (((2 + x) ** y[0]) * ((1 - x) ** (y[1] + y[2])) * (x ** y[3]))
    
    # Scale the true posterior distribution: Quick and dirty way to do this
    truePosterior *= np.max(gaussian_kde(theta).pdf(x)) / np.max(truePosterior)

    x, truePosterior = [list(x) for x in zip(*sorted(zip(x, truePosterior), key=itemgetter(0)))]
    
    sns.distplot(theta, hist = False, kde = True, color = "b").set(xlim = (0, 1))
    sns.distplot(stats.beta.rvs(nu_1, nu_2, size = m), hist = False, kde = True, color = "r").set(xlim = (0, 1), xlabel = "Theta", ylabel = "Density")
    sns.plt.plot(x, truePosterior, color = "g")
    sns.plt.title("Genetic linkage example: Posterior distribution of theta with y = " + str(y))

    if displayPlot:
        sns.plt.show()
        sns.plt.cla()
    else:
        sns.plt.savefig("genetic_linkage_example_" + "_".join([str(i) for i in y]) + ".png", dpi = 150)
        sns.plt.cla()

    return theta

def DirichletSamplingProcessLinkageExample(iterations, m, y, epsilon, displayPlot):
    """
    This function implements the Dirichlet sampling process presented in the paper (Section 4). 
    It's basically an EM algorithm to compute the posterior distribution of a parameter theta. We use
    the Dirichlet sampling when the sampling of theta from p(theta|x) is not simple.

    Parameters
    ----------
    iterations : positive integer
            This number is the number of iterations we want for our algorithm.

    m : positive integer
            This represents the number of sample we'll use.

    y : list
            This represents the augmented data.

    displayPlot : boolean
            If true, the plot is displayed. Otherwise, it is save as a png file in the current directory.

    Returns
    -------
    theta : float
            This is the posterior estimation of theta we've computed.
    """

    print("Computing posterior distribution for the Genetic Linkage example with y = " + str(y) + " using DSP...")

    # Step 1: Generate a sample z_1, ...z_m from the current approximation of theta
    ## Step 1.1. Generate theta from g_i(theta)
    theta = uniform(size = m)

    for i in range(iterations):
        ## Step 1.2. Generate z from p(z|phi, y) where phi is the value obtained in Step 1.1.
        z = binomial(y[0], theta / (theta + 2), m)

        # Step 2: Update the current approximation of p(theta|y)
        ## Sample observations from Dirichlet distribution
        dirichletSamples = []
        for i in range(m):
            augmented_data = [z[i] + 1, y[1] + 1, y[2] + 1, y[3] + 1] # We add 1 to ensure each number is greater than 0
            dirichletSamples.append(stats.dirichlet.rvs(augmented_data)[0] / 2)
        
        dirichletSamples = np.array(dirichletSamples)

        ## Compute theta_hat
        theta_hat_array = []
        for sample in dirichletSamples:
            theta_hat = 2 * (sample[0] + sample[3])
            p_hat = np.array([theta_hat / 4, 1 / 4 - theta_hat / 4, 1 / 4 - theta_hat / 4, theta_hat / 4])

            if np.sqrt(np.sum((np.array(sample) - p_hat) ** 2)) < epsilon:
                theta_hat_array.append(theta_hat)
                
        m = len(theta_hat_array)
        theta_hat_array = np.array(theta_hat_array)
        theta = theta_hat_array

    # Compute the true posterior distribution
    x = uniform(size = m) # Silent variable to plot the true posterior.
    truePosterior = (((2 + x) ** y[0]) * ((1 - x) ** (y[1] + y[2])) * (x ** y[3]))
    
    # Scale the true posterior distribution: Quick and dirty way to do this
    truePosterior *= np.max(gaussian_kde(theta).pdf(x)) / np.max(truePosterior)

    x, truePosterior = [list(x) for x in zip(*sorted(zip(x, truePosterior), key=itemgetter(0)))]
    
    sns.distplot(theta, hist = False, kde = True, color = "b").set(xlim = (0, 1))
    sns.plt.plot(x, truePosterior, color = "g")
    sns.plt.title("Genetic linkage example: Posterior distribution of theta with y = " + str(y) + " using DSP")

    if displayPlot:
        sns.plt.show()
        sns.plt.cla()
    else:
        sns.plt.savefig("genetic_linkage_example_DSP" + "_".join([str(i) for i in y]) + ".png", dpi = 150)
        sns.plt.cla()

    return theta

def BasicAlgorithmMultivariateCovarianceMatrix(iterations, m, displayPlot):
    """
    This function implements the basic algorithm presented in the paper (Section 2). 
    It's basically an EM algorithm to compute the posterior distribution of a parameter theta.

    Parameters
    ----------
    iterations : positive integer
            This number is the number of iterations we want for our algorithm.

    m : positive integer
            This represents the number of sample we'll use.

    displayPlot : boolean
            If true, the plot is displayed. Otherwise, it is save as a png file in the current directory.

    Returns
    -------
    Sigma : numpy array
            This is the posterior estimation of the covariance matrix we've computed.
    """

    print("Computing posterior distribution for the functionals of the multivariate normal covariance matrix example...")

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
        covarianceMatrix = np.array([np.cov(x[i]) / x1.shape[0] for i in range(m)])

        # Step 2: Update the current approximation of p(Sigma|y)
        # Select a distribution from the mixture of inverted Wishart distributions and do it m times to get m samples
        idx = randint(0, m, size = m)

        # Draw a sample for Sigma from the mixture of inverted Wishart distributions and do it m times
        Sigma = np.array([stats.invwishart.rvs(df = 4, scale = covarianceMatrix[idx[i]]) for i in range(m)])

        ## We update the values of sigma1, sigma2 and rho
        sigma1 = np.sqrt(Sigma[:, 0, 0])
        sigma2 = np.sqrt(Sigma[:, 1, 1])
        ### Compute the associated correlation coefficient for each observation
        rho = Sigma[:, 1, 0] / (sigma1 * sigma2)

    # Compute the true posterior distribution
    x = uniform(low = -1, size = m)
    truePosterior = ((1 - (x ** 2)) ** 4.5) / ((1.25 - (x ** 2)) ** 8)
    
    # Scale the true posterior distribution: Quick and dirty way to do this
    truePosterior *= np.max(gaussian_kde(rho).pdf(x)) / np.max(truePosterior)

    x, truePosterior = [list(x) for x in zip(*sorted(zip(x, truePosterior), key=itemgetter(0)))]
    
    sns.distplot(rho, hist = False, kde = True, color = "b").set(xlim = (-1.5, 1.5), xlabel = "Rho", ylabel = "Density")
    sns.plt.plot(x, truePosterior, color = "g")
    sns.plt.title("Functionals of the multivariate normal covariance matrix: Posterior density of the correlation coefficient")

    if displayPlot:
        sns.plt.show()
        sns.plt.cla()
    else:
        sns.plt.savefig("multivariate_normal_covariance_matrix_example.png", dpi = 150)
        sns.plt.cla()

    return Sigma
    

if __name__ == "__main__":
    # Set the PRNG's seed
    np.random.seed(45)

    # Start the timer
    startTime = time.time()

    # Do you want to see plots or save them to files?
    displayPlots = True

    # For linkage example
    iterations = 100
    m = 1600
    data = [[125, 18, 20, 34], [13, 2, 2, 3], [14, 0, 1, 5]]

    for y in data:
        res = BasicAlgorithmLinkageExample(iterations, m, y, displayPlots)

    # For multivariate covariance matrix
    iterations = 15
    m = 6400
    res = BasicAlgorithmMultivariateCovarianceMatrix(iterations, m, displayPlots)

    # For linkage example
    iterations = 20
    m = 10000
    y = [3, 2, 2, 3]
    epsilon = 0.20

    res = DirichletSamplingProcessLinkageExample(iterations, m, y, epsilon, displayPlots)

    # Stop the timer and print the exectution time
    print("Exec: --- %s seconds ---" % (time.time() - startTime))