from scipy import stats
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import warnings



def shapiro_wilk(sample, alpha):
    '''
    Calculates if the sample follows a normal distribution.
    H0: X ~ N(miu, sigma^2).

    Parameters:
    - sample (list of float): List of gene expressions.
    - alpha (int): 1 - confidence interval.

    Returns:
    - p (float): The p-value of the shapiro-wilk test.
    '''
    # data has zero range, might not be accurate, return 1 for non-significance
    if np.ptp(sample) == 0:
        return 1
    stat, p = stats.shapiro(sample)
    return p



def t_test(sample1, sample2, alpha):
    '''
    Calculates the t-test for the means of two independent samples of scores.
    This is a test with a null hypothesis that 2 independent samples have identical mean values.
    We assume that the populations have identical variances.
    H0: miu1 = miu2.
    
    Parameters:
    - sample1 (list of float): List 1 of gene expressions.
    - sample2 (list of float): List 2 of gene expressions.
    - alpha (int): 1 - confidence interval.

    Returns:
    - p (float): The p-value of the t-test.
    '''
    stat, p = ttest_ind(sample1, sample2)
    return p



def wilcoxon_signed_rank(sample1, sample2, alpha):
    '''
    Calculates the wilcoxon signed-rank test for two different samples.
    Samples do not follow a normal distribution.
    H0: miu1 = miu2.

    Parameters:
    - sample1 (list of float): List 1 of gene expressions.
    - sample2 (list of float): List 2 of gene expressions.
    - alpha (int): 1 - confidence interval.

    Returns:
    - p (float): The p-value of the wilcoxon signed-rank test.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = stats.wilcoxon(sample1, sample2)
    return p



def anova(samples, alpha):
    '''
    Calculates ANOVA test for 2 or more samples.

    Parameters:
    - samples (list of lists of floats): List of lists of gene expressions.
    - alpha (int): 1 - confidence interval.

    Returns:
    - p (float): The p-value of the ANOVA test.
    '''
    stat, p = f_oneway(*samples)
    return p



def statistical_study(sample1, sample2):
    '''
    Receiving two samples, chooses to use t-test or wilcoxon, depending on result of the shapiro.

    Parameters:
    - sample1 (list of float): List 1 of gene expressions.
    - sample2 (list of float): List 2 of gene expressions.
    - alpha (int): 1 - confidence interval.

    Returns:
    - p (float): The p-value of the wilcoxon signed-rank test.
    '''
    p_value = 1
    alpha = 0.05
    p1 = shapiro_wilk(sample1, alpha)
    p2 = shapiro_wilk(sample2, alpha)
    if p1 < alpha or p2 < alpha:
        # reject null hypothesis, do not follow normal distribution
        min_length = min(len(sample1), len(sample2))
        # do not work for similar samples
        if not np.allclose(sample1[:min_length], sample2[:min_length]):
           p_value = wilcoxon_signed_rank(sample1[:min_length], sample2[:min_length], alpha)
    elif p1 >= alpha and p2 >= alpha:
        # accept the null hypothesis, follow normal distribution
        p_value = t_test(sample1, sample2, alpha)
    return p_value



if __name__ == "__main__":
    print("Run the app.py")