
import pandas as pd
import numpy as np
import math
from __future__ import division
from scipy import stats



def z_score(confidence_interval):
    return stats.norm.ppf(1 - (1-confidence_interval)/2)

def z_value(
    ctrl_total_counts, 
    ctrl_positive, 
    experiment_total_counts, 
    experiment_positive, 
    z_score
):
    
    ctrl_p_hat = ctrl_positive / ctrl_total_counts
    new_p_hat = experiment_positive / experiment_total_counts
    p_hat = (ctrl_positive + experiment_positive)/(ctrl_total_counts + experiment_total_counts)
    return (ctrl_p_hat - new_p_hat) / math.sqrt(
        p_hat*(1-p_hat)*(1/ctrl_total_counts + 1/experiment_total_counts)
    )

def standard_errors(z_score, total_counts, positive):
    return z_score*math.sqrt(positive/total_counts*(1 - positive/total_counts)/total_counts)


def p_value(Z_value):
    return stats.norm.sf(abs(Z_value))


def experiment_duration(
    ctrl_total_counts,
    ctrl_positive,
    experiment_total_counts,
    experiment_positive, 
    confidence_interval=.95,
    coef=40
):
    P_values = []
    for x in range(1, coef):
        Z_value = z_value(
            ctrl_total_counts*x, 
            ctrl_positive*x, 
            experiment_total_counts*x, 
            experiment_positive*x, 
            z_score(confidence_interval)
        )
        P_values.append(p_value(Z_value))

    plt.plot(experiment_positive*np.array(range(1, 40)), P_values, label='P value at incremented Conversions')
    title = 'P value as a function of Conversion\n Counts at {}% conversion'.format(
        experiment_positive/experiment_total_counts
    )
    plt.title(title)
    label = '{} confidence interval boundary'.format(1-confidence_interval)
    plt.plot(np.full(225, (1-confidence_interval)), label=label)
    plt.xlabel('Number of Conversions')
    plt.legend()
    plt.xlim((-5, 225))
    plt.xticks(np.arange(0, 225, 25))
    ylabel = 'P value for {}% Confidence interval'.format(confidence_interval)
    plt.ylabel(ylabel)