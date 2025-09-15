
import sys

import numpy

sys.path.append('..')

import numpy as np

def RMSE_ATE(tau_true, tau_test):
    if tau_true.shape != tau_test.shape:
        tau_test = tau_test[:,None]
        if tau_true.shape == tau_test.shape:
            return np.abs((np.mean(tau_true) - np.mean(tau_test)))
        print(tau_true.shape, tau_test.shape)
    assert tau_true.shape == tau_test.shape
    return np.abs((np.mean(tau_true) - np.mean(tau_test)))
    # return np.abs((np.mean(tau_true) - np.mean(tau_test))/np.mean(tau_true))


def PEHE_ITE(tau_true, tau_test):
    if tau_true.shape != tau_test.shape:
        tau_test = tau_test[:,None]
        if tau_true.shape == tau_test.shape:
            return np.sqrt(np.mean(np.power((tau_true - tau_test), 2)))
        print(tau_true.shape, tau_test.shape)
    assert tau_true.shape == tau_test.shape
    return np.sqrt(np.mean(np.power((tau_true - tau_test),2)))
    # return np.sqrt(np.mean(np.power((tau_true - tau_test)/tau_true,2)))

