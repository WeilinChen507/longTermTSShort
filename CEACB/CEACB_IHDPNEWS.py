from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split

import random
from decimal import Decimal, ROUND_HALF_UP

def round_half_up(n):
    return int(Decimal(str(n)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))

def conditionalEquiConfoundingBiasEstimator_Tlearner(X, S, Y, T, G, regressionType='linear'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1], T[G == 1][:, None]

    # print(YO.shape)
    muSE1 = regression1D(XE[TE[:,0]==1,:], SE[TE[:,0]==1,:], type=regressionType)
    muSE0 = regression1D(XE[TE[:,0]==0,:], SE[TE[:,0]==0,:], type=regressionType)
    muSO1 = regression1D(XO[TO[:,0]==1,:], SO[TO[:,0]==1,:], type=regressionType)
    muSO0 = regression1D(XO[TO[:,0]==0,:], SO[TO[:,0]==0,:], type=regressionType)
    muYO1 = regression1D(XO[TO[:,0]==1,:], YO[TO[:,0]==1,:], type=regressionType)
    muYO0 = regression1D(XO[TO[:,0]==0,:], YO[TO[:,0]==0,:], type=regressionType)

    # used prediction
    muYO_1XO = muYO1.predict(XO)
    muYO_0XO = muYO0.predict(XO)
    muSE_1XO = muSE1.predict(XO)
    muSE_0XO = muSE0.predict(XO)
    muSO_1XO = muSO1.predict(XO)
    muSO_0XO = muSO0.predict(XO)
    tau = muYO_1XO - muYO_0XO + muSE_1XO - muSE_0XO + muSO_0XO - muSO_1XO

    return tau


def readNEWS(seed, t0):
    matrix = np.loadtxt('../datasets/LTEE_NEWS/Series_X_' + str(seed) + '.txt', delimiter=',')
     # = np.loadtxt('data/NEWS/Series_X_' + str(j) + '.txt', delimiter=',')
    N = matrix.shape[0]

    out_treat = np.loadtxt('../datasets/LTEE_NEWS/Series_y_' + str(seed) + '.txt', delimiter=',')
    ts = out_treat[:, 0]
    ts = np.reshape(ts, (N, 1))
    groundtruth_indi = np.loadtxt('../datasets/LTEE_NEWS/HLCE_groundtruth_' + str(seed) + '.txt')

    ys = np.concatenate((out_treat[:, 1:(t0 + 1)], out_treat[:, -1].reshape(N, 1)), axis=1)
    print(ys.shape)
    # matrix_rep = np.repeat(matrix[:, np.newaxis, :], t0, axis=1)
    X_train, X_test, y_train, y_test, t_train, t_test, ite_train, ite_test\
        = train_test_split(matrix, ys, ts, groundtruth_indi, test_size=0.2)


    SO, YO, TO = y_train[:, :-1], y_train[:, -1][:, None], np.squeeze(t_train)
    XO = X_train
    SE, YE, TE = y_test[:, :-1], y_test[:, -1][:, None], np.squeeze(t_test)
    XE = X_test
    iteO, iteE = ite_train, ite_test
    # print(SO.shape, YO.shape, TO.shape, XO.shape)
    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
        np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()
    # print(X.shape, T.shape, S.shape, Y.shape, G.shape)
    return X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE

def readIHDP(seed, t0):
    TY = np.loadtxt('../datasets/LTEE_IHDP/csv/ihdp_npci_' + str(seed) + '.csv', delimiter=',')
    matrix = TY[:, 5:]
    N = TY.shape[0]

    out_treat = np.loadtxt('../datasets/LTEE_IHDP/Series_y_' + str(seed) + '.txt', delimiter=',')
    ts = out_treat[:, 0]
    ts = np.reshape(ts, (N, 1))
    groundtruth_indi = np.loadtxt('../datasets/LTEE_IHDP/HLCE_groundtruth_' + str(seed) + '.txt')

    ys = np.concatenate((out_treat[:, 1:(t0 + 1)], out_treat[:, -1].reshape(N, 1)), axis=1)
    print(ys.shape)
    matrix_rep = np.repeat(matrix[:, np.newaxis, :], t0, axis=1)
    X_train, X_test, y_train, y_test, t_train, t_test, ite_train, ite_test\
        = train_test_split(matrix_rep, ys, ts, groundtruth_indi, test_size=0.2)


    SO, YO, TO = y_train[:, :-1], y_train[:, -1][:, None], np.squeeze(t_train)
    XO = X_train[:,0,:]
    SE, YE, TE = y_test[:, :-1], y_test[:, -1][:, None], np.squeeze(t_test)
    XE = X_test[:,0,:]
    iteO, iteE = ite_train, ite_test
    # print(SO.shape, YO.shape, TO.shape, XO.shape)
    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
        np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()
    # print(X.shape, T.shape, S.shape, Y.shape, G.shape)
    return X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE


def test(TIME=9, time=6, sizeE=2000, sizeO=4000, regressionType='linear', dataset='NEWS'):
    # from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds, dgp_ts_equ_conf_design_linear

    ite_s1, ate_s1 = [],[]  # using first s
    ite_smid, ate_smid = [],[] # using mid s
    ite_slast, ate_slast = [],[] # using last s
    ite_srand, ate_srand = [],[] # using random s

    for seed in range(1,11):
        from utils.ts_eq_dgp import set_all_seeds
        if dataset is 'NEWS':
            X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE = readNEWS(seed=seed, t0=50)
        else:
            X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE = readIHDP(seed=seed, t0=50)

        X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
            np.concatenate((YO, YE), 0), \
            np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

        iteO = iteO[:,None] # reshape from [size,] to [size,1]
        # print('ground truth ATE: ' + str(np.mean(iteO)))
        set_all_seeds(seed)
        index=0
        ite_est_s1 = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S[:, index][:,None], Y, T, G, regressionType=regressionType)
        ite_s1.append(PEHE_ITE(iteO, ite_est_s1))
        ate_s1.append(RMSE_ATE(iteO, ite_est_s1))

        set_all_seeds(seed)
        index = -1
        ite_est_slast = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S[:, index][:, None], Y, T, G, regressionType=regressionType)
        ite_slast.append(PEHE_ITE(iteO, ite_est_slast))
        ate_slast.append(RMSE_ATE(iteO, ite_est_slast))

        set_all_seeds(seed)
        index = round_half_up(time/2)-1
        ite_est_smid = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S[:, index][:, None], Y, T, G, regressionType=regressionType)
        ite_smid.append(PEHE_ITE(iteO, ite_est_smid))
        ate_smid.append(RMSE_ATE(iteO, ite_est_smid))

        set_all_seeds(seed)
        index = random.randint(0, S.shape[1]-1)
        ite_est_srand = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S[:, index][:, None], Y, T, G, regressionType=regressionType)
        ite_srand.append(PEHE_ITE(iteO, ite_est_srand))
        ate_srand.append(RMSE_ATE(iteO, ite_est_srand))

    # save = ite_s1,ate_s1,/ np.mean(ite_s1), np.mean(ate_s1), np.std(ite_s1), np.std(ate_s1)
    # np.savetxt('./t0_' + time.__str__()+ '_T_'+ TIME.__str__() + '.txt', save,
    #            fmt='%s')

    print('print s1: ', np.mean(ite_s1), np.mean(ate_s1), np.std(ite_s1), np.std(ate_s1))
    print('print smid: ', np.mean(ite_smid), np.mean(ate_smid), np.std(ite_smid), np.std(ate_smid))
    print('print slast: ', np.mean(ite_slast), np.mean(ate_slast), np.std(ite_slast), np.std(ate_slast))
    print('print srand: ', np.mean(ite_srand), np.mean(ate_srand), np.std(ite_srand), np.std(ate_srand))



import warnings, torch
warnings.simplefilter('ignore')
torch.set_default_dtype(torch.double)
torch.manual_seed(123)
np.random.seed(123)


test(time=50, regressionType='linear', dataset='IHDP')
# test(time=50, regressionType='linear', dataset='NEWS')
test(time=50, regressionType='randomForestRegressor', dataset='NEWS')

