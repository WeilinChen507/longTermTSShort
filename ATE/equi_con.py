import sys
from utils.ate_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
import numpy
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor


def conditionalEquiConfoundingBiasEstimator(X, S, Y, T, G, regressionType='kernelRidge', estimator='naive'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

    S, T, G = S[:, None], T[:, None], G[:, None]

    muSE = regression1D(np.concatenate((XE, TE), 1), SE, type=regressionType)
    muSO = regression1D(np.concatenate((XO, TO), 1), SO, type=regressionType)
    muYO = regression1D(np.concatenate((XO, TO), 1), YO, type=regressionType)
    exE = e_x_estimator(XE, TE)
    exO = e_x_estimator(XO, TO)
    exG = e_x_estimator(X, G)  # RCT G=1, OBS G=0

    # used prediction
    exE_prob = exE.predict_proba(XE)[:, 1][:, None]
    exO_prob = exO.predict_proba(XO)[:, 1][:, None]
    exG_XE_prob = exG.predict_proba(XE)[:, 1][:, None]  # prob of G=1, i.e., G=RCT
    muSE_XE = muSE.predict(np.concatenate((XE, TE), 1))
    muSO_XO = muSO.predict(np.concatenate((XO, TO), 1))
    muYO_XO = muYO.predict(np.concatenate((XO, TO), 1))
    muYO_1XO = muYO.predict(np.concatenate((XO, np.ones_like(TO)), 1))
    muYO_0XO = muYO.predict(np.concatenate((XO, np.zeros_like(TO)), 1))
    muSE_1XO = muSE.predict(np.concatenate((XO, np.ones_like(TO)), 1))
    muSE_0XO = muSE.predict(np.concatenate((XO, np.zeros_like(TO)), 1))
    muSO_1XO = muSO.predict(np.concatenate((XO, np.ones_like(TO)), 1))
    muSO_0XO = muSO.predict(np.concatenate((XO, np.zeros_like(TO)), 1))


    Osize, Esize = TO.shape[0], TE.shape[0]
    if estimator == 'IF-based':

        tau = np.sum(np.power(-1, 1 - TE) / (1 - TE + np.power(-1, 1 - TE) * exE_prob) * (Osize + Esize) / Osize *
                      (SE - muSE_XE[:, None]) * (1 / exG_XE_prob - 1)
                      ) \
              + np.sum((Osize + Esize) / Osize * (
                          np.power(-1, 1 - TO) / (1 - TO + np.power(-1, 1 - TO) * exO_prob) * (YO - muYO_XO[:, None] - SO + muSO_XO[:, None])
                          + muYO_1XO[:, None] - muYO_0XO[:, None] + muSE_1XO[:, None] - muSE_0XO[:, None] + muSO_0XO[:, None] - muSO_1XO[:, None]
                                                )
                      )
        tau = tau / G.shape[0]
    elif estimator == 'naive':
        tau = np.mean( muYO_1XO - muYO_0XO + muSE_1XO - muSE_0XO + muSO_0XO - muSO_1XO )

    return tau


def conditionalEquiConfoundingBiasEstimator_Tlearner(X, S, Y, T, G, regressionType='kernelRidge', estimator = 'naive'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

    S, T, G = S[:, None], T[:, None], G[:, None]

    muSE1 = regression1D(XE[TE[:,0]==1,:], SE[TE[:,0]==1,:], type=regressionType)
    muSE0 = regression1D(XE[TE[:,0]==0,:], SE[TE[:,0]==0,:], type=regressionType)
    muSO1 = regression1D(XO[TO[:,0]==1,:], SO[TO[:,0]==1,:], type=regressionType)
    muSO0 = regression1D(XO[TO[:,0]==0,:], SO[TO[:,0]==0,:], type=regressionType)
    muYO1 = regression1D(XO[TO[:,0]==1,:], YO[TO[:,0]==1,:], type=regressionType)
    muYO0 = regression1D(XO[TO[:,0]==0,:], YO[TO[:,0]==0,:], type=regressionType)
    exE = e_x_estimator(XE, TE)
    exO = e_x_estimator(XO, TO)
    exG = e_x_estimator(X, G)  # RCT G=1, OBS G=0

    # used prediction
    exE_prob = exE.predict_proba(XE)[:, 1][:, None]
    exO_prob = exO.predict_proba(XO)[:, 1][:, None]
    exG_XE_prob = exG.predict_proba(XE)[:, 1][:, None]  # prob of G=1, i.e., G=RCT

    muSE_1XE = muSE1.predict(XE)
    muSE_0XE = muSE0.predict(XE)
    muSE_XE = np.where(TE[:,0]==1, muSE_1XE, muSE_0XE)

    muYO_1XO = muYO1.predict(XO)
    muYO_0XO = muYO0.predict(XO)
    muSE_1XO = muSE1.predict(XO)
    muSE_0XO = muSE0.predict(XO)
    muSO_1XO = muSO1.predict(XO)
    muSO_0XO = muSO0.predict(XO)

    muSO_XO = np.where(TO[:,0]==1, muSO_1XO, muSO_0XO)
    muYO_XO = np.where(TO[:,0]==1, muYO_1XO, muYO_0XO)


    Osize, Esize = TO.shape[0], TE.shape[0]

    # IF-based
    if estimator == 'IF-based':
        tau = np.sum(np.power(-1, 1 - TE) / (1 - TE + np.power(-1, 1 - TE) * exE_prob) * (Osize + Esize) / Osize *
                      (SE - muSE_XE[:, None]) * (1 / exG_XE_prob - 1)
                      ) \
              + np.sum((Osize + Esize) / Osize * (
                          np.power(-1, 1 - TO) / (1 - TO + np.power(-1, 1 - TO) * exO_prob) * (YO - muYO_XO[:, None] - SO + muSO_XO[:, None])
                          + muYO_1XO[:, None] - muYO_0XO[:, None] + muSE_1XO[:, None] - muSE_0XO[:, None] + muSO_0XO[:, None] - muSO_1XO[:, None]
                                                )
                      )
        tau = tau / G.shape[0]
    elif estimator == 'naive':
        tau = np.mean( muYO_1XO - muYO_0XO + muSE_1XO - muSE_0XO + muSO_0XO - muSO_1XO )

    return tau


def conditionalEquiConfoundingBiasEstimator_Tlearne_verifyMR(X, S, Y, T, G, regressionType='kernelRidge', case=1):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

    S, T, G = S[:, None], T[:, None], G[:, None]

    muSE1 = regression1D(XE[TE[:,0]==1,:], SE[TE[:,0]==1,:], type=regressionType)
    muSE0 = regression1D(XE[TE[:,0]==0,:], SE[TE[:,0]==0,:], type=regressionType)
    muSO1 = regression1D(XO[TO[:,0]==1,:], SO[TO[:,0]==1,:], type=regressionType)
    muSO0 = regression1D(XO[TO[:,0]==0,:], SO[TO[:,0]==0,:], type=regressionType)
    muYO1 = regression1D(XO[TO[:,0]==1,:], YO[TO[:,0]==1,:], type=regressionType)
    muYO0 = regression1D(XO[TO[:,0]==0,:], YO[TO[:,0]==0,:], type=regressionType)
    exE = e_x_estimator(XE, TE)
    exO = e_x_estimator(XO, TO)
    exG = e_x_estimator(X, G)  # RCT G=1, OBS G=0

    # used prediction
    exE_prob = exE.predict_proba(XE)[:, 1][:, None]
    exO_prob = exO.predict_proba(XO)[:, 1][:, None]
    exG_XE_prob = exG.predict_proba(XE)[:, 1][:, None]  # prob of G=1, i.e., G=RCT

    muSE_1XE = muSE1.predict(XE)
    muSE_0XE = muSE0.predict(XE)
    muSE_XE = np.where(TE[:,0]==1, muSE_1XE, muSE_0XE)

    muYO_1XO = muYO1.predict(XO)
    muYO_0XO = muYO0.predict(XO)
    muSE_1XO = muSE1.predict(XO)
    muSE_0XO = muSE0.predict(XO)
    muSO_1XO = muSO1.predict(XO)
    muSO_0XO = muSO0.predict(XO)

    muSO_XO = np.where(TO[:,0]==1, muSO_1XO, muSO_0XO)
    muYO_XO = np.where(TO[:,0]==1, muYO_1XO, muYO_0XO)

    Osize, Esize = TO.shape[0], TE.shape[0]

    if case == 1:
        exE_prob = 0.5
        exO_prob = 0.5
        exG_XE_prob = 0.5
    elif case == 2:
        muYO_XO = 2. * muYO_XO
        muSO_XO = 2. * muSO_XO
        muYO_1XO = 2. * muYO_1XO
        muYO_0XO = 2. * muYO_0XO
        muSE_1XO = 2. * muSE_1XO
        muSE_0XO = 2. * muSE_0XO
        muSO_0XO = 2. * muSO_0XO
        muSO_1XO = 2. * muSO_1XO
    elif case == 3:
        muSE_1XO = 2. * muSE_1XO
        muSE_0XO = 2. * muSE_0XO
        exO_prob =  0.5
    elif case == 4:
        exO_prob =  0.5
        exG_XE_prob =  0.5
        muYO_XO = 2. * muYO_XO
        muSO_XO = 2. * muSO_XO
        muYO_1XO = 2. * muYO_1XO
        muYO_0XO = 2. * muYO_0XO
        muSO_0XO = 2. * muSO_0XO
        muSO_1XO = 2. * muSO_1XO

    tau = np.sum(np.power(-1, 1 - TE) / (1 - TE + np.power(-1, 1 - TE) * exE_prob) * (Osize + Esize) / Osize *
                      (SE - muSE_XE[:, None]) * (1 / exG_XE_prob - 1)
                 ) \
              + np.sum((Osize + Esize) / Osize * (
                          np.power(-1, 1 - TO) / (1 - TO + np.power(-1, 1 - TO) * exO_prob) * (YO - muYO_XO[:, None] - SO + muSO_XO[:, None])
                          + muYO_1XO[:, None] - muYO_0XO[:, None] + muSE_1XO[:, None] - muSE_0XO[:, None] + muSO_0XO[:, None] - muSO_1XO[:, None]
                                                )
                      )

    tau = tau / G.shape[0]
    return tau



def test():
    Osize, Esize = 3000, 1000
    # XO XE can be slightly different, but UO and UE should be totally same, i.e., X->G but not U->G
    XO, UO = np.random.multivariate_normal(mean=[2, 1], cov=[[1, 0], [0, 1]], size=Osize), \
        np.random.multivariate_normal(mean=[2, 1], cov=[[1, 1], [1, 1]], size=Osize)
    XE, UE = np.random.multivariate_normal(mean=[1, 2], cov=[[1, 0], [0, 1]], size=Esize), \
        np.random.multivariate_normal(mean=[2, 1], cov=[[1, 1], [1, 1]], size=Esize)

    TO = np.random.binomial(n=1, size=Osize, p=(1 / (1 + np.exp(-np.mean(np.concatenate((XO, UO), 1), axis=1)))))
    TE = np.random.binomial(n=1, size=Esize, p=(1 / (1 + np.exp(-np.mean(XE, axis=1)))))

    # rule out extreme data
    indexO = np.logical_and(((1 / (1 + np.exp(-np.mean(np.concatenate((XO, UO), 1), axis=1)))) < 0.95),
                            ((1 / (1 + np.exp(-np.mean(np.concatenate((XO, UO), 1), axis=1)))) > 0.05))
    indexE = np.logical_and(((1 / (1 + np.exp(-np.mean(XE, axis=1)))) < 0.95),
                            ((1 / (1 + np.exp(-np.mean(XE, axis=1)))) > 0.05))
    XO, UO = XO[indexO, :], UO[indexO, :]
    XE, UE = XE[indexE, :], UE[indexE, :]
    TO, TE = TO[indexO], TE[indexE]
    Osize, Esize = TO.shape[0], TE.shape[0]

    print('sizeO,E=' + str(Osize) + str(',') + str(Esize))

    noiseSO, noiseSE = np.random.randn(Osize) * 0.3, np.random.randn(Esize) * 0.3
    SO_1 = 2 * np.mean(XO, axis=1) + np.mean(UO, axis=1) + 3 * 1 * np.mean(XO, axis=1) + noiseSO
    SO_0 = 2 * np.mean(XO, axis=1) + np.mean(UO, axis=1) + 3 * 0 * np.mean(XO, axis=1) + noiseSO
    SE_1 = 2 * np.mean(XE, axis=1) + np.mean(UE, axis=1) + 3 * 1 * np.mean(XE, axis=1) + noiseSE
    SE_0 = 2 * np.mean(XE, axis=1) + np.mean(UE, axis=1) + 3 * 0 * np.mean(XE, axis=1) + noiseSE

    SO = np.where(TO == 1, SO_1, SO_0)
    SE = np.where(TE == 1, SE_1, SE_0)

    noiseO, noiseE = np.random.randn(Osize) * 0.3, np.random.randn(Esize) * 0.3
    YO_1 = np.mean(XO, axis=1) - np.mean(UO, axis=1) + SO_1 + 2 * 1 * np.mean(XO, axis=1) + noiseO
    YO_0 = np.mean(XO, axis=1) - np.mean(UO, axis=1) + SO_0 + 2 * 0 * np.mean(XO, axis=1) + noiseO
    YE_1 = np.mean(XE, axis=1) - np.mean(UE, axis=1) + SE_1 + 2 * 1 * np.mean(XE, axis=1) + noiseE
    YE_0 = np.mean(XE, axis=1) - np.mean(UE, axis=1) + SE_0 + 2 * 0 * np.mean(XE, axis=1) + noiseE

    YO = np.where(TO == 1, YO_1, YO_0)
    YE = np.where(TE == 1, YE_1, YE_0)

    iteO, iteE = YO_1 - YO_0, YE_1 - YE_0

    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
        np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    # print('validata con equ bias')
    # equ1 = np.mean(SO_1[TO==0]) - np.mean(SO_1[TO==1]) - np.mean(YO_1[TO==0]) + np.mean(YO_1[TO==1])
    # print(equ1)

    print('ground truth: ' + str(np.mean(iteO)))
    print('ipw using obs data: ' + str(ipw_estimator(XO, TO, YO)))
    print('ipw using exp data: ' + str(ipw_estimator(XE, TE, YE)))
    print('s_learner using exp data: ' + str(s_learner_estimator(XE, TE, YE, type='randomForestRegressor')))
    print('conditionalEquiConfoundingBiasEstimator: ' + str(conditionalEquiConfoundingBiasEstimator(X, S, Y, T, G, regressionType='best', estimator = 'IF-based')))
    print('conditionalEquiConfoundingBiasEstimator_T: ' + str(conditionalEquiConfoundingBiasEstimator_Tlearner(X, S, Y, T, G, regressionType='best',estimator = 'IF-based')))
    print('conditionalEquiConfoundingBiasEstimator_T_case1: ' + str(conditionalEquiConfoundingBiasEstimator_Tlearne_verifyMR(X, S, Y, T, G, regressionType='best',case = 1)))
    print('conditionalEquiConfoundingBiasEstimator_T_case2: ' + str(conditionalEquiConfoundingBiasEstimator_Tlearne_verifyMR(X, S, Y, T, G, regressionType='best',case = 2)))
    print('conditionalEquiConfoundingBiasEstimator_T_case3: ' + str(conditionalEquiConfoundingBiasEstimator_Tlearne_verifyMR(X, S, Y, T, G, regressionType='best',case = 3)))
    print('conditionalEquiConfoundingBiasEstimator_T_case4: ' + str(conditionalEquiConfoundingBiasEstimator_Tlearne_verifyMR(X, S, Y, T, G, regressionType='best',case = 4)))



import warnings
warnings.simplefilter('ignore')
test()
