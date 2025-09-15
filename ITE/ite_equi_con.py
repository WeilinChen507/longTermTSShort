from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np


def conditionalEquiConfoundingBiasEstimator(X, S, Y, T, G, regressionType='linear'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

    muSE = regression1D(np.concatenate((XE, TE), 1), SE, type=regressionType)
    muSO = regression1D(np.concatenate((XO, TO), 1), SO, type=regressionType)
    muYO = regression1D(np.concatenate((XO, TO), 1), YO, type=regressionType)

    # used prediction
    muYO_1XO = muYO.predict(np.concatenate((XO, np.ones_like(TO)), 1))
    muYO_0XO = muYO.predict(np.concatenate((XO, np.zeros_like(TO)), 1))
    muSE_1XO = muSE.predict(np.concatenate((XO, np.ones_like(TO)), 1))
    muSE_0XO = muSE.predict(np.concatenate((XO, np.zeros_like(TO)), 1))
    muSO_1XO = muSO.predict(np.concatenate((XO, np.ones_like(TO)), 1))
    muSO_0XO = muSO.predict(np.concatenate((XO, np.zeros_like(TO)), 1))

    tau = muYO_1XO - muYO_0XO + muSE_1XO - muSE_0XO + muSO_0XO - muSO_1XO

    return tau


def conditionalEquiConfoundingBiasEstimator_Tlearner(X, S, Y, T, G, regressionType='linear'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

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


def conditionalEquiConfoundingBiasEstimator_Tlearner_Removing(X, S, Y, T, G, regressionType='linear',tauETpye='regression'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

    muYO1 = regression1D(XO[TO[:,0]==1,:], YO[TO[:,0]==1,:], type=regressionType)
    muYO0 = regression1D(XO[TO[:,0]==0,:], YO[TO[:,0]==0,:], type=regressionType)

    S, T= S[:, None], T[:, None]

    muYO_1XO = muYO1.predict(XO)
    muYO_0XO = muYO0.predict(XO)

    from ite_RHC import shortTermIteEstimator
    ite_short, bias_fun = shortTermIteEstimator(X, S, T, G, tauETpye=tauETpye, regressionType=regressionType)

    # tau = muYO_1XO - muYO_0XO + muSE_1XO - muSE_0XO + muSO_0XO - muSO_1XO
    tau = muYO_1XO - muYO_0XO + bias_fun    # + ite_short + muSO_0XO - muSO_1XO

    return tau


def test():
    Osize, Esize = 2000, 1000
        # XO XE can be slightly different, but UO and UE should be totally same, i.e., X->G but not U->G
    XO, UO = np.random.multivariate_normal(mean=[2, 0], cov=[[1, 0], [0, 1]], size=Osize), \
        np.random.multivariate_normal(mean=[2, -1], cov=[[1, 1], [1, 1]], size=Osize)
    XE, UE = np.random.multivariate_normal(mean=[0, 2], cov=[[1, 0], [0, 1]], size=Esize), \
        np.random.multivariate_normal(mean=[2, -1], cov=[[1, 1], [1, 1]], size=Esize)

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
    SO_1 = 2 * np.mean(XO, axis=1) + 2* np.mean(UO, axis=1) + 3 * 1 * np.mean(XO, axis=1) + noiseSO
    SO_0 = 2 * np.mean(XO, axis=1) + 2* np.mean(UO, axis=1) + 3 * 0 * np.mean(XO, axis=1) + noiseSO
    SE_1 = 2 * np.mean(XE, axis=1) + 2* np.mean(UE, axis=1) + 3 * 1 * np.mean(XE, axis=1) + noiseSE
    SE_0 = 2 * np.mean(XE, axis=1) + 2* np.mean(UE, axis=1) + 3 * 0 * np.mean(XE, axis=1) + noiseSE

    SO = np.where(TO == 1, SO_1, SO_0)
    SE = np.where(TE == 1, SE_1, SE_0)

    noiseO, noiseE = np.random.randn(Osize) * 0.3, np.random.randn(Esize) * 0.3
    YO_1 = np.mean(XO, axis=1) + 2* np.mean(UO, axis=1) + SO_1 + 2 * 1 * np.mean(np.power(XO, 2), axis=1) + noiseO
    YO_0 = np.mean(XO, axis=1) + 2* np.mean(UO, axis=1) + SO_0 + 2 * 0 * np.mean(np.power(XO, 2), axis=1) + noiseO
    YE_1 = np.mean(XE, axis=1) + 2* np.mean(UE, axis=1) + SE_1 + 2 * 1 * np.mean(np.power(XE, 2), axis=1) + noiseE
    YE_0 = np.mean(XE, axis=1) + 2* np.mean(UE, axis=1) + SE_0 + 2 * 0 * np.mean(np.power(XE, 2), axis=1) + noiseE

    YO = np.where(TO == 1, YO_1, YO_0)
    YE = np.where(TE == 1, YE_1, YE_0)

    iteO, iteE = YO_1 - YO_0, YE_1 - YE_0

    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
        np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    print('ground truth ATE: ' + str(np.mean(iteO)))

    ite, _, _ = t_learner_estimator(XO, TO, YO, type='best')
    print('t_learner using obs data: ' + str(PEHE_ITE(ite, iteO))+ ' ' + str(RMSE_ATE(ite, iteO)))

    _, r1, r2 = t_learner_estimator(XE, TE, YE, type='best')
    ite_slearner = r1.predict(XO) - r2.predict(XO)
    print('t_learner using exp data: ' + str(PEHE_ITE(ite_slearner, iteO)) + ' ' + str(RMSE_ATE(ite_slearner, iteO)))

    ite_cecb = conditionalEquiConfoundingBiasEstimator(X, S, Y, T, G, regressionType='best')
    print('conditionalEquiConfoundingBiasEstimator: ' + str(PEHE_ITE(iteO,ite_cecb))+ ' ' + str(RMSE_ATE(ite_cecb, iteO)))

    ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S, Y, T, G, regressionType='best')
    print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO,ite_cecb_T))+ ' ' + str(RMSE_ATE(ite_cecb_T, iteO)))

    ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner_Removing(X, S, Y, T, G, tauETpye='reweight', regressionType='best')
    print('conditionalEquiConfoundingBiasEstimator_T_Removing_Reweight: ' + str(PEHE_ITE(iteO,ite_cecb_T))+ ' ' + str(RMSE_ATE(ite_cecb_T, iteO)))

    ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner_Removing(X, S, Y, T, G, regressionType='best')
    print('conditionalEquiConfoundingBiasEstimator_T_Removing_Regression: ' + str(PEHE_ITE(iteO,ite_cecb_T))+ ' ' + str(RMSE_ATE(ite_cecb_T, iteO)))


def test2():
    # generate T and then X
    Osize, Esize = 3000, 300
    TO = np.random.binomial(n=1, size=Osize, p=0.5)
    TE = np.random.binomial(n=1, size=Esize, p=0.5)

    XOL, UOL = [], []
    for i in range(Osize):
        XU = np.random.multivariate_normal(mean=[1, 0],
                                           cov=[[1, TO[i] - 0.5], [TO[i] - 0.5, 1]], size=1)
        XOL.append(XU[0, 0])
        UOL.append(XU[0, 1])
    XO, UO = np.array(XOL)[:, None], np.array(UOL)[:, None]

    XE, UE = np.random.uniform(-1, 1, Esize)[:, None], np.random.rand(Esize)[:, None]

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
    SO_1 = 2 * np.mean(XO, axis=1) + np.mean(UO, axis=1) + 3 * 1 * np.mean(XO, axis=1) + 1 * np.mean(np.power(XO, 2), axis=1) + noiseSO
    SO_0 = 2 * np.mean(XO, axis=1) + np.mean(UO, axis=1) + 3 * 0 * np.mean(XO, axis=1) + 0 * np.mean(np.power(XO, 2), axis=1) + noiseSO
    SE_1 = 2 * np.mean(XE, axis=1) + np.mean(UE, axis=1) + 3 * 1 * np.mean(XE, axis=1) + 1 * np.mean(np.power(XE, 2), axis=1) + noiseSE
    SE_0 = 2 * np.mean(XE, axis=1) + np.mean(UE, axis=1) + 3 * 0 * np.mean(XE, axis=1) + 0 * np.mean(np.power(XE, 2), axis=1) + noiseSE

    SO = np.where(TO == 1, SO_1, SO_0)
    SE = np.where(TE == 1, SE_1, SE_0)

    noiseO, noiseE = np.random.randn(Osize) * 0.3, np.random.randn(Esize) * 0.3
    YO_1 = np.mean(XO, axis=1) + np.mean(UO, axis=1) + SO_1 + 2 * 1 * np.mean(XO, axis=1) + noiseO
    YO_0 = np.mean(XO, axis=1) + np.mean(UO, axis=1) + SO_0 + 2 * 0 * np.mean(XO, axis=1) + noiseO
    YE_1 = np.mean(XE, axis=1) + np.mean(UE, axis=1) + SE_1 + 2 * 1 * np.mean(XE, axis=1) + noiseE
    YE_0 = np.mean(XE, axis=1) + np.mean(UE, axis=1) + SE_0 + 2 * 0 * np.mean(XE, axis=1) + noiseE

    YO = np.where(TO == 1, YO_1, YO_0)
    YE = np.where(TE == 1, YE_1, YE_0)

    iteO, iteE = YO_1 - YO_0, YE_1 - YE_0

    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
        np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    print('ground truth ATE: ' + str(np.mean(iteO)))

    ite, _, _ = t_learner_estimator(XO, TO, YO, type='best')
    print('t_learner using obs data: ' + str(PEHE_ITE(ite, iteO))+ ' ' + str(RMSE_ATE(ite, iteO)))

    _, r1, r2 = t_learner_estimator(XE, TE, YE, type='best')
    ite_slearner = r1.predict(XO) - r2.predict(XO)
    print('t_learner using exp data: ' + str(PEHE_ITE(ite_slearner, iteO)) + ' ' + str(RMSE_ATE(ite_slearner, iteO)))

    ite_cecb = conditionalEquiConfoundingBiasEstimator(X, S, Y, T, G, regressionType='best')
    print('conditionalEquiConfoundingBiasEstimator: ' + str(PEHE_ITE(iteO,ite_cecb))+ ' ' + str(RMSE_ATE(ite_cecb, iteO)))

    ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S, Y, T, G, regressionType='best')
    print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO,ite_cecb_T))+ ' ' + str(RMSE_ATE(ite_cecb_T, iteO)))

    ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner_Removing(X, S, Y, T, G, tauETpye='reweight', regressionType='best')
    print('conditionalEquiConfoundingBiasEstimator_T_Removing_Reweight: ' + str(PEHE_ITE(iteO,ite_cecb_T))+ ' ' + str(RMSE_ATE(ite_cecb_T, iteO)))

    ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner_Removing(X, S, Y, T, G, regressionType='best')
    print('conditionalEquiConfoundingBiasEstimator_T_Removing_Regression: ' + str(PEHE_ITE(iteO,ite_cecb_T))+ ' ' + str(RMSE_ATE(ite_cecb_T, iteO)))



import warnings
warnings.simplefilter('ignore')
test()
test2()