import sys
from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor


'''
implement of Removing Hidden Confounding byExperimental Grounding, Nathan Kallus, Aahlad Manas Puli, Uri Shalit
'''

def shortTermIteEstimator(X, Y, T, G, regressionType='linear', tauETpye='reweight'):
    """
    :param X: array with shape (size, dim)
    :param Y: array with shape (size, 1)
    :param T: array with shape (size, 1)
    :param G: array with shape (size, ), where G_i=1 means sample i belongs to experimental data
    :param regressionType:
    :param tauETpye:
    :return:
    """
    # devide into G=O and G=E
    XO, YO, TO = X[G == 0], Y[G == 0], T[G == 0]
    XE, YE, TE = X[G == 1], Y[G == 1], T[G == 1]

    # omega = muso1_xo - muso0_xo
    muYO_1 = regression1D(XO[TO[:,0]==1,:], YO[TO[:,0]==1,:], type=regressionType)
    muYO_0 = regression1D(XO[TO[:,0]==0,:], YO[TO[:,0]==0,:], type=regressionType)

    muYO_1XE = muYO_1.predict(XE)
    muYO_0XE = muYO_0.predict(XE)
    omega_XE = muYO_1XE - muYO_0XE

    if tauETpye == 'reweight':
        eXE = e_x_estimator(XE, TE)
        eXE_prob = eXE.predict_proba(XE)[:, 1][:, None]
        q_YE = TE / eXE_prob - (1-TE)/(1-eXE_prob)
        tau_E = q_YE * YE
    else:
        muYE_1 = regression1D(XE[TE[:, 0] == 1, :], YE[TE[:, 0] == 1, :], type=regressionType)
        muYE_0 = regression1D(XE[TE[:, 0] == 0, :], YE[TE[:, 0] == 0, :], type=regressionType)
        tau_E = muYE_1.predict(XE) - muYE_0.predict(XE)
        tau_E = tau_E[:,None]  # ensure shape [size, 1]

    # print(omega_XE[:,None].shape, tau_E.shape)

    # eta_XE = regression1D(XE, tau_E-omega_XE[:,None], type=regressionType, bias=False)
    eta_XE = regression1D(XE, tau_E-omega_XE[:,None], type=regressionType, bias=False)

    # print(eta_XE.coef_)

    muYO_1XO = muYO_1.predict(XO)
    muYO_0XO = muYO_0.predict(XO)
    omega_XO = muYO_1XO - muYO_0XO

    # print(eta_XE.predict(XO).shape, omega_XO.shape)
    etaXO_XE = eta_XE.predict(XO)
    tau = etaXO_XE + omega_XO


    return tau, eta_XE.predict(XO)


def test_paper():
    Osize, Esize = 3000, 600

    TO = np.random.binomial(n=1, size=Osize, p=0.5)
    TE = np.random.binomial(n=1, size=Esize, p=0.5)

    XOL, UOL = [], []
    for i in range(Osize):
        XU = np.random.multivariate_normal(mean=[0, 0],
                                           cov=[[1, TO[i]-0.5], [TO[i]-0.5, 1]], size=1)
        XOL.append(XU[0,0])
        UOL.append(XU[0,1])
    XO, UO = np.array(XOL)[:,None], np.array(UOL)[:,None]

    XE, UE = np.random.uniform(-1, 1, Esize)[:,None], np.random.rand(Esize)[:,None]

    print('sizeO,E=' + str(Osize) + str(',') + str(Esize))

    noiseO, noiseE = np.random.randn(Osize), np.random.randn(Esize)
    YO = 1 + TO[:,None] + XO + 2 * TO[:,None] * XO + 0.5 * np.power(XO,2) + 0.75 * TO[:,None] * np.power(XO,2) + UO + 0.5 * noiseO[:,None]
    YE = 1 + TE[:,None] + XE + 2 * TE[:,None] * XE + 0.5 * np.power(XE,2) + 0.75 * TE[:,None] * np.power(XE,2) + UE + 0.5 * noiseE[:,None]

    iteO, iteE = 2* XO + 0.75 * np.power(XO,2) + 1, 2* XE + 0.75 * np.power(XE,2) + 1

    X, T, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0)[:, None], np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    print('ground truth ATE on short-term: ' + str(np.mean(iteO)))

    ite, _, _ = t_learner_estimator(XO, TO, YO, type='best')
    print('t_learner using obs data: ' + str(PEHE_ITE(ite, iteO)))

    _, r1, r2 = t_learner_estimator(XE, TE, YE, type='best')
    ite_slearner = r1.predict(XO) - r2.predict(XO)
    print('t_learner using exp data: ' + str(PEHE_ITE(ite_slearner, iteO)))

    ite_cecb, _ = shortTermIteEstimator(X, Y, T, G, regressionType='best')
    print('shortTermIteEstimator: ' + str(PEHE_ITE(iteO, ite_cecb)))

    ite_cecb, _ = shortTermIteEstimator(X, Y, T, G, tauETpye='regression', regressionType='best')
    print('shortTermIteEstimator: ' + str(PEHE_ITE(iteO, ite_cecb)))



def test():
    Osize, Esize = 5000, 2000

    XO, UO = np.random.normal(1,1, size=Osize),  np.random.normal(1,1, size=Osize)
    XE, UE = np.random.normal(1,1, size=Esize),  np.random.normal(1,1, size=Esize)

    TO = np.random.binomial(n=1, size=Osize, p=(1 / (1 + np.exp(XO+UO))))
    TE = np.random.binomial(n=1, size=Esize, p=(1 / (1 + np.exp(XE))))
    # TE = np.random.binomial(n=1, size=Esize, p=0.5)


    print('sizeO,E=' + str(Osize) + str(',') + str(Esize))

    noiseO, noiseE = np.random.randn(Osize) * 0.5, np.random.randn(Esize) * 0.5
    YO_1 = 1 + 1 + XO + 2 * 1 * XO + 0.5 * np.power(XO, 2) + 0.75 * 1 * np.power(XO, 2) + UO + 0.5 * noiseO
    YO_0 = 1 + 0 + XO + 2 * 0 * XO + 0.5 * np.power(XO, 2) + 0.75 * 0 * np.power(XO, 2) + UO + 0.5 * noiseO

    YE_1 = 1 + 1 + XE + 2 * 1 * XE + 0.5 * np.power(XE, 2) + 0.75 * 1 * np.power(XE,2) + UE + 0.5 * noiseE
    YE_0 = 1 + 0 + XE + 2 * 0 * XE + 0.5 * np.power(XE, 2) + 0.75 * 0 * np.power(XE,2) + UE + 0.5 * noiseE

    YO = np.where(TO == 1, YO_1, YO_0)
    YE = np.where(TE == 1, YE_1, YE_0)

    iteO, iteE = YO_1 - YO_0, YE_1 - YE_0

    X, T, Y, G = np.concatenate((XO, XE), 0)[:,None], np.concatenate((TO, TE), 0)[:, None], np.concatenate((YO, YE), 0)[:, None], \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0)
    XO, XE, TO, TE, YO, YE = XO[:, None], XE[:, None], TO[:, None], TE[:, None], YO[:, None], YE[:, None]

    # print(X.shape, T.shape, Y.shape, G.shape)

    print('ground truth ATE on short-term: ' + str(np.mean(iteO)))

    # ite, _, _ = t_learner_estimator(XO, TO, YO, type='best')
    # print('t_learner using obs data: ' + str(PEHE_ITE(ite, iteO)))

    # _, r1, r2 = t_learner_estimator(XE, TE, YE, type='best')
    # ite_slearner = r1.predict(XO) - r2.predict(XO)
    # print('t_learner using exp data: ' + str(PEHE_ITE(ite_slearner, iteO)))

    ite_cecb, _ = shortTermIteEstimator(X, Y, T, G, regressionType='best')
    print('shortTermIteEstimator: ' + str(PEHE_ITE(iteO, ite_cecb)))

    ite_cecb, _ = shortTermIteEstimator(X, Y, T, G, tauETpye='regression', regressionType='best')
    print('shortTermIteEstimator: ' + str(PEHE_ITE(iteO, ite_cecb)))


if __name__ == '__main__':

    import warnings
    warnings.simplefilter('ignore')
    # test_paper()
    test()
