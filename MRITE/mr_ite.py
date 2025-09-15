from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split


def nuisance_Slearner(X, S, Y, T, G, regressionType='linear'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0]
    XE, SE, TE = X[G == 1], S[G == 1], T[G == 1]

    muSE = regression1D(np.concatenate((XE, TE), 1), SE, type=regressionType)
    muSO = regression1D(np.concatenate((XO, TO), 1), SO, type=regressionType)
    muYO = regression1D(np.concatenate((XO, TO), 1), YO, type=regressionType)
    piE = e_x_estimator(XE, TE)
    piO = e_x_estimator(XO, TO)
    piG = e_x_estimator(X, G)

    return muSE, muSO, muYO, piE, piO, piG

def nuisance_Tlearner(X, S, Y, T, G, regressionType='linear'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0]
    XE, SE, TE = X[G == 1], S[G == 1], T[G == 1]

    muSE1 = regression1D(XE[TE==1,:], SE[TE==1], type=regressionType)
    muSE0 = regression1D(XE[TE==0,:], SE[TE==0], type=regressionType)
    muSO1 = regression1D(XO[TO==1,:], SO[TO==1], type=regressionType)
    muSO0 = regression1D(XO[TO==0,:], SO[TO==0], type=regressionType)
    muYO1 = regression1D(XO[TO==1,:], YO[TO==1], type=regressionType)
    muYO0 = regression1D(XO[TO==0,:], YO[TO==0], type=regressionType)
    piE = e_x_estimator(XE, TE)
    piO = e_x_estimator(XO, TO)
    piG = e_x_estimator(X, G)

    return muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG


def construct_YMR(muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG,
                  X, S, Y, T, G):

    muYO_1X = muYO1.predict(X)
    muYO_0X = muYO0.predict(X)
    muYO_X = np.where(T==1, muYO_1X, muYO_0X)

    muSE_1X = muSE1.predict(X)
    muSE_0X = muSE0.predict(X)
    muSE_X = np.where(T==1, muSE_1X, muSE_0X)

    muSO_1X = muSO1.predict(X)
    muSO_0X = muSO0.predict(X)
    muSO_X = np.where(T==1, muSO_1X, muSO_0X)

    piE_X = piE.predict_proba(X)[:, 1]  # [:, None]  p(T=1|X,G=E)
    piO_X = piO.predict_proba(X)[:, 1]  # [:, None]  p(T=1|X,G=O)
    piG_X = piG.predict_proba(X)[:, 1]  # [:, None]  p(G=E|X)
    pG = np.sum(G==0) / G.shape[0]      #            p(G=O)

    # print( piE_X.shape)
    # print( (2*T-1).shape)
    # print( ((2*T-1)*piE_X).shape)

    Y_MR = G / pG * (2*T-1)/(1-T + (2*T-1)*piE_X) * (S - muSE_X) * ( 1/piG_X -1) + \
           (1- G) / pG * ( (2*T-1)/(1-T + (2*T-1)*piO_X) * (Y - muYO_X - S + muSO_X) +
                           muYO_1X - muYO_0X + muSE_1X - muSE_0X + muSO_0X - muSO_1X )
    # print(Y_MR.shape)
    return Y_MR


def pseudo_reg(Y_MR, X, regressionType='best'):
    cate = regression1D(X, Y_MR, type=regressionType)
    return cate

def MR_LTHE(X, S, Y, T, G, predict_X, cross_fitting=True, learner='T', regressionType='best'):
    # construct D1 and D2
    X,X2, S,S2, Y,Y2, T,T2, G,G2 = train_test_split(X, S, Y, T, G, test_size = 0.5)

    # using D1 construct nuisances
    if learner == 'T':
        # muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG
        muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG = nuisance_Tlearner(X, S, Y, T, G, regressionType=regressionType)
    else:
        muSE, muSO, muYO, piE, piO, piG = nuisance_Slearner(X, S, Y, T, G, regressionType=regressionType)
        muSE1, muSO1, muYO1, muSE0, muSO0, muYO0 =  muSE, muSO, muYO,  muSE, muSO, muYO

    # using D2 construct pseudo Y
    Y_MR = construct_YMR(muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG,
                  X2, S2, Y2, T2, G2)
    # cate estimator
    cate = pseudo_reg(Y_MR, X2, regressionType=regressionType)
    cate_predict = cate.predict(predict_X)

    # print(cate_1.shape)
    if cross_fitting is True:
        if learner == 'T':
            muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG = nuisance_Tlearner(X2, S2, Y2, T2, G2, regressionType=regressionType)
        else:
            muSE, muSO, muYO, piE, piO, piG = nuisance_Slearner(X2, S2, Y2, T2, G2, regressionType=regressionType)
            muSE1, muSO1, muYO1, muSE0, muSO0, muYO0 =  muSE, muSO, muYO,  muSE, muSO, muYO

        # using D construct pseudo Y
        Y_MR = construct_YMR(muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG,
                      X, S, Y, T, G)
        # cate estimator
        cate = pseudo_reg(Y_MR, X, regressionType=regressionType)
        cate_2 = cate.predict(predict_X)
        cate_predict = (cate_predict + cate_2)/2

    return cate_predict


def test(seed=0):
    np.random.seed(seed)

    Osize, Esize = 3000, 1000
        # XO XE can be slightly different, but UO and UE should be totally same, i.e., X->G but not U->G
    XO, UO = np.random.multivariate_normal(mean=[1, -1], cov=[[1, 0], [0, 1]], size=Osize), \
        np.random.multivariate_normal(mean=[1, -1], cov=[[1, 0], [0, 1]], size=Osize)
    XE, UE = np.random.multivariate_normal(mean=[-1, 1], cov=[[1, 0], [0, 1]], size=Esize), \
        np.random.multivariate_normal(mean=[1, -1], cov=[[1, 0], [0, 1]], size=Esize)

    unconf_strength=0
    x2t = np.random.normal(0, 1, size=2)
    u2t = np.random.normal(0, 1, size=2) + unconf_strength

    TO = np.random.binomial(n=1, size=Osize, p=(1 / (1 + np.exp(-(np.matmul(XO,x2t)+np.matmul(UO,u2t))))))
    TE = np.random.binomial(n=1, size=Esize, p=(1 / (1 + np.exp(-np.matmul(XE,x2t)))))

    # rule out extreme data
    indexO = np.logical_and(((1 / (1 + np.exp(-(np.matmul(XO,x2t)+np.matmul(UO,u2t))))) < 0.95),
                            ((1 / (1 + np.exp(-(np.matmul(XO,x2t)+np.matmul(UO,u2t))))) > 0.05))
    indexE = np.logical_and(((1 / (1 + np.exp(-np.matmul(XE,x2t)))) < 0.95),
                            ((1 / (1 + np.exp(-np.matmul(XE,x2t)))) > 0.05))
    XO, UO = XO[indexO, :], UO[indexO, :]
    XE, UE = XE[indexE, :], UE[indexE, :]
    TO, TE = TO[indexO], TE[indexE]
    Osize, Esize = TO.shape[0], TE.shape[0]

    print('sizeO,E=' + str(Osize) + str(',') + str(Esize))

    noise_strength = 1

    noiseSO, noiseSE = np.random.randn(Osize) * noise_strength, np.random.randn(Esize) * noise_strength
    SO_1 = 2 * np.mean(XO, axis=1) + 2* np.mean(UO, axis=1) + 3 * 1 * np.mean(XO, axis=1) + 2 + noiseSO
    SO_0 = 2 * np.mean(XO, axis=1) + 2* np.mean(UO, axis=1) + 3 * 0 * np.mean(XO, axis=1) + 0 + noiseSO
    SE_1 = 2 * np.mean(XE, axis=1) + 2* np.mean(UE, axis=1) + 3 * 1 * np.mean(XE, axis=1) + 2 + noiseSE
    SE_0 = 2 * np.mean(XE, axis=1) + 2* np.mean(UE, axis=1) + 3 * 0 * np.mean(XE, axis=1) + 0 + noiseSE

    SO = np.where(TO == 1, SO_1, SO_0)
    SE = np.where(TE == 1, SE_1, SE_0)

    noiseO, noiseE = np.random.randn(Osize) * noise_strength, np.random.randn(Esize) * noise_strength
    YO_1 = np.mean(XO, axis=1) + 2* np.mean(UO, axis=1) + SO_1 + 2 * 1 * np.mean(np.power(XO, 2), axis=1) + 1 + noiseO
    YO_0 = np.mean(XO, axis=1) + 2* np.mean(UO, axis=1) + SO_0 + 2 * 0 * np.mean(np.power(XO, 2), axis=1) + 0 + noiseO
    YE_1 = np.mean(XE, axis=1) + 2* np.mean(UE, axis=1) + SE_1 + 2 * 1 * np.mean(np.power(XE, 2), axis=1) + 1 + noiseE
    YE_0 = np.mean(XE, axis=1) + 2* np.mean(UE, axis=1) + SE_0 + 2 * 0 * np.mean(np.power(XE, 2), axis=1) + 0 + noiseE

    YO = np.where(TO == 1, YO_1, YO_0)
    YE = np.where(TE == 1, YE_1, YE_0)

    iteO, iteE = YO_1 - YO_0, YE_1 - YE_0

    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
        np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    print('ground truth ATE: ' + str(np.mean(iteO)))

    ite, _, _ = t_learner_estimator(XO, TO, YO, type='kernelRidge')
    print('t_learner using obs data: ' + str(PEHE_ITE(ite, iteO)) + ' ' + str(RMSE_ATE(ite, iteO)))

    ate = ipw_estimator(XO, TO, YO)
    print('ipw using obs data: ' + str(np.abs(np.mean(iteO) - np.mean(ate)) ))

    _, r1, r2 = t_learner_estimator(XE, TE, YE, type='kernelRidge')
    ite_slearner = r1.predict(XO) - r2.predict(XO)
    print('t_learner using exp data: ' + str(PEHE_ITE(ite_slearner, iteO)) + ' ' + str(RMSE_ATE(ite_slearner, iteO)))

    ite_mr = MR_LTHE(X, S, Y, T, G, predict_X=XO, cross_fitting=True, regressionType='kernelRidge')
    print('MR_LTCE: ' + str(PEHE_ITE(ite_mr, iteO)) + ' ' + str(RMSE_ATE(ite_mr, iteO)))

def test_dgp(seed=0):
    from utils.dgp import dgp_equ_conf, dgp_equ_conf_design
    X, T, S, Y, G, \
        XO, TO, SO, YO, iteO, \
        XE, TE, SE, YE, iteE = \
    dgp_equ_conf_design(size_O=10000, size_E=2000, seed=seed)

    # dgp_equ_conf(dim_x=1, dim_u=1, size_O=2000, size_E=500, seed=seed)

    print('ground truth ATE: ' + str(np.mean(iteO)))

    ite, _, _ = t_learner_estimator(XO, TO, YO, type='best')
    print('t_learner using obs data: ' + str(PEHE_ITE(ite, iteO)) + ' ' + str(RMSE_ATE(ite, iteO)))

    # ate = ipw_estimator(XO, TO, YO)
    # print('ipw using obs data: ' + str(np.abs(np.mean(iteO) - np.mean(ate)) ))
    #
    # _, r1, r2 = t_learner_estimator(XE, TE, YE, type='kernelRidge')
    # ite_slearner = r1.predict(XO) - r2.predict(XO)
    # print('t_learner using exp data: ' + str(PEHE_ITE(ite_slearner, iteO)) + ' ' + str(RMSE_ATE(ite_slearner, iteO)))
    #
    # ite_naive = naive_LTHE(X, S, Y, T, G, predict_X=XO, regressionType='best')
    # print('naive_LTCE: ' + str(PEHE_ITE(ite_naive, iteO)) + ' ' + str(RMSE_ATE(ite_naive, iteO)))

    # ite_mr = MR_LTHE(X, S, Y, T, G, predict_X=XO, cross_fitting=True, regressionType='linear')
    # print(ite_mr.shape, iteO.shape)
    # print('MR_LTCE: ' + str(PEHE_ITE(ite_mr, iteO)) + ' ' + str(RMSE_ATE(ite_mr, iteO)))

    return  PEHE_ITE(ite, iteO),RMSE_ATE(ite, iteO) , 0,0#PEHE_ITE(ite_naive, iteO), RMSE_ATE(ite_naive, iteO)



if __name__ == '__main__':

    import warnings
    warnings.simplefilter('ignore')
    iteT, ateT, iteN, ateN = [], [],[], []
    for i in range(10):
        print('seed' + str(i))
        ite_errorT, ate_errorT, ite_errorN, ate_errorN = test_dgp(i)
        iteN.append(ite_errorN)
        ateN.append(ate_errorN)
        iteT.append(ite_errorT)
        ateT.append(ate_errorT)

    print(np.mean(iteN),np.std(iteN))
    print(np.mean(ateN),np.std(ateN))

    print(np.mean(iteT),np.std(iteT))
    print(np.mean(ateT),np.std(ateT))
    # test2()

# from utils.save import save_pkl, read_pkl
# data = {
#     'ite_mean': np.mean(ite),
#     'ite_std': np.std(ite),
#     'ate_mean': np.mean(ate),
#     'ate_std': np.mean(ate),
#     'ite_error': ite,
#     'ate_error': ate,
# }
# save_pkl('../results/mr_ite_mlp.pkl',data)
#
# d = read_pkl('../results/mr_ite_mlp.pkl')
# print(d)