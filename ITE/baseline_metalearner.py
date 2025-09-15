from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split
from utils.dgp import set_all_seeds


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


def test_IHDP(seed=123):
    from utils.IHDP_datasets import LongTermIHDP
    import warnings
    warnings.simplefilter('ignore')

    ipw_ate_Error = []
    tlearner_ate_Error, tlearner_ite_Error = [],[]
    tlearner_ate_outError, tlearner_ite_outError = [],[]
    lti = LongTermIHDP()
    for i, (train, valid, test) in enumerate(lti.get_train_valid_test()):
        set_all_seeds(seed)

        g_train, t_train, s_train, y_train, _, _, y1_train, y0_train, x_train, _ = train
        g_valid, t_valid, s_valid, y_valid, _, _, y1_valid, y0_valid, x_valid, _ = valid
        t_test, s_test, y_test, _, _, y1_test, y0_test, x_test, _ = test
        # ground true
        iteO_test = y1_test - y0_test
        # print(x_train.shape)
        # tau = imputationApproach(x_train, np.squeeze(s_train), np.squeeze(y_train),
        #                          np.squeeze(t_train), np.squeeze(g_train), regressionType='linear')
        # print(np.abs(np.mean(iteO_test) - np.mean(tau)))
        # ate_Error.append(np.abs(np.mean(iteO_test) - np.mean(tau)))
        o_index = np.squeeze(g_train==1)
        XO = x_train[o_index,:]
        YO = y_train[o_index]
        TO = t_train[o_index]
        iteO = y1_train[o_index] - y0_train[o_index]

        ite, regression_1, regression_0 = t_learner_estimator(XO, TO, YO, type='kernelRidge')
        # ite = ite[:,None]
        # print(ite.shape, iteO.shape)
        # print('t_learner using obs data: ' + str(PEHE_ITE(ite, iteO)) + ' ' + str(RMSE_ATE(ite, iteO)))
        tlearner_ate_Error.append(RMSE_ATE(ite, iteO))
        tlearner_ite_Error.append(PEHE_ITE(ite, iteO))

        y1 = regression_1.predict(X=x_test)
        y0 = regression_0.predict(X=x_test)
        ite_test = y1-y0
        # ite_test = ite_test[:,None]
        tlearner_ate_outError.append(RMSE_ATE(ite_test, iteO_test))
        tlearner_ite_outError.append(PEHE_ITE(ite_test, iteO_test))
        print('Tlearner ' + str(i) + ': ' + str(PEHE_ITE(ite_test, iteO_test)) + ' ' + str(RMSE_ATE(ite_test, iteO_test)))

        ate = ipw_estimator(XO, TO, YO)
        # print('ipw using obs data: ' + str(np.abs(np.mean(iteO) - np.mean(ate))))
        ipw_ate_Error.append(np.abs(np.mean(iteO) - np.mean(ate)))


        # _, r1, r2 = t_learner_estimator(XE, TE, YE, type='kernelRidge')
        # ite_slearner = r1.predict(XO) - r2.predict(XO)
        # print(
        #     't_learner using exp data: ' + str(PEHE_ITE(ite_slearner, iteO)) + ' ' + str(RMSE_ATE(ite_slearner, iteO)))

    return ipw_ate_Error , tlearner_ate_Error, tlearner_ite_Error,tlearner_ate_outError, tlearner_ite_outError


def test_NEWS_val(seed=123):
    from utils.NEWS_dataset import LongTermNEWS
    import warnings
    warnings.simplefilter('ignore')

    lti = LongTermNEWS()
    train, valid, test =lti.get_tune_train_valid_test()
    set_all_seeds(seed)
    g_train, t_train, s_train, y_train, y1_train, y0_train, x_train = train
    g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid = valid
    t_test, s_test, y_test, y1_test, y0_test, x_test = test

    # ground true
    ite_test = y1_test - y0_test
    ite_train = y1_train - y0_train
    ite_valid = y1_valid - y0_valid
    # tau = imputationApproach(x_train, np.squeeze(s_train), np.squeeze(y_train),
    #                          np.squeeze(t_train), np.squeeze(g_train), regressionType='linear')
    # print(np.abs(np.mean(iteO_test) - np.mean(tau)))
    # ate_Error.append(np.abs(np.mean(iteO_test) - np.mean(tau)))
    o_index = np.squeeze(g_train==0)
    XO = x_train[o_index,:]
    YO = y_train[o_index]
    TO = t_train[o_index]
    iteO = y1_train[o_index] - y0_train[o_index]

    ite, regression_1, regression_0 = t_learner_estimator(XO, TO, YO, type='kernelRidge')
    # ite = ite[:,None]
    # print(ite.shape, iteO.shape)
    # print('t_learner using obs data: ' + str(PEHE_ITE(ite, iteO)) + ' ' + str(RMSE_ATE(ite, iteO)))

    y1 = regression_1.predict(X=x_test)
    y0 = regression_0.predict(X=x_test)
    ite_est_test = y1-y0
    y1 = regression_1.predict(X=x_train)
    y0 = regression_0.predict(X=x_train)
    ite_est_train = y1-y0
    y1 = regression_1.predict(X=x_valid)
    y0 = regression_0.predict(X=x_valid)
    ite_est_valid = y1-y0
    # ite_test = ite_test[:,None]

    print(RMSE_ATE(ite_test, ite_est_test), PEHE_ITE(ite_test, ite_est_test))
    print(RMSE_ATE(ite_train, ite_est_train), PEHE_ITE(ite_train, ite_est_train))
    print(RMSE_ATE(ite_valid, ite_est_valid), PEHE_ITE(ite_valid, ite_est_valid))


    # return ipw_ate_Error , tlearner_ate_Error, tlearner_ite_Error,tlearner_ate_outError, tlearner_ite_outError


def test_NEWS(seed=123):
    from utils.NEWS_dataset import LongTermNEWS
    import warnings
    warnings.simplefilter('ignore')

    ipw_ate_Error = []
    tlearner_ate_Error, tlearner_ite_Error = [],[]
    tlearner_ate_outError, tlearner_ite_outError = [],[]
    lti = LongTermNEWS()
    for i, (train, valid, test) in enumerate(lti.get_train_valid_test()):
        set_all_seeds(seed)
        g_train, t_train, s_train, y_train, y1_train, y0_train, x_train = train
        g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid = valid
        t_test, s_test, y_test, y1_test, y0_test, x_test = test

        # ground true
        iteO_test = y1_test - y0_test
        # tau = imputationApproach(x_train, np.squeeze(s_train), np.squeeze(y_train),
        #                          np.squeeze(t_train), np.squeeze(g_train), regressionType='linear')
        # print(np.abs(np.mean(iteO_test) - np.mean(tau)))
        # ate_Error.append(np.abs(np.mean(iteO_test) - np.mean(tau)))
        o_index = np.squeeze(g_train==1)
        XO = x_train[o_index,:]
        YO = y_train[o_index]
        TO = t_train[o_index]
        iteO = y1_train[o_index] - y0_train[o_index]

        ite, regression_1, regression_0 = t_learner_estimator(XO, TO, YO, type='kernelRidge')
        # ite = ite[:,None]
        # print(ite.shape, iteO.shape)
        # print('t_learner using obs data: ' + str(PEHE_ITE(ite, iteO)) + ' ' + str(RMSE_ATE(ite, iteO)))
        tlearner_ate_Error.append(RMSE_ATE(ite, iteO))
        tlearner_ite_Error.append(PEHE_ITE(ite, iteO))

        y1 = regression_1.predict(X=x_test)
        y0 = regression_0.predict(X=x_test)
        ite_test = y1-y0
        # ite_test = ite_test[:,None]
        tlearner_ate_outError.append(RMSE_ATE(ite_test, iteO_test))
        tlearner_ite_outError.append(PEHE_ITE(ite_test, iteO_test))
        print('Tlearner ' + str(i) + ': ' + str(PEHE_ITE(ite_test, iteO_test)) + ' ' + str(RMSE_ATE(ite_test, iteO_test)))

        ate = ipw_estimator(XO, TO, YO)
        # print('ipw using obs data: ' + str(np.abs(np.mean(iteO) - np.mean(ate))))
        ipw_ate_Error.append(np.abs(np.mean(iteO) - np.mean(ate)))


        # _, r1, r2 = t_learner_estimator(XE, TE, YE, type='kernelRidge')
        # ite_slearner = r1.predict(XO) - r2.predict(XO)
        # print(
        #     't_learner using exp data: ' + str(PEHE_ITE(ite_slearner, iteO)) + ' ' + str(RMSE_ATE(ite_slearner, iteO)))

    return ipw_ate_Error , tlearner_ate_Error, tlearner_ite_Error,tlearner_ate_outError, tlearner_ite_outError


if __name__ == '__main__':

    import warnings
    warnings.simplefilter('ignore')
    # test_NEWS_val(123)
    # ipw_ate_Error , tlearner_ate_Error, tlearner_ite_Error,tlearner_ate_outError, tlearner_ite_outError\
    #     = test_NEWS(123)
    ipw_ate_Error, tlearner_ate_Error, tlearner_ite_Error, tlearner_ate_outError, tlearner_ite_outError \
        = test_IHDP(123)
    print(np.mean(ipw_ate_Error), np.std(ipw_ate_Error))
    print(np.mean(tlearner_ite_Error), np.std(tlearner_ite_Error))
    print(np.mean(tlearner_ate_Error), np.std(tlearner_ate_Error))

    print(len(tlearner_ite_outError))
    print(np.mean(tlearner_ite_outError), np.std(tlearner_ite_outError))
    print(np.mean(tlearner_ate_outError), np.std(tlearner_ate_outError))

#0.17390769337100437 0.07643203974660244
# 0.24279401641643003 0.16254282319937213
# 1.4954700352483103 0.4750284325513066
# 0.23121325749246974 0.2287339585904603
# 1.7103180524237556 0.8866819709756265

# NEWS
# 2.880188813247508 0.11885870160620747
# 0.6972480140079578 0.1248186761368429