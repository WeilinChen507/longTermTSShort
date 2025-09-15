import sys

sys.path.append('..')

import torch
import argparse
from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split


def fitShortTermNuisance(X, S, T, G, regressionType='linear',power=1):
    # devide into G=O and G=E
    XO, SO, TO = X[G == 0], S[G == 0], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1], T[G == 1][:, None]

    # S shape = [sampe size, time length]
    sample_size = SO.shape[0]
    short_time = SO.shape[1]
    bias = np.zeros_like(SO)
    muSE1, muSE0, muSO1, muSO0 = [],[],[],[]

    muSE1 = [regression1D(XE[TE[:, 0] == 1, :], np.squeeze(SE[TE[:, 0] == 1, i]), power=power, type=regressionType)
                for i in range(short_time)]
    muSE0 = [regression1D(XE[TE[:, 0] == 0, :], np.squeeze(SE[TE[:, 0] == 0, i]), power=power, type=regressionType)
                for i in range(short_time)]
    muSO1 = [regression1D(XO[TO[:, 0] == 1, :], np.squeeze(SO[TO[:, 0] == 1, i]), power=power, type=regressionType)
                for i in range(short_time)]
    muSO0 = [regression1D(XO[TO[:, 0] == 0, :], np.squeeze(SO[TO[:, 0] == 0, i]), power=power, type=regressionType)
                for i in range(short_time)]

    def bias(X_bias):
        # devide into G=O and G=E
        # XO, SO, TO = X[G == 0], S[G == 0], T[G == 0][:, None]
        # XE, SE, TE = X[G == 1], S[G == 1], T[G == 1][:, None]
        # S shape = [sampe size, time length]
        # sample_size = SO.shape[0]
        # short_time = SO.shape[1]
        biases = np.zeros([X_bias.shape[0], len(muSO1)])
        for i in range(len(muSO1)):
            # used prediction
            muSE_1XO = muSE1[i].predict(X_bias)
            muSE_0XO = muSE0[i].predict(X_bias)
            muSO_1XO = muSO1[i].predict(X_bias)
            muSO_0XO = muSO0[i].predict(X_bias)
            biases[:, i] = muSE_1XO - muSE_0XO + muSO_0XO - muSO_1XO
        return biases

    return bias, muSE1, muSE0, muSO1, muSO0

def fitFCAECB(X, biases, regressionType='linear', power=1, lr=1e-2):
    # fit the function between biases
    # b_t(x) \times f(x) = b_{t+1}(x)
    # \hat f(x) = \argmin | b_{t+1}(x)/b_t(x) - f(x) |^2
    sample_size, short_time = biases.shape[0], biases.shape[1]

    # reshape bias to fit a bias model
    bias_train = np.zeros([sample_size * (short_time - 1), 2])
    X_train = np.zeros([sample_size * (short_time - 1), X.shape[1]])
    for i in range(short_time - 1):
        bias_train[i * sample_size:(i + 1) * sample_size, 0] = biases[:, i]
        bias_train[i * sample_size:(i + 1) * sample_size, 1] = biases[:, i + 1]
        X_train[i * sample_size:(i + 1) * sample_size, :] = X[:,:]

    # input = np.repeat(X, repeats=short_time - 1, axis=0)
    # output = bias_train[:, 1] / bias_train[:, 0]

    # fitting
    # model = regression1D(input, output, type=regressionType, power=power)
    from utils.basic_model import TrainedMLPLinear
    model = TrainedMLPLinear(input_dim=X.shape[1], output_dim=1, regressionType=regressionType, device='cuda').\
        fit(X=X_train,
            Y=bias_train[:, 1][:,None],
            factor=bias_train[:, 0][:,None],
            # batch_size=200,
            batch_size=int(X_train.shape[0]/20),
            epochs=400,
            lr=lr)  # for linear, lr can be relatively large
    # print(model.coef_())
    return model


def conditionalEquiConfoundingBiasEstimator_split(X, S, Y, T, G, TIME=15, regressionType='linear', power=1, split=True, lr=1e-2):
    if split:
        # sample split into D1 D2
        X1,X2, S1,S2, Y1,Y2, T1,T2, G1,G2 = train_test_split(X, S, Y, T, G, test_size = 0.5)

        # devide into G=O and G=E
        # XE, SE, TE = X[G == 1], S[G == 1], T[G == 1][:, None]

        # fit short-term nuisance function and short-term confounding bias using D1
        bias_func1, _,_,_,_ = fitShortTermNuisance(X1, S1, T1, G1, regressionType,power=1)
        # fit bias function using D2
        biases1 = bias_func1(X2)
        model1 = fitFCAECB(X2, biases1, regressionType, power=power,lr=lr)

        # fit short-term nuisance function and short-term confounding bias using D2
        bias_func2, _,_,_,_ = fitShortTermNuisance(X2, S2, T2, G2, regressionType,power=1)
        # fit bias function using D1
        biases2 = bias_func2(X1)
        model2 = fitFCAECB(X1, biases2, regressionType, power=power,lr=lr)
    else:
        # fit short-term nuisance function and short-term confounding bias using D
        bias_func1, _,_,_,_ = fitShortTermNuisance(X, S, T, G, regressionType,power=1)
        bias_func2 = bias_func1
        # fit bias function using D
        biases2 = biases1 = bias_func1(X)
        model2 = model1 = fitFCAECB(X, biases1, regressionType, power=power,lr=lr)

    XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0][:, None], T[G == 0][:, None]

    # fit long-term nuisance
    # print(np.squeeze(YO[TO[:,0]==1,:]).shape)
    muYO1 = regression1D(XO[TO[:,0]==1,:], np.squeeze(YO[TO[:,0]==1,:]), type=regressionType,power=1)
    muYO0 = regression1D(XO[TO[:,0]==0,:], np.squeeze(YO[TO[:,0]==0,:]), type=regressionType,power=1)
    muYO_1XO = muYO1.predict(XO)
    muYO_0XO = muYO0.predict(XO)

    # construct long-term confounding bias
    biases_XO_1 = bias_func1(XO)
    biases_XO_2 = bias_func2(XO)
    # print(biases.shape)
    bias_final_1 = 0
    bias_final_2 = 0
    short_time = biases_XO_1.shape[1]
    # print(TIME-short_time)
    for i in range(TIME-short_time):
        if i == 0:
            # bias_final = model.predict(np.concatenate((XO, bias[:, -1][:, None]), axis=1))
            bias_final_1 = model2.predict(XO) * biases_XO_1[:, -1]
            bias_final_2 = model1.predict(XO) * biases_XO_2[:, -1]
            # print(((model1.predict(XO) + model2.predict(XO))/2).shape)
            # print(biases[:, -1].shape)
        else:
            # bias_final = model.predict(np.concatenate((XO, bias_final), axis=1))
            bias_final_1 = model2.predict(XO) * bias_final_1
            bias_final_2 = model1.predict(XO) * bias_final_2
        # print('final shape')
        # print(bias_final.shape)
    # print(muYO_1XO.shape)
    # print(bias_final.shape)
    # print(muYO_1XO.shape)
    # print(bias_final.shape)
    bias_final = (bias_final_1+bias_final_2)/2
    tau = muYO_1XO - muYO_0XO + bias_final # [:,None]
    if np.isnan(np.mean(tau)):
        tau = conditionalEquiConfoundingBiasEstimator_split(X, S, Y, T, G,
                                                      TIME=TIME, regressionType=regressionType,
                                                      power=power, split=split, lr=lr/10)
    # print(muYO_1XO[:,None].shape, bias_final.shape, tau.shape)
    # print(tau.shape)
    return tau

# def conditionalEquiConfoundingBiasEstimator_split(X, S, Y, T, G, TIME=15, regressionType='linear', power=1):
#     # sample split into D1 D2
#     X1,X2, S1,S2, Y1,Y2, T1,T2, G1,G2 = train_test_split(X, S, Y, T, G, test_size = 0.5)
#
#     # devide into G=O and G=E
#     XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0][:, None], T[G == 0][:, None]
#     # XE, SE, TE = X[G == 1], S[G == 1], T[G == 1][:, None]
#
#     # fit short-term nuisance function and short-term confounding bias using D1
#     bias_func1, _,_,_,_ = fitShortTermNuisance(X1, S1, T1, G1, regressionType,power=1)
#     # fit bias function using D2
#     biases1 = bias_func1(X2)
#     model1 = fitFCAECB(X2, biases1, regressionType, power=power)
#
#     # fit short-term nuisance function and short-term confounding bias using D2
#     bias_func2, _,_,_,_ = fitShortTermNuisance(X2, S2, T2, G2, regressionType,power=1)
#     # fit bias function using D1
#     biases2 = bias_func2(X1)
#     model2 = fitFCAECB(X1, biases2, regressionType, power=power)
#
#     # fit long-term nuisance
#     # print(np.squeeze(YO[TO[:,0]==1,:]).shape)
#     muYO1 = regression1D(XO[TO[:,0]==1,:], np.squeeze(YO[TO[:,0]==1,:]), type=regressionType,power=1)
#     muYO0 = regression1D(XO[TO[:,0]==0,:], np.squeeze(YO[TO[:,0]==0,:]), type=regressionType,power=1)
#     muYO_1XO = muYO1.predict(XO)
#     muYO_0XO = muYO0.predict(XO)
#
#     # construct long-term confounding bias
#     biases = (bias_func1(XO) + bias_func2(XO))/2
#     bias_final = 0
#     sample_size, short_time = biases.shape[0], biases.shape[1]
#     # print(TIME-short_time)
#     for i in range(TIME-short_time):
#         if i == 0:
#             # bias_final = model.predict(np.concatenate((XO, bias[:, -1][:, None]), axis=1))
#             bias_final = (model1.predict(XO) + model2.predict(XO))/2 * biases[:, -1]
#         else:
#             # bias_final = model.predict(np.concatenate((XO, bias_final), axis=1))
#             bias_final = (model1.predict(XO) + model2.predict(XO))/2 * bias_final
#
#     # print(muYO_1XO.shape)
#     # print(bias_final.shape)
#     tau = muYO_1XO - muYO_0XO + bias_final # [:,None]
#     # print(muYO_1XO[:,None].shape, bias_final.shape, tau.shape)
#     # print(tau.shape)
#     return tau

def test_optimal(regressionType='linear',sizeE=2000, sizeO=4000):
    from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds

    ite_obs, ate_obs = [],[]
    ite_exp, ate_exp = [],[]
    ite_66, ate_66 = [],[]
    ite_33, ate_33 = [],[]
    ite_22, ate_22 = [],[]
    ite_21, ate_21 = [],[]
    ite_11, ate_11 = [],[]
    for seed in range(100):
        TIME, time = 13, 7
        X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(size_E=sizeE, size_O=sizeO, seed=seed, time=time,TIME=TIME)

        X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
            np.concatenate((YO, YE), 0), \
            np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()


        # print('ground truth ATE: ' + str(np.mean(iteO)))

        set_all_seeds(seed)
        ite, _, _ = t_learner_estimator(XO, TO, YO, type=regressionType)
        # print('t_learner using obs data: ' + str(PEHE_ITE(iteO, ite))+ ' ' + str(RMSE_ATE(ite, iteO)))
        ite_obs.append(PEHE_ITE(iteO, ite))
        ate_obs.append(RMSE_ATE(iteO, ite))

        set_all_seeds(seed)
        _, r1, r2 = t_learner_estimator(XE, TE, YE, type=regressionType)
        ite_tlearner = r1.predict(XO) - r2.predict(XO)
        # print('t_learner using exp data: ' + str(PEHE_ITE(iteO, ite_tlearner)) + ' ' + str(RMSE_ATE(ite_tlearner, iteO)))
        ite_exp.append(PEHE_ITE(iteO, ite_tlearner))
        ate_exp.append(RMSE_ATE(iteO, ite_tlearner))

        # optimally choose time gap
        S_ = S
        TIME = TIME
        # 6/sqrt(6)   280 69
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(X, S_, Y, T, G, regressionType=regressionType, TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_66.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_66.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,2][:,None],S[:,4][:,None],S[:,6][:,None]), 1)
        TIME=7
        # 3/ sqrt(3)  333 131
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
            X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=1)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_33.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_33.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,3][:,None],S[:,6][:,None]), 1)
        TIME=5
        # 2/ sqrt(2) 281 70
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
            X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=2)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_22.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_22.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,4][:,None]), 1)
        TIME=4
        # 2/sqrt(1)  307 34
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(X, S_, Y, T, G, regressionType=regressionType, TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_21.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_21.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,6][:,None]), 1)
        TIME=3
        # 1/sqrt(1)  249 14
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(X, S_, Y, T, G, regressionType=regressionType, TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_11.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_11.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

    print('t_learner using obs data: ', np.mean(ite_obs), np.mean(ate_obs),  np.std(ite_obs), np.std(ate_obs))
    print('t_learner using exp data: ', np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.mean(ate_exp))
    print('ite is equi con 66=2.4, 1234567 -> 13: ', np.mean(ite_66), np.mean(ate_66), np.std(ite_66), np.std(ate_66))
    print('ite is equi con 21=2,   15      -> 13: ', np.mean(ite_21), np.mean(ate_21), np.std(ite_21), np.std(ate_21))
    print('ite is equi con 33=1.7, 1357    -> 13: ', np.mean(ite_33), np.mean(ate_33),  np.std(ite_33), np.std(ate_33))
    print('ite is equi con 22=1.4, 147     -> 13: ', np.mean(ite_22), np.mean(ate_22), np.std(ite_22), np.std(ate_22))
    print('ite is equi con 11=1,   17      -> 13: ', np.mean(ite_11), np.mean(ate_11), np.std(ite_11), np.std(ate_11))


def test_optimal_611(regressionType='linear',sizeE=2000, sizeO=4000):
    from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds

    ite_obs, ate_obs = [],[]
    ite_exp, ate_exp = [],[]
    ite_55, ate_55 = [],[]
    ite_32, ate_32 = [],[]
    ite_21, ate_21 = [],[]
    ite_11, ate_11 = [],[]
    for seed in range(30):
        TIME, time = 11, 6
        X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(size_O=sizeO, size_E=sizeE, seed=seed, time=time,TIME=TIME)

        X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
            np.concatenate((YO, YE), 0), \
            np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()


        # print('ground truth ATE: ' + str(np.mean(iteO)))

        set_all_seeds(seed)
        ite, _, _ = t_learner_estimator(XO, TO, YO, type=regressionType)
        # print('t_learner using obs data: ' + str(PEHE_ITE(iteO, ite))+ ' ' + str(RMSE_ATE(ite, iteO)))
        ite_obs.append(PEHE_ITE(iteO, ite))
        ate_obs.append(RMSE_ATE(iteO, ite))

        set_all_seeds(seed)
        _, r1, r2 = t_learner_estimator(XE, TE, YE, type=regressionType)
        ite_tlearner = r1.predict(XO) - r2.predict(XO)
        # print('t_learner using exp data: ' + str(PEHE_ITE(iteO, ite_tlearner)) + ' ' + str(RMSE_ATE(ite_tlearner, iteO)))
        ite_exp.append(PEHE_ITE(iteO, ite_tlearner))
        ate_exp.append(RMSE_ATE(iteO, ite_tlearner))

        # optimally choose time gap
        S_ = S
        TIME = TIME
        # 6/sqrt(6)   280 69
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(X, S_, Y, T, G, regressionType=regressionType, TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_55.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_55.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,2][:,None],S[:,4][:,None]), 1)
        TIME=6
        # 3/ sqrt(3)  333 131
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(X, S_, Y, T, G, regressionType=regressionType, TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_32.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_32.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,3][:,None]), 1)
        TIME=4
        # 2/ sqrt(2) 281 70
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(X, S_, Y, T, G, regressionType=regressionType, TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_21.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_21.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,5][:,None]), 1)
        TIME=3
        # 2/sqrt(1)  307 34
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(X, S_, Y, T, G, regressionType=regressionType, TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_11.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_11.append(RMSE_ATE(iteO[:, None], ite_cecb_T))


    print('t_learner using obs data: ', np.mean(ite_obs), np.mean(ate_obs),  np.std(ite_obs), np.std(ate_obs))
    print('t_learner using exp data: ', np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.mean(ate_exp))
    print('ite is equi con 55=2.4,  123456 -> 11 : ', np.mean(ite_55), np.mean(ate_55), np.std(ite_55), np.std(ate_55))
    print('ite is equi con 32=2,    135    -> 11 : ', np.mean(ite_32), np.mean(ate_32), np.std(ite_32), np.std(ate_32))
    print('ite is equi con 21=1.7,  14     -> 11 : ', np.mean(ite_21), np.mean(ate_21),  np.std(ite_21), np.std(ate_21))
    print('ite is equi con 11=1,    16     -> 11 : ', np.mean(ite_11), np.mean(ate_11), np.std(ite_11), np.std(ate_11))



def test_optimal_69(regressionType='linear',sizeE=2000, sizeO=4000
    ,split = True, ifprint=True):
    from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds, dgp_ts_equ_conf_design_x,dgp_ts_equ_conf_design_ex

    ite_obs, ate_obs = [],[]
    ite_exp, ate_exp = [],[]
    ite_35, ate_35 = [],[]
    ite_22, ate_22 = [],[]
    ite_11, ate_11 = [],[]
    ite_31, ate_31 = [],[]
    ite_21, ate_21 = [],[]
    # for seed in range(50):
    for seed in [39]:
        TIME, time = 9, 6
        X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design_ex(size_E=sizeE, size_O=sizeO, seed=seed, time=time,TIME=TIME)
            # XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design_x(size_E=sizeE, size_O=sizeO, seed=seed, time=time,TIME=TIME)
        # XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(size_E=sizeE, size_O=sizeO, seed=seed, time=time,TIME=TIME)

        X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
            np.concatenate((YO, YE), 0), \
            np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()


        # print('ground truth ATE: ' + str(np.mean(iteO)))

        set_all_seeds(seed)
        ite, _, _ = t_learner_estimator(XO, TO, YO, type=regressionType, power=1)
        # ite, _, _ = t_learner_estimator(XO, TO, YO, type='Poly', power=1)
        if ifprint:
            print('t_learner using obs data: ' + str(PEHE_ITE(iteO, ite))+ ' ' + str(RMSE_ATE(ite, iteO)))
        ite_obs.append(PEHE_ITE(iteO, ite))
        ate_obs.append(RMSE_ATE(iteO, ite))

        set_all_seeds(seed)
        _, r1, r2 = t_learner_estimator(XE, TE, YE, type=regressionType, power=1)
        ite_tlearner = r1.predict(XO) - r2.predict(XO)
        if ifprint:
            print('t_learner using exp data: ' + str(PEHE_ITE(iteO, ite_tlearner)) + ' ' + str(RMSE_ATE(ite_tlearner, iteO)))
        ite_exp.append(PEHE_ITE(iteO, ite_tlearner))
        ate_exp.append(RMSE_ATE(iteO, ite_tlearner))

        # optimally choose time gap
        # S_ = S
        # TIME = TIME
        # # 6/sqrt(6)   280 69
        # set_all_seeds(seed)
        # ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
        #     X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=1, split=split)
        # # print(ite_cecb_T.shape, iteO.shape)
        # if ifprint:
        #     print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        # ite_35.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        # ate_35.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
        #
        # S_ = np.concatenate((S[:,0][:,None],S[:,2][:,None],S[:,4][:,None]), 1)
        # TIME=5
        # # 3/ sqrt(3)  333 131
        # set_all_seeds(seed)
        # ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
        #     X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=1, split=split)
        # # print(ite_cecb_T.shape, iteO.shape)
        # if ifprint:
        #     print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        # ite_22.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        # ate_22.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,4][:,None]), 1)
        TIME=3
        # 2/ sqrt(2) 281 70
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
            X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=1, split=split)
        # print(ite_cecb_T.shape, iteO.shape)
        if ifprint:
            print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
            RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_11.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_11.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,2][:,None]), 1)
        TIME=5
        # 2/sqrt(1)  307 34
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
            X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=1, split=split)
        # print(ite_cecb_T.shape, iteO.shape)
        if ifprint:
            print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
            RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_31.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_31.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,2][:,None],S[:,4][:,None]), 1)
        TIME=4
        # 2/sqrt(1)  307 34
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
            X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=1, split=split)
        # print(ite_cecb_T.shape, iteO.shape)
        if ifprint:
            print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
            RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_21.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_21.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

    print('t_learner using obs data: ', np.mean(ite_obs), np.mean(ate_obs),  np.std(ite_obs), np.std(ate_obs))
    print('t_learner using exp data: ', np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.std(ate_exp))
    print('ite is equi con 13    -> 9: ', np.mean(ite_31), np.mean(ate_31),  np.std(ite_31), np.std(ate_31))
    print('ite is equi con 35    -> 9: ', np.mean(ite_21), np.mean(ate_21),  np.std(ite_21), np.std(ate_21))
    print('ite is equi con 135   -> 9: ', np.mean(ite_22), np.mean(ate_22), np.std(ite_22), np.std(ate_22))
    print('ite is equi con 123456-> 9: ', np.mean(ite_35), np.mean(ate_35), np.std(ite_35), np.std(ate_35))
    print('ite is equi con 15    -> 9: ', np.mean(ite_11), np.mean(ate_11), np.std(ite_11), np.std(ate_11))


# def test_vary(TIME=9, time=6, regressionType='linear'):
#     from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds, dgp_ts_equ_conf_design_linear
#
#     ite_obs, ate_obs = [],[]
#     ite_exp, ate_exp = [],[]
#     ite_our, ate_our = [],[]
#     for seed in range(50):
#         X, T, S, Y, G, \
#             XO, TO, SO, YO, iteO, \
#             XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design( sizeE=2000, sizeO=4000, seed=seed, time=time,TIME=TIME)
#
#         X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
#             np.concatenate((YO, YE), 0), \
#             np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()
#
#
#         # print('ground truth ATE: ' + str(np.mean(iteO)))
#
#         set_all_seeds(seed)
#         ite, _, _ = t_learner_estimator(XO, TO, YO, type=regressionType, power=1)
#         # print('t_learner using obs data: ' + str(PEHE_ITE(iteO, ite))+ ' ' + str(RMSE_ATE(ite, iteO)))
#         ite_obs.append(PEHE_ITE(iteO, ite))
#         ate_obs.append(RMSE_ATE(iteO, ite))
#
#         set_all_seeds(seed)
#         _, r1, r2 = t_learner_estimator(XE, TE, YE, type=regressionType, power=1)
#         ite_tlearner = r1.predict(XO) - r2.predict(XO)
#         # print('t_learner using exp data: ' + str(PEHE_ITE(iteO, ite_tlearner)) + ' ' + str(RMSE_ATE(ite_tlearner, iteO)))
#         ite_exp.append(PEHE_ITE(iteO, ite_tlearner))
#         ate_exp.append(RMSE_ATE(iteO, ite_tlearner))
#
#         # optimally choose time gap
#         S_ = S
#         TIME = TIME
#         # 6/sqrt(6)   280 69
#         set_all_seeds(seed)
#         ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
#             X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=1)
#         # print(ite_cecb_T.shape, iteO.shape)
#         # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
#         #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
#         ite_our.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
#         ate_our.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
#
#
#     print('t_learner using obs data: ', np.mean(ite_obs), np.mean(ate_obs),  np.std(ite_obs), np.std(ate_obs))
#     print('t_learner using exp data: ', np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.std(ate_exp))
#     print('ite is equi con 31: ', np.mean(ite_our), np.mean(ate_our),  np.std(ite_our), np.std(ate_our))


def test_vary(TIME=9, time=6, sizeE=2000, sizeO=4000, regressionType='linear'):
    from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds, dgp_ts_equ_conf_design_linear

    ite_obs, ate_obs = [],[]
    ite_exp, ate_exp = [],[]
    ite_our, ate_our = [],[]
    for seed in range(50):
        X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(sizeE, sizeO, seed=seed, time=time,TIME=TIME)

        print(XO.shape, TO.shape, SO.shape, YO.shape, iteO.shape)

        X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
            np.concatenate((YO, YE), 0), \
            np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

        # print('ground truth ATE: ' + str(np.mean(iteO)))

        set_all_seeds(seed)
        ite, _, _ = t_learner_estimator(XO, TO, YO, type=regressionType, power=1)
        # print('t_learner using obs data: ' + str(PEHE_ITE(iteO, ite))+ ' ' + str(RMSE_ATE(ite, iteO)))
        ite_obs.append(PEHE_ITE(iteO, ite))
        ate_obs.append(RMSE_ATE(iteO, ite))

        set_all_seeds(seed)
        _, r1, r2 = t_learner_estimator(XE, TE, YE, type=regressionType, power=1)
        ite_tlearner = r1.predict(XO) - r2.predict(XO)
        # print('t_learner using exp data: ' + str(PEHE_ITE(iteO, ite_tlearner)) + ' ' + str(RMSE_ATE(ite_tlearner, iteO)))
        ite_exp.append(PEHE_ITE(iteO, ite_tlearner))
        ate_exp.append(RMSE_ATE(iteO, ite_tlearner))

        # optimally choose time gap
        S_ = S
        TIME = TIME
        # 6/sqrt(6)   280 69
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
            X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=1)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_our.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_our.append(RMSE_ATE(iteO[:, None], ite_cecb_T))


    print('t_learner using obs data: ', np.mean(ite_obs), np.mean(ate_obs),  np.std(ite_obs), np.std(ate_obs))
    print('t_learner using exp data: ', np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.std(ate_exp))
    print('ite is equi con 31: ', np.mean(ite_our), np.mean(ate_our),  np.std(ite_our), np.std(ate_our))


def run_tss(XO, TO, SO, YO, XE, TE, SE, YE,
            TIME, iteO, repeat_run=10,
            regressionType='linear', split=True):
    # 对外开放的接口，用于调用模型
    # repeat_run: 跑多少次
    # iteO: observational data 的 individual effect
    # regressionType:  linear kernelRidge
    # shape requirement: numpy
    # XO: (sample size, dim)
    # TO: (sample size, )
    # SO: (sample size, time)  e.g., 短期包含3个时间刻，time = 3
    # YO: (sample size, )
    # ite0: (sample size, )
    # TIME:
    ite_obs, ate_obs = [],[]
    ite_exp, ate_exp = [],[]
    ite_our, ate_our = [],[]
    from utils.ts_eq_dgp import set_all_seeds

    for seed in range(repeat_run):
        X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0),\
            np.concatenate((YO, YE), 0), np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

        # run T-learner using obs data
        set_all_seeds(seed)
        ite, _, _ = t_learner_estimator(XO, TO, YO, type=regressionType, power=1)
        # print('t_learner using obs data: ' + str(PEHE_ITE(iteO, ite))+ ' ' + str(RMSE_ATE(ite, iteO)))
        ite_obs.append(PEHE_ITE(iteO, ite))
        ate_obs.append(RMSE_ATE(iteO, ite))

        # run T-learner using exp data
        set_all_seeds(seed)
        _, r1, r2 = t_learner_estimator(XE, TE, YE, type=regressionType, power=1)
        ite_tlearner = r1.predict(XO) - r2.predict(XO)
        # print('t_learner using exp data: ' + str(PEHE_ITE(iteO, ite_tlearner)) + ' ' + str(RMSE_ATE(ite_tlearner, iteO)))
        ite_exp.append(PEHE_ITE(iteO, ite_tlearner))
        ate_exp.append(RMSE_ATE(iteO, ite_tlearner))

        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
                X, S, Y, T, G, regressionType=regressionType, TIME=TIME, power=1, split=split)

        ite_our.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_our.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

    # return: pehe_mean, pehe_std, mae_mean, mae_std


    print('t_learner using obs data: ', np.mean(ite_obs), np.mean(ate_obs),  np.std(ite_obs), np.std(ate_obs))
    print('t_learner using exp data: ', np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.std(ate_exp))
    print('ite is equi con 31: ', np.mean(ite_our), np.mean(ate_our),  np.std(ite_our), np.std(ate_our))

    return np.mean(ite_obs), np.mean(ate_obs),  np.std(ite_obs), np.std(ate_obs), \
        np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.std(ate_exp), \
        np.mean(ite_our), np.mean(ate_our),  np.std(ite_our), np.std(ate_our)



if __name__ == '__main__':
    import warnings

    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_fitting', type=int, default=0, help='seed all set to 123 for all methods, do NOT change')
    parser.add_argument('--dataset', type=str, default='NEWS')
    parser.add_argument('--regtype', type=str, default='kernelRidge') # linear kernelRidge
    parser.add_argument('--sizeE', type=int, default=2000, help='sample size of experimental data')
    parser.add_argument('--sizeO', type=int, default=4000, help='sample size of observational data')

    args = parser.parse_args()

    warnings.simplefilter('ignore')
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    np.random.seed(123)
    # test_optimal_69(args.regtype, sizeE=args.sizeE, sizeO=args.sizeO, split=True)
    test_vary()



    # test_optimal('linear', sizeE=2000, sizeO=4000)
    # test_optimal_611('linear', sizeE=2000, sizeO=4000)

    # fix time=6, TIME= 7 8 9 10 11 12
    # for T in [7,8,9,10,11,12]:
    #     print('t=6, T='+str(T))
    #     test_vary(time=6, TIME=T, sizeE=2000, sizeO=4000, regressionType='linear')
    #
    #
    # # fix TIME= 9 , time=45678
    # for t in [4,5,6,7,8]:
    #     print('t='+str(t)+', T=9')
    #     test_vary(time=t, TIME=9, sizeE=2000, sizeO=4000, regressionType='linear')

    # for i in [1,2,3,4,5]:
    #     sizeE= int(1000*i)
    #     sizeO= int(2000*i)
    #     print(sizeO, sizeE)
    #     test_vary(time=6, TIME=9, sizeE=sizeE, sizeO=sizeO, regressionType='linear')
    # test_vary(time=6, TIME=7, sizeE=2000, sizeO=4000, regressionType='linear')
