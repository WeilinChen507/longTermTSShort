from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np

def conditionalEquiConfoundingBiasEstimator_Tlearner(X, S, Y, T, G, TIME=15, regressionType='linear'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1], T[G == 1][:, None]


    muYO1 = regression1D(XO[TO[:,0]==1,:], np.squeeze(YO[TO[:,0]==1,:]), type=regressionType)
    muYO0 = regression1D(XO[TO[:,0]==0,:], np.squeeze(YO[TO[:,0]==0,:]), type=regressionType)

    muYO_1XO = muYO1.predict(XO)
    muYO_0XO = muYO0.predict(XO)

    # S shape = [sampe size, time length]
    sample_size = SO.shape[0]
    short_time = SO.shape[1]
    bias = np.zeros_like(SO)
    for i in range(short_time):
        muSE1 = regression1D(XE[TE[:,0]==1,:], SE[TE[:,0]==1, i], type=regressionType)
        muSE0 = regression1D(XE[TE[:,0]==0,:], SE[TE[:,0]==0, i], type=regressionType)
        muSO1 = regression1D(XO[TO[:,0]==1,:], SO[TO[:,0]==1, i], type=regressionType)
        muSO0 = regression1D(XO[TO[:,0]==0,:], SO[TO[:,0]==0, i], type=regressionType)

        # used prediction
        muSE_1XO = muSE1.predict(XO)
        muSE_0XO = muSE0.predict(XO)
        muSO_1XO = muSO1.predict(XO)
        muSO_0XO = muSO0.predict(XO)
        bias[:, i] = muSE_1XO - muSE_0XO + muSO_0XO - muSO_1XO
        # print('coef ' + str(i) + ' ')
        # print(muSO1.coef_)
        # print(muSO0.coef_)
        # print(muSE1.coef_)
        # print(muSE0.coef_)

    # reshape bias to fit a bias model
    bias_train = np.zeros([sample_size * (short_time-1), 2])
    # print(bias_train.shape)
    for i in range(short_time-1):
        bias_train[i*sample_size:(i+1)*sample_size,0] = bias[:,i]
        bias_train[i*sample_size:(i+1)*sample_size,1] = bias[:,i+1]

    # MLP-based
    # model = mlp_estimator(XO.shape[1]+1, 1).cuda()
    # model.fit(np.concatenate((np.repeat(XO, repeats=short_time-1, axis=0), bias_train[:, 0][:, None]), axis=1), bias_train[:, 1][:, None])
    input = np.repeat(XO, repeats=short_time-1, axis=0)
    output = bias_train[:,1] / bias_train[:,0]
    # output = bias_train[:,1:] / (bias_train[:,0:-1] + 1e-6)
    model = regression1D(input, output, type=regressionType)
    # print('coef, intercept :', model.coef_, model.intercept_)

    # middle_bias = np.zeros([sample_size, 1])
    bias_final = 0
    # print(TIME-short_time)
    for i in range(TIME-short_time):
        if i == 0:
            # bias_final = model.predict(np.concatenate((XO, bias[:, -1][:, None]), axis=1))
            bias_final = model.predict(XO) * bias[:, -1]
        else:
            # bias_final = model.predict(np.concatenate((XO, bias_final), axis=1))
            bias_final = model.predict(XO) * bias_final

    # print(bias_final.shape)
    tau = muYO_1XO - muYO_0XO + bias_final #[:,None]
    # print(muYO_1XO[:,None].shape, bias_final.shape, tau.shape)

    return tau




def test_optimal():
    from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds

    ite_obs, ate_obs = [],[]
    ite_exp, ate_exp = [],[]
    ite_66, ate_66 = [],[]
    ite_33, ate_33 = [],[]
    ite_22, ate_22 = [],[]
    ite_21, ate_21 = [],[]
    ite_11, ate_11 = [],[]
    for seed in range(50):
        TIME, time = 13, 7
        X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(10000, 10000, seed=seed, time=time,TIME=TIME)

        X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
            np.concatenate((YO, YE), 0), \
            np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

        # print(S.shape)

        print('ground truth ATE: ' + str(np.mean(iteO)))

        set_all_seeds(seed)
        ite, _, _ = t_learner_estimator(XO, TO, YO, type='linear')
        # print('t_learner using obs data: ' + str(PEHE_ITE(iteO, ite))+ ' ' + str(RMSE_ATE(ite, iteO)))
        ite_obs.append(PEHE_ITE(iteO, ite))
        ate_obs.append(RMSE_ATE(iteO, ite))

        set_all_seeds(seed)
        _, r1, r2 = t_learner_estimator(XE, TE, YE, type='linear')
        ite_tlearner = r1.predict(XO) - r2.predict(XO)
        # print('t_learner using exp data: ' + str(PEHE_ITE(iteO, ite_tlearner)) + ' ' + str(RMSE_ATE(ite_tlearner, iteO)))
        ite_exp.append(PEHE_ITE(iteO, ite_tlearner))
        ate_exp.append(RMSE_ATE(iteO, ite_tlearner))

        # optimally choose time gap
        S_ = S
        TIME = TIME
        # 6/sqrt(6)   280 69
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S_, Y, T, G, regressionType='linear', TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_66.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_66.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,2][:,None],S[:,4][:,None],S[:,6][:,None]), 1)
        TIME=7
        # 3/ sqrt(3)  333 131
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S_, Y, T, G, regressionType='linear', TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_33.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_33.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,3][:,None],S[:,6][:,None]), 1)
        TIME=5
        # 2/ sqrt(2) 281 70
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S_, Y, T, G, regressionType='linear', TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_22.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_22.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,4][:,None]), 1)
        TIME=4
        # 2/sqrt(1)  307 34
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S_, Y, T, G, regressionType='linear', TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_21.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_21.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

        S_ = np.concatenate((S[:,0][:,None],S[:,6][:,None]), 1)
        TIME=3
        # 1/sqrt(1)  249 14
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S_, Y, T, G, regressionType='linear', TIME=TIME)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        ite_11.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
        ate_11.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

    print('t_learner using obs data: ', np.mean(ite_obs), np.mean(ate_obs),  np.std(ite_obs), np.std(ate_obs))
    print('t_learner using exp data: ', np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.mean(ate_exp))
    print('ite is equi con 66=2.4: ', np.mean(ite_66), np.mean(ate_66), np.std(ite_66), np.std(ate_66))
    print('ite is equi con 21=2: ', np.mean(ite_21), np.mean(ate_21), np.std(ite_21), np.std(ate_21))
    print('ite is equi con 33=1.7: ', np.mean(ite_33), np.mean(ate_33),  np.std(ite_33), np.std(ate_33))
    print('ite is equi con 22=1.4: ', np.mean(ite_22), np.mean(ate_22), np.std(ite_22), np.std(ate_22))
    print('ite is equi con 11=1: ', np.mean(ite_11), np.mean(ate_11), np.std(ite_11), np.std(ate_11))



def test_varyT_fixt():
    from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds

    ite_177, ate_177 = [],[]
    ite_157, ate_157 = [],[]
    ite_137, ate_137 = [],[]
    ite_117, ate_117 = [],[]
    ite_97, ate_97 = [],[]
    for seed in range(80):
        TIME, time = 13, 7
        for TIME in {9,11,13,15,17}:
            X, T, S, Y, G, \
                XO, TO, SO, YO, iteO, \
                XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(4000, 1000, seed=seed, time=time,TIME=TIME)

            X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
                np.concatenate((YO, YE), 0), \
                np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()
            print('ground truth ATE: ' + str(np.mean(iteO)))

            # optimally choose time gap
            S_ = S
            # TIME = TIME
            set_all_seeds(seed)
            ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S_, Y, T, G, regressionType='linear', TIME=TIME)
            if TIME == 9:
                ite_97.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_97.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
            elif TIME == 11:
                ite_117.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_117.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
            elif TIME == 13:
                ite_137.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_137.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
            elif TIME == 15:
                ite_157.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_157.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
            elif TIME == 17:
                ite_177.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_177.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

    print('ite is equi con 97: ', np.mean(ite_97), np.mean(ate_97))
    print('ite is equi con 117: ', np.mean(ite_117), np.mean(ate_117))
    print('ite is equi con 137: ', np.mean(ite_137), np.mean(ate_137))
    print('ite is equi con 157: ', np.mean(ite_157), np.mean(ate_157))
    print('ite is equi con 177: ', np.mean(ite_177), np.mean(ate_177))



def test_varyt_fixT():
    from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds

    ite_1311, ate_1311 = [],[]
    ite_139, ate_139 = [],[]
    ite_137, ate_137 = [],[]
    ite_135, ate_135 = [],[]
    ite_133, ate_133 = [],[]
    for seed in range(80):
        TIME, time = 13, 7
        for time in {3,5,7,9,11}:
        # for time in {5}:
            X, T, S, Y, G, \
                XO, TO, SO, YO, iteO, \
                XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(4000, 1000, seed=seed, time=time,TIME=TIME)

            X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
                np.concatenate((YO, YE), 0), \
                np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()
            print('ground truth ATE: ' + str(np.mean(iteO)))

            # optimally choose time gap
            S_ = S
            # TIME = TIME
            set_all_seeds(seed)
            ite_cecb_T = conditionalEquiConfoundingBiasEstimator_Tlearner(X, S_, Y, T, G, regressionType='linear', TIME=TIME)
            if time == 3:
                ite_133.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_133.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
            elif time == 5:
                ite_135.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_135.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
            elif time == 7:
                ite_137.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_137.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
            elif time == 9:
                ite_139.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_139.append(RMSE_ATE(iteO[:, None], ite_cecb_T))
            elif time == 11:
                ite_1311.append(PEHE_ITE(iteO[:, None], ite_cecb_T))
                ate_1311.append(RMSE_ATE(iteO[:, None], ite_cecb_T))

    print('ite is equi con 133: ', np.mean(ite_133), np.mean(ate_133))
    print('ite is equi con 135: ', np.mean(ite_135), np.mean(ate_135))
    print('ite is equi con 137: ', np.mean(ite_137), np.mean(ate_137))
    print('ite is equi con 139: ', np.mean(ite_139), np.mean(ate_139))
    print('ite is equi con 1311: ', np.mean(ite_1311), np.mean(ate_1311))

import warnings, torch
warnings.simplefilter('ignore')
torch.set_default_dtype(torch.double)
torch.manual_seed(123)
np.random.seed(123)
test_optimal()
# test_varyT_fixt()
# test_varyt_fixT()
# test2()