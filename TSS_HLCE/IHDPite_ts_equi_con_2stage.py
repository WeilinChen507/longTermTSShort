from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

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

def fitFCAECB(X, biases, regressionType='linear', power=1):
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
    input = np.repeat(X, repeats=short_time - 1, axis=0)
    output = bias_train[:, 1] / bias_train[:, 0]

    # fitting
    # model = regression1D(input, output, type=regressionType, power=power)
    from utils.basic_model import TrainedMLPLinear
    model = TrainedMLPLinear(input_dim=X.shape[1], output_dim=1, device='cuda').fit(\
        X=X_train, Y=bias_train[:, 1][:,None], factor=bias_train[:, 0][:,None], epochs=200)
    # print(model.coef_)
    return model


def conditionalEquiConfoundingBiasEstimator_split(X, S, Y, T, G, TIME=15, regressionType='linear', power=1):
    # sample split into D1 D2
    X1,X2, S1,S2, Y1,Y2, T1,T2, G1,G2 = train_test_split(X, S, Y, T, G, test_size = 0.5)

    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0][:, None], T[G == 0][:, None]
    # XE, SE, TE = X[G == 1], S[G == 1], T[G == 1][:, None]

    # fit short-term nuisance function and short-term confounding bias using D1
    bias_func1, _,_,_,_ = fitShortTermNuisance(X1, S1, T1, G1, regressionType,power=1)
    # fit bias function using D2
    biases1 = bias_func1(X2)
    model1 = fitFCAECB(X2, biases1, regressionType, power=power)

    # fit short-term nuisance function and short-term confounding bias using D2
    bias_func2, _,_,_,_ = fitShortTermNuisance(X2, S2, T2, G2, regressionType,power=1)
    # fit bias function using D1
    biases2 = bias_func2(X1)
    model2 = fitFCAECB(X1, biases2, regressionType, power=power)

    # fit long-term nuisance
    # print(np.squeeze(YO[TO[:,0]==1,:]).shape)
    muYO1 = regression1D(XO[TO[:,0]==1,:], np.squeeze(YO[TO[:,0]==1,:]), type=regressionType,power=1)
    muYO0 = regression1D(XO[TO[:,0]==0,:], np.squeeze(YO[TO[:,0]==0,:]), type=regressionType,power=1)
    muYO_1XO = muYO1.predict(XO)
    muYO_0XO = muYO0.predict(XO)

    # construct long-term confounding bias
    biases = (bias_func1(XO) + bias_func2(XO))/2
    bias_final = 0
    sample_size, short_time = biases.shape[0], biases.shape[1]
    # print(TIME-short_time)
    for i in range(TIME-short_time):
        if i == 0:
            # bias_final = model.predict(np.concatenate((XO, bias[:, -1][:, None]), axis=1))
            # print(model1.predict(XO).shape)
            # print(biases[:, -1].shape)
            bias_final = (model1.predict(XO) + model2.predict(XO))/2 * biases[:, -1][:,None]
            # print(bias_final.shape)
        else:
            # bias_final = model.predict(np.concatenate((XO, bias_final), axis=1))
            bias_final = (model1.predict(XO) + model2.predict(XO))/2 * bias_final
            # print(bias_final.shape)


    # print(muYO_1XO.shape)
    # print(bias_final.shape)
    tau = muYO_1XO[:,None] - muYO_0XO[:,None] + bias_final # [:,None]
    # print(muYO_1XO[:,None].shape, bias_final.shape, tau.shape)
    # print(tau.shape)
    return tau

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

def IHDP(TIME=100, time=50, regressionType='linear'):

    ite_obs, ate_obs = [],[]
    ite_exp, ate_exp = [],[]
    ite_our, ate_our = [],[]
    t0 = time
    for seed in range(1,11):
        X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE= readIHDP(seed=seed, t0=t0)


        X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
            np.concatenate((YO, YE), 0), \
            np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

        iteO = iteO[:,None]

        print('ground truth ATE: ' + str(np.mean(iteO)))

        set_all_seeds(seed)
        ite, _, _ = t_learner_estimator(XO, TO, YO, type=regressionType, power=1)
        # print('t_learner using obs data: ' + str(PEHE_ITE(iteO, ite))+ ' ' + str(RMSE_ATE(ite, iteO)))
        # print(iteO.shape, ite.shape)
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
        Time = 100
        # S_ = S[:,[9,19,29,39,49]]
        # TIME=10
        # S_ = S[:,[19,39]]
        # TIME=5
        # S_ = S[:,[24,49]]
        # TIME=4
        set_all_seeds(seed)
        ite_cecb_T = conditionalEquiConfoundingBiasEstimator_split(
            X, S_, Y, T, G, regressionType=regressionType, TIME=TIME, power=1)
        # print(ite_cecb_T.shape, iteO.shape)
        # print('conditionalEquiConfoundingBiasEstimator_T: ' + str(PEHE_ITE(iteO[:, None], ite_cecb_T)) + ' ' + str(
        #     RMSE_ATE(iteO[:, None], ite_cecb_T)))
        print(PEHE_ITE(iteO, ite_cecb_T), RMSE_ATE(iteO, ite_cecb_T))
        ite_our.append(PEHE_ITE(iteO, ite_cecb_T))
        ate_our.append(RMSE_ATE(iteO, ite_cecb_T))


    print('t_learner using obs data: ', np.mean(ite_obs), np.mean(ate_obs),  np.std(ite_obs), np.std(ate_obs))
    print('t_learner using exp data: ', np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.std(ate_exp))
    print('ite ours: ', np.mean(ite_our), np.mean(ate_our),  np.std(ite_our), np.std(ate_our))



import warnings, torch
warnings.simplefilter('ignore')
torch.set_default_dtype(torch.double)
torch.manual_seed(123)
np.random.seed(123)
# test_optimal_69('linear')
# test_optimal('linear')

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
IHDP(time=50, TIME=100, regressionType='linear')
