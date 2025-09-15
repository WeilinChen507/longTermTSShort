import numpy as np
import os
import random
import torch
from sklearn.gaussian_process.kernels import Matern


def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


def sample_gp_prior(X, nu=1.5, l=1.0):
    '''
    :param X:  shape of (sample size, dim)
    :param nu:
    :param l:
    :return:  y shape of (sample size,)
    '''
    # bigger nu, smoother function
    # hard to distinguish between values of nu ≥ 7/2, i.e., smooth enough
    n = X.shape[0]
    kernel = Matern(length_scale=l, length_scale_bounds="fixed", nu=nu)
    sigma = kernel(X)
    #print(np.argwhere(np.isnan(sigma)))
    assert not np.isnan(sigma).any()
    y = np.squeeze(np.random.multivariate_normal(mean=np.zeros(n), cov=sigma, size=1, check_valid="raise"))
    return y


def dgp_ts_equ_conf_design_smooth(size_O, size_E, seed, time, TIME, nu=2):
    np.random.seed(seed)

    TO = np.random.binomial(n=1, size=size_O, p=0.6)
    TE = np.random.binomial(n=1, size=size_E, p=0.4)

    XOL, UOL = [], []
    for i in range(size_O):
        XU = np.random.multivariate_normal(mean=[(TO[i] - 0.5)*2, 0],
                                           cov=[[1, TO[i] - 0.5],
                                                [TO[i] - 0.5, 1]], size=1)
        XOL.append(XU[0, 0])
        UOL.append(XU[0, 1])
    XO, UO = np.array(XOL)[:, None], np.array(UOL)[:, None]

    XEL, UEL = [], []
    for i in range(size_E):
        XU = np.random.multivariate_normal(mean=[(TE[i] - 0.5)*2, 0],
                                           cov=[[1, 0],
                                                [0, 1]], size=1)
        XEL.append(XU[0, 0])
        UEL.append(XU[0, 1])
    XE, UE = np.array(XEL)[:, None], np.array(UEL)[:, None]

    # time = 15
    SO_1, SO_0, SO = np.zeros(shape=[size_O, TIME]), np.zeros(shape=[size_O, TIME]), np.zeros(shape=[size_O, TIME])
    SE_1, SE_0, SE = np.zeros(shape=[size_E, TIME]), np.zeros(shape=[size_E, TIME]), np.zeros(shape=[size_E, TIME])

    coef_u = .5
    f_s = sample_gp_prior()
    for t in range(TIME):
        noiseSO, noiseSE = np.random.randn(size_O) * 0.1, np.random.randn(size_E) * 0.1
        if t == 0:
            SO_1[:, t] = 1.2 + 3 * np.mean(XO, axis=1) + coef_u*np.mean(UO, axis=1) + noiseSO
            SO_0[:, t] = 1 + 1 * np.mean(XO, axis=1) + coef_u*np.mean(UO, axis=1) + noiseSO
            SE_1[:, t] = 1.2 + 3 * np.mean(XE, axis=1) + coef_u*np.mean(UE, axis=1) + noiseSE
            SE_0[:, t] = 1 + 1 * np.mean(XE, axis=1) + coef_u*np.mean(UE, axis=1) + noiseSE
        else:
            # print(t)
            # print(SO_1[:, :t].shape)
            # print(np.sum(SO_1[:, :t], axis=1).shape)
            SO_1[:, t] = 1.2 + 3 * np.mean(XO, axis=1) + coef_u*np.mean(UO, axis=1) + np.sum(SO_1[:, :t], axis=1) + noiseSO
            SO_0[:, t] = 1 + 1 * np.mean(XO, axis=1) + coef_u*np.mean(UO, axis=1) + np.sum(SO_0[:, :t], axis=1) + noiseSO
            SE_1[:, t] = 1.2 + 3 * np.mean(XE, axis=1) + coef_u*np.mean(UE, axis=1) + np.sum(SE_1[:, :t], axis=1) + noiseSE
            SE_0[:, t] = 1 + 1 * np.mean(XE, axis=1) + coef_u*np.mean(UE, axis=1) + np.sum(SE_0[:, :t], axis=1) + noiseSE

        SO[:, t] = np.where(TO == 1, SO_1[:, t], SO_0[:, t])
        SE[:, t] = np.where(TE == 1, SE_1[:, t], SE_0[:, t])
        iteO, iteE = SO_1[:, t]-SO_0[:, t], SE_1[:, t]-SE_0[:, t]

    YO, YE = SO[:,-1], SE[:,-1]
    SO, SE = SO[:,:time], SE[:,:time]

    YO, YE, SO, SE = YO.squeeze(), YE.squeeze(), SO.squeeze(), SE.squeeze()
    iteO, iteE = iteO.squeeze(), iteE.squeeze()
    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), \
        np.concatenate((SO, SE), 0), np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    return X, T, S, Y, G, \
        XO, TO, SO, YO, iteO, \
        XE, TE, SE, YE, iteE


# X, T, S, Y, G, \
#     XO, TO, SO, YO, iteO, \
#     XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(100,100,1)
# print(X.shape, T.shape, S.shape, Y.shape, G.shape)
# print(XO.shape, TO.shape, SO.shape, YO.shape, G.shape)
# print(XE.shape, TE.shape, SE.shape, YE.shape, G.shape)
# print(iteO.shape)
#
# for i in range(5):
#     dgp_equ_conf(dim_x=10, dim_u=10, size_O=2000, size_E=500, seed=i)