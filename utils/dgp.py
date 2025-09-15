import numpy as np
import os
import random
import torch

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


# should be checked
def dgp_equ_conf(dim_x, dim_u, size_O, size_E, seed):
    np.random.seed(seed)

    # XO XE should be slightly different, but UO and UE should be totally same, i.e., X->G but not U->G
    meanXO, meanXE = np.random.normal(0, 1, size=dim_x), np.random.normal(0, 1, size=dim_x)
    covX = np.eye(dim_x)
    meanU, covU = np.random.normal(0, 1, size=dim_u), np.eye(dim_u)
    XO, UO = np.random.multivariate_normal(mean=meanXO, cov=covX, size=size_O), \
        np.random.multivariate_normal(mean=meanU, cov=covU, size=size_O)
    XE, UE = np.random.multivariate_normal(mean=meanXE, cov=covX, size=size_E), \
        np.random.multivariate_normal(mean=meanU, cov=covU, size=size_E)

    unconf_strength = 0
    x2t = np.random.normal(0, 1, size=dim_x) / dim_x
    u2t = np.random.normal(0, 1, size=dim_u) / dim_u + unconf_strength

    TO = np.random.binomial(n=1, size=size_O, p=(1 / (1 + np.exp(-(np.matmul(XO, x2t) + np.matmul(UO, u2t))))))
    TE = np.random.binomial(n=1, size=size_E, p=(1 / (1 + np.exp(-np.matmul(XE, x2t)))))

    # rule out extreme data
    # indexO = np.logical_and(((1 / (1 + np.exp(-(np.matmul(XO, x2t) + np.matmul(UO, u2t))))) < 0.95),
    #                         ((1 / (1 + np.exp(-(np.matmul(XO, x2t) + np.matmul(UO, u2t))))) > 0.05))
    # indexE = np.logical_and(((1 / (1 + np.exp(-np.matmul(XE, x2t)))) < 0.95),
    #                         ((1 / (1 + np.exp(-np.matmul(XE, x2t)))) > 0.05))
    # XO, UO = XO[indexO, :], UO[indexO, :]
    # XE, UE = XE[indexE, :], UE[indexE, :]
    # TO, TE = TO[indexO], TE[indexE]
    # size_O, size_E = TO.shape[0], TE.shape[0]
    # print('sizeO,E=' + str(size_O) + str(',') + str(size_E))

    noise_strength = 1

    noiseSO, noiseSE = np.random.randn(size_O) * noise_strength, np.random.randn(size_E) * noise_strength
    SO_1 = 2 * np.mean(XO, axis=1) + 2 * np.mean(UO, axis=1) + 3 * 1 * np.mean(XO, axis=1) + 2 + noiseSO
    SO_0 = 2 * np.mean(XO, axis=1) + 2 * np.mean(UO, axis=1) + 3 * 0 * np.mean(XO, axis=1) + 0 + noiseSO
    SE_1 = 2 * np.mean(XE, axis=1) + 2 * np.mean(UE, axis=1) + 3 * 1 * np.mean(XE, axis=1) + 2 + noiseSE
    SE_0 = 2 * np.mean(XE, axis=1) + 2 * np.mean(UE, axis=1) + 3 * 0 * np.mean(XE, axis=1) + 0 + noiseSE

    SO = np.where(TO == 1, SO_1, SO_0)
    SE = np.where(TE == 1, SE_1, SE_0)

    noiseO, noiseE = np.random.randn(size_O) * noise_strength, np.random.randn(size_E) * noise_strength
    YO_1 = np.mean(XO, axis=1) + 2 * np.mean(UO, axis=1) + SO_1 + 2 * 1 * np.mean(np.power(XO, 2), axis=1) + 1 + noiseO
    YO_0 = np.mean(XO, axis=1) + 2 * np.mean(UO, axis=1) + SO_0 + 2 * 0 * np.mean(np.power(XO, 2), axis=1) + 0 + noiseO
    YE_1 = np.mean(XE, axis=1) + 2 * np.mean(UE, axis=1) + SE_1 + 2 * 1 * np.mean(np.power(XE, 2), axis=1) + 1 + noiseE
    YE_0 = np.mean(XE, axis=1) + 2 * np.mean(UE, axis=1) + SE_0 + 2 * 0 * np.mean(np.power(XE, 2), axis=1) + 0 + noiseE

    YO = np.where(TO == 1, YO_1, YO_0)
    YE = np.where(TE == 1, YE_1, YE_0)

    iteO, iteE = YO_1 - YO_0, YE_1 - YE_0

    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
        np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    return X, T, S, Y, G, \
        XO, TO, SO, YO, iteO, \
        XE, TE, SE, YE, iteE


def dgp_equ_conf_design(size_O, size_E, seed):
    np.random.seed(seed)

    TO = np.random.binomial(n=1, size=size_O, p=0.5)
    TE = np.random.binomial(n=1, size=size_E, p=0.5)

    XOL, UOL = [], []
    for i in range(size_O):
        XU = np.random.multivariate_normal(mean=[(0.5-TO[i])*0.5, 0],
                                           cov=[[1, TO[i] - 0.5],
                                                [TO[i] - 0.5, 1]], size=1)
        XOL.append(XU[0, 0])
        UOL.append(XU[0, 1])
    XO, UO = np.array(XOL)[:, None], np.array(UOL)[:, None]

    XEL, UEL = [], []
    for i in range(size_E):
        XU = np.random.multivariate_normal(mean=[(TE[i]-0.5)*0.5, 0],
                                           cov=[[1, 0],
                                                [0, 1]], size=1)
        XEL.append(XU[0, 0])
        UEL.append(XU[0, 1])
    XE, UE = np.array(XEL)[:, None], np.array(UEL)[:, None]

    # XE = np.random.normal(loc=TE[:, None], scale=1)#[:, None]
    # UE = np.random.normal(loc=0, scale=1, size=[size_E])[:, None]

    # prb_e = 1 / (1 + 2/3 * np.exp(2*XE))
    # print(np.sum(prb_e>0.95) + np.sum(prb_e< 0.05) )
    # prb_o = 1 / (1 + 2/3 * np.exp(-2*XO))
    # print(np.sum(prb_o>0.95) + np.sum(prb_o< 0.05) )

    # S=1+T+X+2×A×X+0.5X\power{2}+1×T×X\power{2}+U+0.5𝜖\index{S}
    niose_strength=.5
    noiseO, noiseE = niose_strength*np.random.randn(size_O), niose_strength*np.random.randn(size_E)
    SO = 1 + TO[:,None] + XO + 2 * TO[:,None] * XO + 0.5 * np.power(XO,2) + TO[:,None] * np.power(XO,2) + 1*UO + noiseO[:,None]
    SE = 1 + TE[:,None] + XE + 2 * TE[:,None] * XE + 0.5 * np.power(XE,2) + TE[:,None] * np.power(XE,2) + 1*UE + noiseE[:,None]
    # 𝜏\index{S}=1X\power{2}+2X+1
    # iteSO, iteSE = 2* XO + np.power(XO,2) + 1, 2* XE + np.power(XE,2) + 1

    # Y=2+3T+X+4×A×X+X\power{2}+2×T×X\power{2}+2U-S+0.5𝜖\index{Y}
    noiseO, noiseE = niose_strength*np.random.randn(size_O), niose_strength*np.random.randn(size_E)
    YO = 2 + 3 * TO[:,None] + XO + 4 * TO[:,None] * XO + np.power(XO,2) + 2 * TO[:,None] * np.power(XO,2) + 2*UO - SO + noiseO[:,None]
    YE = 2 + 3 * TE[:,None] + XE + 4 * TE[:,None] * XE + np.power(XE,2) + 2 * TE[:,None] * np.power(XE,2) + 2*UE - SE + noiseE[:,None]
    # 𝜏\index{Y}=2+2X+1X\power{2}
    iteO = 2* XO + 1*np.power(XO,2) + 2
    iteE = 2* XE + 1*np.power(XE,2) + 2


    YO, YE, SO, SE = YO.squeeze(), YE.squeeze(), SO.squeeze(), SE.squeeze()
    iteO, iteE = iteO.squeeze(), iteE.squeeze()
    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), \
        np.concatenate((SO, SE), 0), np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    return X, T, S, Y, G, \
        XO, TO, SO, YO, iteO, \
        XE, TE, SE, YE, iteE

#
# X, T, S, Y, G, \
#     XO, TO, SO, YO, iteO, \
#     XE, TE, SE, YE, iteE = dgp_equ_conf_design(100,100,1)
# print(X.shape, T.shape, S.shape, Y.shape, G.shape)
# print(XO.shape, TO.shape, SO.shape, YO.shape, G.shape)
# print(XE.shape, TE.shape, SE.shape, YE.shape, G.shape)
# print(iteO.shape)
#
# X, T, S, Y, G, \
#     XO, TO, SO, YO, iteO, \
#     XE, TE, SE, YE, iteE = dgp_equ_conf(dim_x=1, dim_u=1, size_O=100, size_E=100, seed=1)
# print(X.shape, T.shape, S.shape, Y.shape, G.shape)
# print(XO.shape, TO.shape, SO.shape, YO.shape, G.shape)
# print(XE.shape, TE.shape, SE.shape, YE.shape, G.shape)
# print(iteO.shape)

#
# for i in range(5):
#     dgp_equ_conf(dim_x=10, dim_u=10, size_O=2000, size_E=500, seed=i)