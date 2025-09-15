import sys
sys.path.append('..')

from utils.ite_estimator import e_x_estimator, regression1D, regressionPoly, \
    e_x_estimator_only_quadratic, e_x_estimator_quadratic, e_x_estimator_mlp, e_x_estimator_G
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split


def nuisance_Tlearner(X, S, Y, T, G,  model_specified='1111'):
    '''
            &  \mu^O_S(x,A), \mu^E_S(x,A),\mu^O_Y(x,A) ; \\  using poly 2
            &  \pi^E(x), \pi^O(x),  \pi^G(x); \\ using logistic, linear, pi^G using cosnt \frac{p(G=E)|p(G=E)+p(G=O)}
            &  \mu^E_S(x,A),  \pi^O(x); \\
            &  \pi^E(x), \mu^O_S(x,A), \mu^O_Y(x,A), \pi^G(x).
    '''
    # devide into G=O and G=E
    if model_specified == '1111':
        XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0]
        XE, SE, TE = X[G == 1], S[G == 1], T[G == 1]
        muSE1 = regressionPoly(XE[TE==1,:], SE[TE==1], degree=2)
        muSE0 = regressionPoly(XE[TE==0,:], SE[TE==0], degree=2)
        muSO1 = regressionPoly(XO[TO==1,:], SO[TO==1], degree=2)
        muSO0 = regressionPoly(XO[TO==0,:], SO[TO==0], degree=2)
        muYO1 = regressionPoly(XO[TO==1,:], YO[TO==1], degree=2)
        muYO0 = regressionPoly(XO[TO==0,:], YO[TO==0], degree=2)
        piE = e_x_estimator(XE, TE, bias=True)
        piO = e_x_estimator(XO, TO, bias=True)
        piG = XE.shape[0] / ( XE.shape[0] + XO.shape[0])
        # piG = e_x_estimator_G(X, G)
        # piG=.5
        # print(np.sum(piO.predict_proba(XO)[:, 1]>0.95)+np.sum(piO.predict_proba(XO)[:, 1]<0.05))
        # print(np.sum(piO.predict_proba(XO)[:, 1]>0.95)+np.sum(piO.predict_proba(XO)[:, 1]<0.05))
        # print(np.sum(piG.predict_proba(X)[:, 1]>0.9)+np.sum(piG.predict_proba(X)[:, 1]<0.1))
    elif model_specified == '1000':
        XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0]
        XE, SE, TE = X[G == 1], S[G == 1], T[G == 1]
        # muso muse muyo
        muSE1 = regressionPoly(XE[TE==1,:], SE[TE==1], degree=2)  # correct
        muSE0 = regressionPoly(XE[TE==0,:], SE[TE==0], degree=2)  # correct
        muSO1 = regressionPoly(XO[TO==1,:], SO[TO==1], degree=2)  # correct
        muSO0 = regressionPoly(XO[TO==0,:], SO[TO==0], degree=2)  # correct
        muYO1 = regressionPoly(XO[TO==1,:], YO[TO==1], degree=2)  # correct
        muYO0 = regressionPoly(XO[TO==0,:], YO[TO==0], degree=2)  # correct
        piE = e_x_estimator_only_quadratic(XE, TE, bias=False)
        piO = e_x_estimator_only_quadratic(XO, TO, bias=False)
        # piG = XO.shape[0] / ( XE.shape[0] + XO.shape[0])
        # piG = e_x_estimator_only_quadratic(X, G, bias=False)  # incorrect
        piG = XO.shape[0] / ( XE.shape[0] + XO.shape[0])  # incorrect
        # piG = 0.25

    elif model_specified == '0100':
        XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0]
        XE, SE, TE = X[G == 1], S[G == 1], T[G == 1]
        # piE piO piG
        muSE1 = regression1D(XE[TE==1,:], SE[TE==1], type='linear')
        muSE0 = regression1D(XE[TE==0,:], SE[TE==0], type='linear')
        muSO1 = regression1D(XO[TO==1,:], SO[TO==1], type='linear')
        muSO0 = regression1D(XO[TO==0,:], SO[TO==0], type='linear')
        muYO1 = regression1D(XO[TO==1,:], YO[TO==1], type='linear')
        muYO0 = regression1D(XO[TO==0,:], YO[TO==0], type='linear')
        piE = e_x_estimator(XE, TE, bias=True)  # correct
        piO = e_x_estimator(XO, TO, bias=True)  # correct

        # print(np.sum(piE.predict_proba(XE)[:,1]>0.95)+np.sum(piE.predict_proba(XE)[:,1]<0.05))
        piG = XE.shape[0] / ( XE.shape[0] + XO.shape[0])  # correct
        # print(piG)
        # piG = e_x_estimator_quadratic(X, G)  # correct
        # piG = e_x_estimator_G(X, G)  # correct
        # print(np.sum(piG.predict_proba(X)[:, 1]>0.9)+np.sum(piG.predict_proba(X)[:, 1]<0.1))

        # piG = 0.5  # correct
        # piG = e_x_estimator_mlp(X, G)  # correct

    elif model_specified == '0010':
        XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0]
        XE, SE, TE = X[G == 1], S[G == 1], T[G == 1]
        # muse piO
        muSE1 = regressionPoly(XE[TE==1,:], SE[TE==1], degree=2)  # correct
        muSE0 = regressionPoly(XE[TE==0,:], SE[TE==0], degree=2)  # correct
        muSO1 = regression1D(XO[TO == 1, :], SO[TO == 1], type='linear')
        muSO0 = regression1D(XO[TO == 0, :], SO[TO == 0], type='linear')
        muYO1 = regression1D(XO[TO == 1, :], YO[TO == 1], type='linear')
        muYO0 = regression1D(XO[TO == 0, :], YO[TO == 0], type='linear')

        piE = e_x_estimator_only_quadratic(XE, TE, bias=False)
        piO = e_x_estimator(XO, TO)  # correct
        # piG = XO.shape[0] / ( XE.shape[0] + XO.shape[0])
        # piG = e_x_estimator_only_quadratic(X, G, bias=False)
        piG = XO.shape[0] / ( XE.shape[0] + XO.shape[0])


    elif model_specified == '0001':
        XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0]
        XE, SE, TE = X[G == 1], S[G == 1], T[G == 1]
        # muse piO
        muSE1 = regression1D(XE[TE == 1, :], SE[TE == 1], type='linear')
        muSE0 = regression1D(XE[TE == 0, :], SE[TE == 0], type='linear')
        muSO1 = regressionPoly(XO[TO == 1, :], SO[TO == 1], degree=2)  # correct
        muSO0 = regressionPoly(XO[TO == 0, :], SO[TO == 0], degree=2)  # correct
        muYO1 = regressionPoly(XO[TO == 1, :], YO[TO == 1], degree=2)  # correct
        muYO0 = regressionPoly(XO[TO == 0, :], YO[TO == 0], degree=2)  # correct

        piE = e_x_estimator(XE, TE, bias=True)  # correct
        piO = e_x_estimator_only_quadratic(XO, TO, bias=False)
        # piG = XE.shape[0] / ( XE.shape[0] + XO.shape[0])  # correct
        # piG = e_x_estimator_G(X, G)  # correct
        piG = XE.shape[0] / ( XE.shape[0] + XO.shape[0])  # correct


    elif model_specified == '0000':
        XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0]
        XE, SE, TE = X[G == 1], S[G == 1], T[G == 1]
        # muse piO
        muSE1 = regression1D(XE[TE==1,:], SE[TE==1], type='linear')
        muSE0 = regression1D(XE[TE==0,:], SE[TE==0], type='linear')
        muSO1 = regression1D(XO[TO==1,:], SO[TO==1], type='linear')
        muSO0 = regression1D(XO[TO==0,:], SO[TO==0], type='linear')
        muYO1 = regression1D(XO[TO==1,:], YO[TO==1], type='linear')
        muYO0 = regression1D(XO[TO==0,:], YO[TO==0], type='linear')

        piE = e_x_estimator_only_quadratic(XE, TE, bias=False)
        piO = e_x_estimator_only_quadratic(XO, TO, bias=False)
        # piG = e_x_estimator_only_quadratic(X, G, bias=False)
        piG = XO.shape[0] / ( XE.shape[0] + XO.shape[0])  # incorrect


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
    # piG_X = piG.predict_proba(X)[:, 1]  # [:, None]  p(G=E|X)
    piG_X = piG  # [:, None]  p(G=E|X)
    pG = np.sum(G==0) / G.shape[0]      #            p(G=O)

    # print( piE_X.shape)
    # print( (2*T-1).shape)
    # print( ((2*T-1)*piE_X).shape)

    Y_MR = G / pG * (2*T-1)/(1-T + (2*T-1)*piE_X) * (S - muSE_X) * ( 1/piG_X -1) + \
           (1- G) / pG * ( (2*T-1)/(1-T + (2*T-1)*piO_X) * (Y - muYO_X - S + muSO_X) +
                           muYO_1X - muYO_0X + muSE_1X - muSE_0X + muSO_0X - muSO_1X )
    # print(Y_MR.shape)
    return Y_MR


def pseudo_reg(Y_MR, X, model_specified='1111'):
    cate = regressionPoly(X, Y_MR,  degree=2)
    return cate

def MR_LTHE(X, S, Y, T, G, predict_X, cross_fitting=True, learner='T', model_specified='1111'):
    # construct D1 and D2
    X,X2, S,S2, Y,Y2, T,T2, G,G2 = train_test_split(X, S, Y, T, G, test_size = 0.5)

    # using D1 construct nuisances
    # muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG
    muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG = nuisance_Tlearner(X, S, Y, T, G, model_specified=model_specified)

    # using D2 construct pseudo Y
    Y_MR = construct_YMR(muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG,
                  X2, S2, Y2, T2, G2)
    # cate estimator
    cate = pseudo_reg(Y_MR, X2,  model_specified=model_specified)
    # polyreg.coef_[2]  coef of x^2
    # polyreg.coef_[1]  coef of x
    # polyreg.intercept_  intercept
    # est = cate._final_estimator
    # print(est.coef_[2], est.coef_[1], est.intercept_ )
    cate_predict = cate.predict(predict_X)

    # print(cate_1.shape)
    if cross_fitting is True:
        muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG = nuisance_Tlearner(X2, S2, Y2, T2, G2,  model_specified=model_specified)

        # using D construct pseudo Y
        Y_MR = construct_YMR(muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG,
                      X, S, Y, T, G)
        # cate estimator
        cate = pseudo_reg(Y_MR, X,  model_specified=model_specified)
        cate_2 = cate.predict(predict_X)
        cate_predict = (cate_predict + cate_2)/2

    return cate_predict

def naive_LTHE(X, S, Y, T, G, predict_X, model_specified='1111'):
    muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG = nuisance_Tlearner(X, S, Y, T, G, model_specified=model_specified)

    muYO_1X = muYO1.predict(predict_X)
    muYO_0X = muYO0.predict(predict_X)
    muSE_1X = muSE1.predict(predict_X)
    muSE_0X = muSE0.predict(predict_X)
    muSO_1X = muSO1.predict(predict_X)
    muSO_0X = muSO0.predict(predict_X)
    return muYO_1X - muYO_0X + muSE_1X - muSE_0X + muSO_0X - muSO_1X

def naive_est_dgp(seed=0, model_specified='1'):
    from utils.dgp import dgp_equ_conf, dgp_equ_conf_design
    X, T, S, Y, G, \
        XO, TO, SO, YO, iteO, \
        XE, TE, SE, YE, iteE = \
        dgp_equ_conf_design(size_O=1000, size_E=500, seed=seed)
    if model_specified == '1':
        ite_naive = naive_LTHE(X, S, Y, T, G, predict_X=XO, model_specified='1111')
    else:
        ite_naive = naive_LTHE(X, S, Y, T, G, predict_X=XO, model_specified='0000')
    # print('naive_LTCE: ' + str(PEHE_ITE(ite_naive, iteO)) + ' ' + str(RMSE_ATE(ite_naive, iteO)))
    return PEHE_ITE(ite_naive, iteO), RMSE_ATE(ite_naive, iteO)

def test_dgp(seed=0, model_specified='1111'):
    from utils.dgp import dgp_equ_conf_design
    X, T, S, Y, G, \
        XO, TO, SO, YO, iteO, \
        XE, TE, SE, YE, iteE = \
    dgp_equ_conf_design(size_O=15000, size_E=10000, seed=seed)


    print('ground truth ATE: ' + str(np.mean(iteO)))

    ite_mr = MR_LTHE(X, S, Y, T, G, predict_X=XO, cross_fitting=True,  model_specified=model_specified)
    # print(ite_mr.shape, iteO.shape)
    print('MR_LTCE: ' + str(PEHE_ITE(ite_mr, iteO)) + ' ' + str(RMSE_ATE(ite_mr, iteO)))
    return PEHE_ITE(ite_mr, iteO), RMSE_ATE(ite_mr, iteO)

import warnings 
warnings.simplefilter('ignore')
c_a, c1, c2, c3, c4, c5, naive, naive_mis = [],[],[],[],[],[], [],[]
c_a_ate, c1_ate, c2_ate, c3_ate, c4_ate, c5_ate, naive_ate, naive_mis_ate = [],[],[],[],[],[],[],[]
for i in range(10):
    print('seed' + str(i))
    error_ite, error_ate = test_dgp(i,model_specified='1111')
    c_a.append(error_ite)
    c_a_ate.append(error_ate)
    error_ite, error_ate = test_dgp(i,model_specified='1000')
    c1.append(error_ite)
    c1_ate.append(error_ate)
    error_ite, error_ate = test_dgp(i,model_specified='0100')
    c2.append(error_ite)
    c2_ate.append(error_ate)
    error_ite, error_ate = test_dgp(i,model_specified='0010')
    c3.append(error_ite)
    c3_ate.append(error_ate)
    error_ite, error_ate = test_dgp(i,model_specified='0001')
    c4.append(error_ite)
    c4_ate.append(error_ate)
    error_ite, error_ate = test_dgp(i,model_specified='0000')
    c5.append(error_ite)
    c5_ate.append(error_ate)
    # error_ite, error_ate = naive_est_dgp(i, model_specified='1')
    # naive.append(error_ite)
    # naive_ate.append(error_ate)
    # error_ite, error_ate = naive_est_dgp(i, model_specified='0')
    # naive_mis.append(error_ite)
    # naive_mis_ate.append(error_ate)

print('== ite result==')
print(np.mean(c_a),np.std(c_a))
print(np.mean(c1),np.std(c1))
print(np.mean(c2),np.std(c2))
print(np.mean(c3),np.std(c3))
print(np.mean(c4),np.std(c4))
print(np.mean(c5),np.std(c5))
print(np.mean(naive),np.std(naive))
print(np.mean(naive_mis),np.std(naive_mis))
print('== ate result==')
print(np.mean(c_a_ate),np.std(c_a))
print(np.mean(c1_ate),np.std(c1))
print(np.mean(c2_ate),np.std(c2))
print(np.mean(c3_ate),np.std(c3))
print(np.mean(c4_ate),np.std(c4))
print(np.mean(c5_ate),np.std(c5))
print(np.mean(naive_ate),np.std(naive))
print(np.mean(naive_mis_ate),np.std(naive_mis))

from utils.save import save_pkl, read_pkl
data = {
    'call_mean': np.mean(c_a),
    'call_std': np.std(c_a),
    'call_ate_mean': np.mean(c_a_ate),
    'call_ate_std': np.std(c_a_ate),
    'c1_mean': np.mean(c1),
    'c1_std': np.std(c1),
    'c1_ate_mean': np.mean(c1_ate),
    'c1_ate_std': np.std(c1_ate),
    'c2_mean': np.mean(c2),
    'c2_std': np.std(c2),
    'c2_ate_mean': np.mean(c2_ate),
    'c2_ate_std': np.std(c2_ate),
    'c3_mean': np.mean(c3),
    'c3_std': np.std(c3),
    'c3_ate_mean': np.mean(c3_ate),
    'c3_ate_std': np.std(c3_ate),
    'c4_mean': np.mean(c4),
    'c4_std': np.std(c4),
    'c4_ate_mean': np.mean(c4_ate),
    'c4_ate_std': np.std(c4_ate),
    'c5_mean': np.mean(c5),
    'c5_std': np.std(c5),
    'c5_ate_mean': np.mean(c5_ate),
    'c5_ate_std': np.std(c5_ate),
    'naive_mean': np.mean(naive),
    'naive_std': np.std(naive),
    'naive_ate_mean': np.mean(naive_ate),
    'naive_ate_std': np.std(naive_ate),
    'naive_mis_mean': np.mean(naive_mis),
    'naive_mis_std': np.std(naive_mis),
    'naive_mis_ate_mean': np.mean(naive_mis_ate),
    'naive_mis_ate_std': np.mean(naive_mis_ate),
    'c_a':c_a,
    'c_a_ate':c_a_ate,
    'c1':c1,
    'c1_ate':c1_ate,
    'c2':c2,
    'c2_ate':c2_ate,
    'c3':c3,
    'c3_ate':c3_ate,
    'c4':c4,
    'c4_ate':c4_ate,
    'c5':c5,
    'c5_ate':c5_ate,
    'naive':naive,
    'naive_ate':naive_ate,
    'naive_mis':naive_mis,
    'naive_mis_ate':naive_mis_ate,
}
save_pkl('../results/mr_ite_mis.pkl',data)

d = read_pkl('../results/mr_ite_mis.pkl')
print(d)
#CUDA_VISIBLE_DEVICES=2