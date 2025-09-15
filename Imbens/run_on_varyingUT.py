from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split
from Imbens.Ridge import outcome_ridge_bridge, selection_ridge_bridge

from decimal import Decimal, ROUND_HALF_UP

def round_half_up(n):
    return int(Decimal(str(n)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))

def run_imbens(obsX, obsS2, obsS1, obsS3, obsY, obsA,
               expX, expS2, expS1, expS3, expY, expA):
    # hyper
    lambda_ = 0.05


    h1 = outcome_ridge_bridge(obsY[obsA == 1], obsS2[obsA == 1], obsS1[obsA == 1], obsX[obsA == 1], obsS3[obsA == 1],
                              lambda_)
    exph1 = np.apply_along_axis(h1, 1, np.hstack((expS3[expA == 1], expS2[expA == 1], expX[expA == 1])))
    h0 = outcome_ridge_bridge(obsY[obsA == 0], obsS2[obsA == 0], obsS1[obsA == 0], obsX[obsA == 0], obsS3[obsA == 0],
                              lambda_)
    exph0 = np.apply_along_axis(h0, 1, np.hstack((expS3[expA == 0], expS2[expA == 0], expX[expA == 0])))
    ate_or = np.mean(exph1) - np.mean(exph0)

    q1 = selection_ridge_bridge(obsS2[obsA == 1], obsS1[obsA == 1], obsX[obsA == 1], obsS3[obsA == 1],
                                expS2[expA == 1], expS1[expA == 1], expX[expA == 1], expS3[expA == 1], lambda_)
    obsq1 = np.apply_along_axis(q1, 1, np.hstack((obsS2[obsA == 1], obsS1[obsA == 1], obsX[obsA == 1])))
    q0 = selection_ridge_bridge(obsS2[obsA == 0], obsS1[obsA == 0], obsX[obsA == 0], obsS3[obsA == 0],
                                expS2[expA == 0], expS1[expA == 0], expX[expA == 0], expS3[expA == 0], lambda_)
    obsq0 = np.apply_along_axis(q0, 1, np.hstack((obsS2[obsA == 0], obsS1[obsA == 0], obsX[obsA == 0])))

    obsh1 = np.apply_along_axis(h1, 1, np.hstack((obsS3[obsA == 1], obsS2[obsA == 1], obsX[obsA == 1])))
    obsh0 = np.apply_along_axis(h0, 1, np.hstack((obsS3[obsA == 0], obsS2[obsA == 0], obsX[obsA == 0])))
    ate_dr = ate_or + np.mean(obsq1 * (obsY[obsA == 1] - obsh1)) - np.mean(obsq0 * (obsY[obsA == 0] - obsh0))

    sigma = np.mean((exph1 - ate_dr) ** 2) / np.sum(expA) + np.mean(
        obsq1 ** 2 * (obsY[obsA == 1] - obsh1) ** 2) / np.sum(obsA)
    sd = np.sqrt(sigma)

    # np.savetxt(f"tmp/result_ridge_{idx}.csv", [ate_dr, ate_or, sd], delimiter=",")

    # ate_or
    return ate_dr, ate_or

def test_vary(TIME=9, time=6, sizeE=2000, sizeO=4000, regressionType='linear'):
    from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds, dgp_ts_equ_conf_design_linear

    ate_imbens_or, ate_imbens_dr = [],[]
    ite_exp, ate_exp = [],[]
    ite_our, ate_our = [],[]
    # defaut 50
    for seed in range(50):
        X, T, S, Y, G, \
            XO, TO, SO, YO, iteO, \
            XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(sizeE, sizeO, seed=seed, time=time,TIME=TIME)

        X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
            np.concatenate((YO, YE), 0), \
            np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

        # print('ground truth ATE: ' + str(np.mean(iteO)))

        S_ = S
        TIME = TIME

        # group data for imbens method
        set_all_seeds(seed)
        mid_t = round_half_up(time/2)-1
        obsS2, expS2 = SO[:, mid_t][:,None], SE[:, mid_t][:,None]
        # 选择临近
        obsS1, expS1 = SO[:, mid_t-1][:,None], SE[:, mid_t-1][:,None]
        obsS3, expS3 = SO[:, mid_t+1][:,None], SE[:, mid_t+1][:,None]
        # 选择最后一个
        obsS1, expS1 = SO[:,0][:,None], SE[:,0][:,None]
        obsS3, expS3 = SO[:, -1][:,None], SE[:, -1][:,None]
        # 选择所有
        # obsS1, expS1 = SO[:,:mid_t], SE[:,:mid_t]
        # obsS3, expS3 = SO[:, mid_t+1:mid_t*2+1], SE[:, mid_t+1:mid_t*2+1]

        # print(SO.shape, obsS1.shape, obsS2.shape, obsS3.shape)

        tao_est_dr, tao_est_or = run_imbens(obsX=XO, obsS2=obsS2, obsS1=obsS1, obsS3=obsS3, obsY=YO, obsA=TO,
                             expX=XE, expS2=expS2, expS1=expS1, expS3=expS3, expY=YE, expA=TE)

        # print(tao_est_dr.shape)
        # print(iteO.shape)
        # print(np.mean(tao_est_dr), np.mean(iteO))
        ate_imbens_dr.append(np.abs((np.mean(tao_est_dr) - np.mean(iteO))))
        ate_imbens_or.append(np.abs((np.mean(tao_est_or) - np.mean(iteO))))

        # run imbens



    # print('t_learner using exp data: ', np.mean(ite_exp), np.mean(ate_exp), np.std(ite_exp), np.std(ate_exp))
    # print('ite is equi con 31: ', np.mean(ite_our), np.mean(ate_our),  np.std(ite_our), np.std(ate_our))
    print('imbens_dr: ', np.mean(ate_imbens_dr), np.std(ate_imbens_dr))
    print('imbens_or: ', np.mean(ate_imbens_or), np.std(ate_imbens_or))




import warnings, torch
warnings.simplefilter('ignore')
torch.set_default_dtype(torch.double)
torch.manual_seed(123)
np.random.seed(123)


# varying mu result, correspond to Figure 4 a/b
# fix time=6, TIME= 7 8 9 10 11 12
for T in [7,8,9,10,11,12]:
    print('t=6, T='+str(T))
    test_vary(time=6, TIME=T, sizeE=2000, sizeO=4000, regressionType='linear')


# varying mu result, correspond to Figure 4 c/d
# fix TIME= 9 , time=45678
for t in [4,5,6,7,8]:
    print('t='+str(t)+', T=9')
    test_vary(time=t, TIME=9, sizeE=2000, sizeO=4000, regressionType='linear')
