from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split
from ATE.latent_uncon import imputationApproach
from utils.dgp import set_all_seeds

def test_dgp(seed=0):

    from utils.dgp import dgp_equ_conf, dgp_equ_conf_design
    X, T, S, Y, G, \
        XO, TO, SO, YO, iteO, \
        XE, TE, SE, YE, iteE = \
    dgp_equ_conf_design(size_O=5000, size_E=2000, seed=seed)

    # dgp_equ_conf(dim_x=1, dim_u=1, size_O=2000, size_E=500, seed=seed)

    print('ground truth ATE: ' + str(np.mean(iteO)))

    tau = imputationApproach(X, S, Y, T, G, regressionType='linear')

    # ite_mr = MR_LTHE(X, S, Y, T, G, predict_X=XO, cross_fitting=True, regressionType='linear')
    # print(ite_mr.shape, iteO.shape)
    # print('MR_LTCE: ' + str(PEHE_ITE(ite_mr, iteO)) + ' ' + str(RMSE_ATE(ite_mr, iteO)))

    return  np.abs(np.mean(tau) - np.mean(iteO))

def run_dgp():
    iteT, ateT, iteN, ateN = [], [], [], []
    for i in range(10):
        print('seed' + str(i))
        ate_errorT = test_dgp(i)
        ateT.append(ate_errorT)

    print(np.mean(ateT), np.std(ateT))
    # test2()

def test_IHDP(seed=123):
    from utils.IHDP_datasets import LongTermIHDP
    import warnings
    warnings.simplefilter('ignore')

    ate_Error = []
    lti = LongTermIHDP()
    for i, (train, valid, test) in enumerate(lti.get_train_valid_test()):
        set_all_seeds(seed)

        g_train, t_train, s_train, y_train, _, _, y1_train, y0_train, x_train, _ = train
        g_valid, t_valid, s_valid, y_valid, _, _, y1_valid, y0_valid, x_valid, _ = valid
        t_test, s_test, y_test, _, _, y1_test, y0_test, x_test, _ = test
        # ground true
        iteO_test = y1_test - y0_test

        tau = imputationApproach(x_train, np.squeeze(s_train), np.squeeze(y_train),
                                 np.squeeze(t_train), np.squeeze(g_train), regressionType='linear')
        print(np.abs(np.mean(iteO_test) - np.mean(tau)))
        ate_Error.append(np.abs(np.mean(iteO_test) - np.mean(tau)))
        # 0.7516767959554502 0.725913588825251
    return ate_Error

if __name__ == '__main__':

    import warnings
    warnings.simplefilter('ignore')
    # ate_Error = test_IHDP()
    ate_Error = test_IHDP()
    print('results')
    print(ate_Error)
    print(np.mean(ate_Error), np.std(ate_Error))
