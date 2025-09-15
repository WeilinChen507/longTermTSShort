
from TSS_HLCE.ite_ts_equi_con_2stage import run_tss
from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.ts_eq_dgp import dgp_ts_equ_conf_design, set_all_seeds, dgp_ts_equ_conf_design_x, dgp_ts_equ_conf_design_ex
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == '__main__':
    # data: 假装这是你都到的数据
    X, T, S, Y, G, \
        XO, TO, SO, YO, iteO, \
        XE, TE, SE, YE, iteE = dgp_ts_equ_conf_design(size_E=1000, size_O=2000, seed=123, time=3, TIME=7)

    Y,YO,YE = Y[:,None],YO[:,None],YE[:,None]
    # 拼接
    # X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
    #     np.concatenate((YO, YE), 0), \
    #     np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    # 随机把数据分为两半，
    #   一半用于作模拟器，这里短期S跟长期Y都要拟合，做完模拟器后这些样本就丢掉  index为first的fir
    #   另一半，只需要用到他的X跟T，其S跟Y都用上述模拟器替换掉               index为second的sec
    #   ITE也是用模拟器生成
    XO_fir, XO_sec, TO_fir, TO_sec, SO_fir, SO_sec, YO_fir, YO_sec = train_test_split(XO, TO, SO, YO, test_size=0.5)
    XE_fir, XE_sec, TE_fir, TE_sec, SE_fir, SE_sec, YE_fir, YE_sec = train_test_split(XE, TE, SE, YE, test_size=0.5)

    # 模拟器用第一份数据fir的样本拟合
    # exp 跟 obs 需要分开拟合
    regressionType = 'mlp'       # 可以替换 linear mlp best kernelRidge ..
    _, rE1, rE0 = t_learner_estimator(XE_fir, TE_fir, YE_fir, type=regressionType)
    _, rO1, rO0 = t_learner_estimator(XO_fir, TO_fir, YO_fir, type=regressionType)

    # 生成第二份数据sec的potential outcome
    PO_obs_1, PO_obs_0 = rO1.predict(X=XO_sec), rO0.predict(X=XO_sec)
    PO_exp_1, PO_exp_0 = rE1.predict(X=XE_sec), rE0.predict(X=XE_sec)

    # 生成index为2的ite
    iteO, iteE = PO_obs_1 - PO_obs_0, PO_exp_1 - PO_exp_0

    # S的部分也需要上述操作
    print('一共有 '+str(S.shape[1]) + ' 步短期结果')
    PO_short_exp_1, PO_short_exp_0, PO_short_obs_1, PO_short_obs_0 =\
        np.zeros_like(SE_sec), np.zeros_like(SE_sec),np.zeros_like(SO_sec),np.zeros_like(SO_sec)
    for i in range(S.shape[1]):
        regressionType = 'mlp'  # 可以替换 linear mlp best kernelRidge ..
        _, rE1, rE0 = t_learner_estimator(XE_fir, TE_fir, SE_fir[:,i][:,None], type=regressionType)
        _, rO1, rO0 = t_learner_estimator(XO_fir, TO_fir, SO_fir[:,i][:,None], type=regressionType)

        # 生成sec的短期 potential outcome
        PO_obs_1, PO_obs_0 = rO1.predict(X=XO_sec), rO0.predict(X=XO_sec)
        PO_exp_1, PO_exp_0 = rE1.predict(X=XE_sec), rE0.predict(X=XE_sec)

        # 放入集合存储
        PO_short_exp_1[:,i] = PO_exp_1
        PO_short_exp_0[:,i] = PO_exp_0
        PO_short_obs_1[:,i] = PO_obs_1
        PO_short_obs_0[:,i] = PO_obs_0


    # 最终用于训练的数据：
    XE, TE, SE1, SE0, YE1, YE0 = XE_sec, TE_sec, PO_short_exp_1, PO_short_exp_0, PO_exp_1, PO_exp_0
    XO, TO, SO1, SO0, YO1, YO0 = XO_sec, TO_sec, PO_short_obs_1, PO_short_obs_0, PO_obs_1, PO_obs_0
    iteO, iteE = iteO, iteE

    # 检查样本大小
    print(XE.shape, TE.shape, SE1.shape, SE0.shape, YE1.shape, YE0.shape)
    print(XO.shape, TO.shape, SO1.shape, SO0.shape, YO1.shape, YO0.shape)
    print(iteO.shape)
    print(iteE.shape)