
from TSS_HLCE.ite_ts_equi_con_2stage import run_tss

if __name__ == '__main__':
    # data:

    t_learn_obs_ite, t_learn_obs_ate, t_learn_obs_ite_std, t_learn_obs_ate_std, \
        t_learn_exp_ite, t_learn_exp_ate, t_learn_obs_exp_std, t_learn_obs_exp_std, \
        tss_ite, tss_ate, tss_ite_std, tss_ate_std \
        = run_tss(XO, TO, SO, YO, XE, TE, SE, YE,
            TIME, iteO)

