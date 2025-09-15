import sys

import numpy

sys.path.append('..')

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import copy
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.tree import DecisionTreeRegressor as reg_tree
from sklearn.ensemble import AdaBoostRegressor as ada_reg
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from utils.basic_model import TrainedMLP, TrainedLogisticMLP, TrainedLogistic_G, TrainedMLPLinear
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def get_best_for_data(X, Y):
    regs = [rfr(n_estimators=i) for i in [10, 20, 40, 60, 100, 150, 200]]
    regs += [reg_tree(max_depth=i) for i in [5, 10, 20, 30, 40, 50]]
    regs += [ada_reg(n_estimators=i) for i in [10, 20, 50, 70, 100, 150, 200]]
    regs += [gbr(n_estimators=i) for i in [50, 70, 100, 150, 200]]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    val_errs = []
    models = []
    for reg in regs:
        model = copy.deepcopy(reg)
        model.fit(x_train, y_train)
        val_errs.append(mse(y_test, model.predict(x_test)))
        models.append(copy.deepcopy(model))
    min_ind = val_errs.index(min(val_errs))
    # print(str(model)[:40], val_errs[min_ind])
    # print(copy.deepcopy(models[min_ind]))
    return copy.deepcopy(models[min_ind])


def e_x_estimator(x, w, bias=True):
    """estimate P(W_i=1|X_i=x)"""
    log_reg = LogisticRegression(fit_intercept=bias).fit(x, w)
    return log_reg


class e_x_only_quadratic:

    def __init__(self, fit_intercept=True):
        self.log_reg = LogisticRegression(fit_intercept=fit_intercept)

    def fit(self, x, w):
        self.log_reg.fit(np.power(x, 2), w)
        return self

    def predict(self, x):
        return self.log_reg.predict(np.power(x, 2))

    def predict_proba(self, x):
        return self.log_reg.predict_proba(np.power(x, 2))


class e_x_quadratic:

    def __init__(self):
        self.log_reg = LogisticRegression()

    def fit(self, x, w):
        self.log_reg.fit(np.concatenate((np.power(x, 2),x), axis=1), w)
        return self

    def predict(self, x):
        return self.log_reg.predict(np.concatenate((np.power(x, 2),x), axis=1))

    def predict_proba(self, x):
        return self.log_reg.predict_proba(np.concatenate((np.power(x, 2),x), axis=1))

def e_x_estimator_mlp(x, w, epochs=100):
    log_reg = TrainedLogisticMLP(input_dim=x.shape[1], output_dim=1, hidden_dim=100, n_layers=3,
                                activation='lrelu', slope=.1, device='cuda').fit(X=x, Y=w, epochs=epochs)
    return log_reg

def e_x_estimator_G(x, g, epochs=100):
    """estimate P(G=1|X_i=x)"""
    log_reg = TrainedLogistic_G().fit(x, g, epochs=epochs)
    return log_reg


def e_x_estimator_quadratic(x, w):
    """estimate P(W_i=1|X_i=x)"""
    log_reg = e_x_quadratic().fit(x, w)
    return log_reg

def e_x_estimator_only_quadratic(x, w, bias=True):
    """estimate P(W_i=1|X_i=x)"""
    log_reg = e_x_only_quadratic(fit_intercept=bias).fit(x, w)
    return log_reg


def regression1D(x, y, type='linear', power=4, bias=True):
    # ensure y shape is (sample size, ) instead of (sample size, 1)
    # fit regression of E[Y|X]
    if type == 'linear':
        regression = LinearRegression(fit_intercept=bias).fit(X=x, y=y)
    elif type == 'Power':
        def power_func(X):
            return X ** power
        pipeline = Pipeline([
            ('fourth_power', FunctionTransformer(power_func)),  # Extract only x^4
            ('regressor', LinearRegression())  # Fit linear regression
        ])
        pipeline.fit(x, y)
        return pipeline

    elif type == 'Poly':
        polyreg=Pipeline([("poly", PolynomialFeatures(degree=4)),
                  ("lin_reg", LinearRegression())])
        polyreg.fit(x, y)
        return polyreg
    elif type == 'kernelRidge':
        # print(x.shape)
        # print(y.shape)
        regression = KernelRidge().fit(X=x, y=y)
        # print(regression)
    elif type == 'randomForestRegressor':
        regression = RandomForestRegressor().fit(X=x, y=y)
    elif type == 'best':
        regression = get_best_for_data(x, y)
    elif type == 'mlp':
        regression = TrainedMLP(input_dim=x.shape[1], output_dim=1, hidden_dim=256, n_layers=2,
                                activation='lrelu', slope=.1, device='cuda').fit(X=x, Y=y, epochs=200)
    elif type == 'mlp_linear':
        regression = TrainedMLPLinear(input_dim=x.shape[1], output_dim=1, device='cuda').fit(X=x, Y=y, epochs=200)
    else:
        raise Exception('undefined regression type')

    return regression



def regressionPoly(x, y, degree=2):
    polyreg = Pipeline([("poly", PolynomialFeatures(degree=degree)),
                        ("lin_reg", LinearRegression())])
    polyreg.fit(x, y)

    # polyreg.coef_[2]  coef of x^2
    # polyreg.coef_[1]  coef of x
    # polyreg.intercept_  intercept
    return polyreg


def naive_estimator(t, y):
    """estimate E[Y|T=1] - E[Y|T=0]"""
    index_t1 = np.squeeze(t == 1)
    index_t0 = np.squeeze(t == 0)
    y1 = y[index_t1,]
    y0 = y[index_t0,]

    tau = np.mean(y1) - np.mean(y0)
    return tau


def ipw_estimator(x, t, y):
    """estimate ATE using ipw method"""
    propensity_socre_reg = e_x_estimator(x, t)
    propensity_socre = propensity_socre_reg.predict_proba(x)
    propensity_socre = propensity_socre[:, 1][:, None]  # prob of treatment=1

    ps1 = 1. / np.sum(t / propensity_socre)
    y1 = ps1 * np.sum(y * t / propensity_socre)
    ps0 = 1. / np.sum((1. - t) / (1. - propensity_socre))
    y0 = ps0 * np.sum(y * ((1. - t) / (1 - propensity_socre)))
    # print((1. - t).sum())
    # print(t.sum())

    tau = y1 - y0
    return tau


def s_learner_estimator(x, t, y, type='linear'):
    """ estimate E(Y|X,T=1)-E(Y|X,T=0)
        s_learner: naive estimator using same regression function
    """
    # fit regression of E[Y|X]
    x_t = np.concatenate((x, t[:, None]), axis=1)
    if type == 'linear':
        regression = LinearRegression().fit(X=x_t, y=y)
    elif type == 'kernelRidge':
        regression = KernelRidge().fit(X=x_t, y=y)
    elif type == 'randomForestRegressor':
        regression = RandomForestRegressor().fit(X=x_t, y=y)
    elif type == 'best':
        regression = get_best_for_data(x_t, y)
    elif type == 'mlp':
        regression = TrainedMLP(input_dim=x.shape[1], output_dim=1, hidden_dim=200, n_layers=3,
                                activation='lrelu', slope=.1, device='cuda').fit(X=x, Y=y, epochs=100)
    else:
        raise Exception('undefined regression type')
    x_t1 = np.concatenate((x, numpy.ones_like(t)[:, None]), axis=1)
    x_t0 = np.concatenate((x, numpy.zeros_like(t)[:, None]), axis=1)
    y1 = regression.predict(X=x_t1)
    y0 = regression.predict(X=x_t0)

    tau = y1 - y0
    return tau, regression


def t_learner_estimator(x, t, y, power=2, type='linear'):
    """ estimate E(Y|X,T=1)-E(Y|X,T=0)
        t_learner: naive estimator using different regression function
    """
    index_t1 = np.squeeze(t == 1)
    index_t0 = np.squeeze(t == 0)
    x_t1 = x[index_t1]
    x_t0 = x[index_t0]
    # print(x_t1.shape)
    # print(y[index_t1,].shape)

    if type == 'linear':
        regression_1 = LinearRegression().fit(X=x_t1, y=y[index_t1,])
        regression_0 = LinearRegression().fit(X=x_t0, y=y[index_t0,])
    elif type == 'Power':
        def power_func(X):
            return X ** power
        regression_1 = Pipeline([
            ('fourth_power', FunctionTransformer(power_func)),  # Extract only x^4
            ('regressor', LinearRegression())  # Fit linear regression
        ])
        regression_1.fit(X=x_t1, y=y[index_t1,])

        regression_0 = Pipeline([
            ('fourth_power', FunctionTransformer(power_func)),  # Extract only x^4
            ('regressor', LinearRegression())  # Fit linear regression
        ])
        regression_0.fit(X=x_t0, y=y[index_t0,])

    elif type == 'Poly':
        regression_1=Pipeline([("poly", PolynomialFeatures(degree=4)),
                  ("lin_reg", LinearRegression())])

        regression_1.fit(x_t1, y[index_t1,])
        regression_0=Pipeline([("poly", PolynomialFeatures(degree=4)),
                  ("lin_reg", LinearRegression())])
        regression_0.fit(x_t0, y[index_t0,])

    elif type == 'kernelRidge':
        regression_1 = KernelRidge().fit(X=x_t1, y=y[index_t1,])
        regression_0 = KernelRidge().fit(X=x_t0, y=y[index_t0,])
    elif type == 'randomForestRegressor':
        regression_1 = RandomForestRegressor().fit(X=x_t1, y=y[index_t1,])
        regression_0 = RandomForestRegressor().fit(X=x_t0, y=y[index_t0,])
    elif type == 'best':
        regression_1 = get_best_for_data(x_t1, y[index_t1,])
        regression_0 = get_best_for_data(x_t0, y[index_t0,])
    elif type == 'mlp':
        regression_1 = TrainedMLP(input_dim=x.shape[1], output_dim=1, hidden_dim=200, n_layers=3,
                                activation='lrelu', slope=.1, device='cuda').fit(X=x_t1, Y=y[index_t1,], epochs=100)

        regression_0 = TrainedMLP(input_dim=x.shape[1], output_dim=1, hidden_dim=200, n_layers=3,
                                activation='lrelu', slope=.1, device='cuda').fit(X=x_t0, Y=y[index_t0,], epochs=100)
    else:
        raise Exception('undefined regression type')

    y1 = regression_1.predict(X=x)
    y0 = regression_0.predict(X=x)

    tau = y1 - y0
    return tau, regression_1, regression_0


def x_learner_estimator():
    pass


def double_robust_estimator(x, t, y):
    pass


def tmle_estimator(x, t, y):
    pass
