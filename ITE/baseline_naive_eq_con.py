import numpy as np
from sklearn.model_selection import train_test_split
from MRITE.mr_ite import nuisance_Slearner, nuisance_Tlearner
from utils.dgp import set_all_seeds
import torch
import argparse

import copy

from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from utils.basic_model import MLP
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.dgp import set_all_seeds

class NAIVE_EQ_CON_MLP(nn.Module):
    '''
    construct MLP model for naive estimator \tau(x) = mu_yo(1) - mu_yo(0) + mu_se(1) - mu_se(0) + mu_so(0) - mu_so(1)
            - p(any | G=E,X)
    base-   - p(G|X)
            - p(any | G=E,X)
    '''

    def __init__(self, x_dim, args):
        torch.set_default_tensor_type(torch.DoubleTensor)
        super().__init__()
        if args.dropout != 0:
            self.dropout = nn.Dropout(args.dropout)
        self.BCE = nn.BCELoss(reduction='none')
        self.MSE = nn.MSELoss(reduction='none')
        self.sigmoid = nn.functional.sigmoid
        set_all_seeds(args.seed)
        self.base = MLP(input_dim=x_dim, output_dim=args.hidden_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                        dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        # heads for G=E
        self.base_E = MLP(input_dim=args.hidden_dim, output_dim=args.hidden_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                          dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        # self.pT_XE = nn.Linear(in_features=args.hidden_dim, out_features=1, bias=True)
        self.muSE1 = MLP(input_dim=args.hidden_dim, output_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                         dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        self.muSE0 = MLP(input_dim=args.hidden_dim, output_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                         dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        # heads for G=O
        self.base_O = MLP(input_dim=args.hidden_dim, output_dim=args.hidden_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                          dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        # self.pT_XO = nn.Linear(in_features=args.hidden_dim, out_features=1, bias=True)
        self.muSO1 = MLP(input_dim=args.hidden_dim, output_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                         dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        self.muSO0 = MLP(input_dim=args.hidden_dim, output_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                         dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        self.muYO1 = MLP(input_dim=args.hidden_dim, output_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                         dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        self.muYO0 = MLP(input_dim=args.hidden_dim, output_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                         dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)

        # second-step MLP E[\hat Y_MR | X=x]
        set_all_seeds(args.seed)
        self.ite_estimator = MLP(input_dim=x_dim, output_dim=1, hidden_dim=args.hidden_dim_2, n_layers=args.n_layers_2,
                                 dropout=args.dropout_2, activation=args.activation, slope=args.slope, device=args.device)
        self.early_stop = args.early_stop
        self.early_stop_iter = args.early_stop_iter
        self.normalization = args.normalization
        self.normalization_x = args.normalization_x
        self.device = args.device
        self.args = args
        self.to(args.device)
        set_all_seeds(args.seed)


    def forward(self, X):
        X_base = self.base(X)

        # G=E group
        hidden_E = self.base_E(X_base)
        muSE1 = self.muSE1(hidden_E)
        muSE0 = self.muSE0(hidden_E)

        # G=O group
        hidden_O = self.base_O(X_base)
        muSO1 = self.muSO1(hidden_O)
        muSO0 = self.muSO0(hidden_O)
        muYO1 = self.muYO1(hidden_O)
        muYO0 = self.muYO0(hidden_O)

        return  muSE1, muSE0, \
           muSO1, muSO0, muYO1, muYO0

    def fit(self, XO, TO, SO, YO, XE, TE, SE,
            XO_val=None, TO_val=None, SO_val=None, YO_val=None, XE_val=None, TE_val=None, SE_val=None,
            only_first=False):
        XO, TO, SO, YO = torch.tensor(XO).to(self.device), torch.tensor(TO).to(self.device), \
            torch.tensor(SO).to(self.device), torch.tensor(YO).to(self.device),
        XE, TE, SE = torch.tensor(XE).to(self.device), torch.tensor(TE).to(self.device), \
            torch.tensor(SE).to(self.device)
        # val
        if XO_val is not None:
            XO_val, TO_val, SO_val, YO_val = torch.tensor(XO_val).to(self.device), torch.tensor(TO_val).to(self.device), \
                torch.tensor(SO_val).to(self.device), torch.tensor(YO_val).to(self.device),
            XE_val, TE_val, SE_val = torch.tensor(XE_val).to(self.device), torch.tensor(TE_val).to(self.device), \
                torch.tensor(SE_val).to(self.device)

        if self.normalization_x:
            x_both = torch.cat((XO,XE), dim=0)
            self.x_mean, self.x_std = torch.mean(x_both, dim=0), torch.std(x_both, dim=0)
            XO = (XO - self.x_mean) / self.x_std
            XE = (XE - self.x_mean) / self.x_std
            if XO_val is not None:
                XO_val, XE_val = (XO_val - self.x_mean) / self.x_std, (XE_val - self.x_mean) / self.x_std

        self.first_step_fit(XO, TO, SO, YO, XE, TE, SE,
                            XO_val, TO_val, SO_val, YO_val, XE_val, TE_val, SE_val)
        return self

    def predict(self, X):
        self.eval()
        X = torch.tensor(X).to(self.device)
        if self.normalization_x:
            X = (X - self.x_mean) / self.x_std

        muSE1, muSE0, \
            muSO1, muSO0, muYO1, muYO0 = self.forward(X)
        if self.normalization:
            muYO1, muYO0 = muYO1*self.y_std+self.y_mean, muYO0*self.y_std+self.y_mean
            muSO1, muSO0 = muSO1*self.so_std+self.so_mean, muSO0*self.so_std+self.so_mean
            muSE1, muSE0 = muSE1*self.se_std+self.se_mean, muSE0*self.se_std+self.se_mean
        ite_est = muYO1 - muYO0 + muSE1 - muSE0 + muSO0 - muSO1
        ite_est = ite_est.cpu().detach().numpy()
        return ite_est

    def first_step_fit(self, XO, TO, SO, YO, XE, TE, SE,
                       XO_val=None, TO_val=None, SO_val=None, YO_val=None, XE_val=None, TE_val=None, SE_val=None):

        if self.normalization:
            self.y_mean, self.y_std = torch.mean(YO), torch.std(YO)
            self.so_mean, self.so_std = torch.mean(SO), torch.std(SO)
            self.se_mean, self.se_std = torch.mean(SE), torch.std(SE)
            YO = (YO-self.y_mean) / self.y_std
            SO = (SO-self.so_mean) / self.so_std
            SE = (SE-self.se_mean) / self.se_std
            if XO_val is not None:
                YO_val = (YO_val - self.y_mean) / self.y_std
                SO_val = (SO_val - self.so_mean) / self.so_std
                SE_val = (SE_val - self.se_mean) / self.se_std

        X, T, S, Y, G = self.input_preparation(XO, TO, SO, YO, XE, TE, SE)
        if XO_val is not None:
            X_val, T_val, S_val, Y_val, G_val = \
                self.input_preparation(XO_val, TO_val, SO_val, YO_val, XE_val, TE_val, SE_val)

        self.ratio_e, self.ratio_o = torch.sum(G)/G.shape[0], 1- torch.sum(G)/G.shape[0]
        # print('ratio= '+ str(self.ratio_e))

        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        dataset = TensorDataset(X, T, S, Y, G)
        train_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        losses_train, losses_valid = [], []
        early_stop_flag = 0
        for epoch in range(self.args.epochs):
            self.train()
            for x, t, s, y, g in train_loader:
                optimizer.zero_grad()
                loss = self.first_step_loss(x, t, s, y, g)
                loss.backward()
                optimizer.step()
                losses_train.append(loss.item())
            # validation
            if self.early_stop:
                self.eval()
                loss_val = self.first_step_loss(X_val, T_val, S_val, Y_val, G_val)
                losses_valid.append(loss_val.item())
                if epoch>self.args.base_epochs and losses_valid[-1] > losses_valid[-2]:
                    self.best_frist_loss_val = loss_val.item() #loss_val.item().cpu().detach().numpy()
                    early_stop_flag +=1
                    if early_stop_flag >= self.early_stop_iter:
                        print('early_stop of first training in epoch:' + str(epoch))
                        self.load_state_dict(best_model)
                        return self
                else:
                    early_stop_flag = 0
                    best_model = copy.deepcopy(self.state_dict())

        return self

    def first_step_loss(self, x, t, s, y, g):
        muSE1, muSE0, \
            muSO1, muSO0, muYO1, muYO0 = self.forward(x)
        # loss func
        if self.args.reweight:
            weE, weO = 1 / self.ratio_e, 1 / self.ratio_o
        else:
            weE, weO = 1., 1.
        assert (muSE1.shape == s.shape) & (muSO1.shape == s.shape) & (muYO1.shape == y.shape)
        loss_muSE = weE * self.MSE(input=muSE1, target=s) * t * g + \
                    weE * self.MSE(input=muSE0, target=s) * (1 - t) * g
        loss_muSO = weO * self.MSE(input=muSO1, target=s) * t * (1 - g) + \
                    weO * self.MSE(input=muSO0, target=s) * (1 - t) * (1 - g)
        loss_muYO = weO * self.MSE(input=muYO1, target=y) * t * (1 - g) + \
                    weO * self.MSE(input=muYO0, target=y) * (1 - t) * (1 - g)
        loss = torch.mean( loss_muSE + loss_muSO + loss_muYO)
        return loss


    def input_preparation(self, XO, TO, SO, YO, XE, TE, SE):
        YE_fill, GE, GO = torch.tensor(torch.zeros_like(TE)).to(self.device), \
            torch.ones_like(TE).to(self.device), torch.zeros_like(TO).to(self.device)
        X, T, S, Y, G = torch.cat((XO, XE), dim=0), torch.cat((TO, TE), dim=0)[:, None]*1., \
            torch.cat((SO, SE), dim=0)[:, None]*1., torch.cat((YO, YE_fill), dim=0)[:, None]*1., \
            torch.cat((GO, GE), dim=0)[:, None]*1.
        return X, T, S, Y, G



def naive_LTHE(X, S, Y, T, G, learner='T', regressionType='best'):
    '''
    :param X:
    :param S:
    :param Y:
    :param T:
    :param G:
    :param predict_X:
    :param learner:
    :param regressionType:
    :return:  function of predictor
    '''
    if learner == 'T':
        muSE1, muSO1, muYO1, muSE0, muSO0, muYO0, piE, piO, piG = nuisance_Tlearner(X, S, Y, T, G, regressionType=regressionType)
    else:
        muSE, muSO, muYO, piE, piO, piG = nuisance_Slearner(X, S, Y, T, G, regressionType=regressionType)
        muSE1, muSO1, muYO1, muSE0, muSO0, muYO0 =  muSE, muSO, muYO,  muSE, muSO, muYO

    def predict(input):
        muYO_1X = muYO1.predict(input)
        muYO_0X = muYO0.predict(input)
        muSE_1X = muSE1.predict(input)
        muSE_0X = muSE0.predict(input)
        muSO_1X = muSO1.predict(input)
        muSO_0X = muSO0.predict(input)
        return muYO_1X - muYO_0X + muSE_1X - muSE_0X + muSO_0X - muSO_1X

    return predict



def IHDP(seed=123):
    from utils.IHDP_datasets import LongTermIHDP
    import warnings
    warnings.simplefilter('ignore')

    ite, ate = [], []
    lti = LongTermIHDP()
    for i, (train, valid, test) in enumerate(lti.get_train_valid_test()):
        set_all_seeds(seed)

        g_train, t_train, s_train, y_train, _, _, y1_train, y0_train, x_train, _ = train
        g_valid, t_valid, s_valid, y_valid, _, _, y1_valid, y0_valid, x_valid, _ = valid
        t_test, s_test, y_test, _, _, y1_test, y0_test, x_test, _ = test
        # ground true
        iteO_test = y1_test - y0_test

        ite_naive_predictor = naive_LTHE(x_train, np.squeeze(s_train), np.squeeze(y_train),
                         np.squeeze(t_train), np.squeeze(g_train), regressionType='kernelRidge')
        ite_est = ite_naive_predictor(x_test)[:,None]

        assert ite_est.shape == iteO_test.shape
        print('Naive ' + str(i) + ': ' + str(PEHE_ITE(ite_est, iteO_test)) + ' ' + str(RMSE_ATE(ite_est, iteO_test)))

        ite.append(PEHE_ITE(ite_est, iteO_test))
        ate.append(RMSE_ATE(ite_est, iteO_test))
    return ite, ate



def NEWS(seed=123):
    from utils.NEWS_dataset import LongTermNEWS
    import warnings
    warnings.simplefilter('ignore')

    ite, ate = [], []
    lti = LongTermNEWS()
    for i, (train, valid, test) in enumerate(lti.get_train_valid_test()):
        set_all_seeds(seed)
        g_train, t_train, s_train, y_train, y1_train, y0_train, x_train = train
        g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid = valid
        t_test, s_test, y_test, y1_test, y0_test, x_test = test
        # ground true
        iteO_test = y1_test - y0_test

        ite_naive_predictor = naive_LTHE(x_train, np.squeeze(s_train), np.squeeze(y_train),
                         np.squeeze(t_train), np.squeeze(g_train), regressionType='kernelRidge')
        ite_est = ite_naive_predictor(x_test)[:,None]

        print('shape: ', ite_est.shape, iteO_test.shape)
        assert ite_est.shape == iteO_test.shape
        print('Naive ' + str(i) + ': ' + str(PEHE_ITE(ite_est, iteO_test)) + ' ' + str(RMSE_ATE(ite_est, iteO_test)))

        ite.append(PEHE_ITE(ite_est, iteO_test))
        ate.append(RMSE_ATE(ite_est, iteO_test))
    return ite, ate


if __name__ == '__main__':
    ite, ate = IHDP(123)
    # ite, ate = NEWS(123)
    print(len(ite))
    print(np.mean(ite), np.std(ite))
    print(np.mean(ate), np.std(ate))

# IHDP
# 2.799472930194016 0.6484757067492971
 # 0.622968495596119 0.46795396196716854

# NEWS
# 3.501066481697836 0.23260504797943837
# 0.7214068953668424 0.14288778323377394