import copy
import sys
sys.path.append('..')

import torch
import argparse

from utils.ite_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
from utils.metric import RMSE_ATE, PEHE_ITE
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from utils.basic_model import MLP
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.dgp import set_all_seeds

class PROPEN_MLP(nn.Module):
    '''
    construct MLP model for MR_ITE
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
        self.pG_X = MLP(input_dim=args.hidden_dim, output_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                        dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        # self.pG_X = nn.Linear(in_features=args.hidden_dim, out_features=1, bias=True)
        # heads for G=E
        self.base_E = MLP(input_dim=args.hidden_dim, output_dim=args.hidden_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                          dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        self.pT_XE = MLP(input_dim=args.hidden_dim, output_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                        dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        # self.pT_XE = nn.Linear(in_features=args.hidden_dim, out_features=1, bias=True)
        # heads for G=O
        self.base_O = MLP(input_dim=args.hidden_dim, output_dim=args.hidden_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                          dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        self.pT_XO = MLP(input_dim=args.hidden_dim, output_dim=1, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                        dropout=args.dropout, activation=args.activation, slope=args.slope, device=args.device)
        # self.pT_XO = nn.Linear(in_features=args.hidden_dim, out_features=1, bias=True)

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
        prob_G = self.sigmoid(self.pG_X(X_base))

        # G=E group
        hidden_E = self.base_E(X_base)
        prob_TE = self.sigmoid(self.pT_XE(hidden_E))
        # prob_TE = self.sigmoid(self.pT_XE(X_base))

        # G=O group
        hidden_O = self.base_O(X_base)
        prob_TO = self.sigmoid(self.pT_XO(hidden_O))
        # prob_TO = self.sigmoid(self.pT_XO(X_base))

        return prob_G,  prob_TE, prob_TO

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
        if only_first is False:
            self.second_step_fit(XO, TO, SO, YO, XE, TE, SE,
                             XO_val, TO_val, SO_val, YO_val, XE_val, TE_val, SE_val)
        return self.ite_estimator

    def predict(self, X):
        self.eval()
        X = torch.tensor(X).to(self.device)
        if self.normalization_x:
            X = (X - self.x_mean) / self.x_std
        ite_est = self.ite_estimator(X)
        if self.normalization:
            ite_est = ite_est * self.y_mr_std + self.y_mr_mean
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
                    self.best_frist_loss_val = loss_val.item()#.cpu().detach().numpy()
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
        prob_G, prob_TE, prob_TO = self.forward(x)
        # loss func
        if self.args.reweight:
            weE, weO = 1 / self.ratio_e, 1 / self.ratio_o
        else:
            weE, weO = 1., 1.
        assert (prob_G.shape == g.shape) & (prob_TE.shape == t.shape) & (prob_TO.shape == t.shape)

        loss_G = self.BCE(input=prob_G, target=g)
        loss_TE = weE * self.BCE(input=prob_TE, target=t) * g
        loss_TO = weO * self.BCE(input=prob_TO, target=t) * (1 - g)

        loss = torch.mean(loss_G + loss_TE + loss_TO)
        return loss

    def second_step_fit(self, XO, TO, SO, YO, XE, TE, SE,
                        XO_val=None, TO_val=None, SO_val=None, YO_val=None, XE_val=None, TE_val=None, SE_val=None):
        X, T, S, Y, G = self.input_preparation(XO, TO, SO, YO, XE, TE, SE)
        with torch.no_grad():
            Y_MR = self.construct_YMR(X, T, S, Y, G)
            if self.normalization:
                self.y_mr_mean, self.y_mr_std = torch.mean(Y_MR), torch.std(Y_MR)
                Y_MR = (Y_MR - self.y_mr_mean) / self.y_mr_std

        if XO_val is not None:
            X_val, T_val, S_val, Y_val, G_val = \
                self.input_preparation(XO_val, TO_val, SO_val, YO_val, XE_val, TE_val, SE_val)
            with torch.no_grad():
                Y_MR_val = self.construct_YMR(X_val, T_val, S_val, Y_val, G_val)
                if self.normalization:
                    Y_MR_val = (Y_MR_val - self.y_mr_mean) / self.y_mr_std

        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr_2, weight_decay=self.args.weight_decay_2)
        elif self.args.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.args.lr_2, weight_decay=self.args.weight_decay_2)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr_2, weight_decay=self.args.weight_decay_2)

        dataset = TensorDataset(X, Y_MR)
        train_loader = DataLoader(dataset, batch_size=self.args.batch_size_2, shuffle=True)

        losses_train, losses_valid = [], []
        early_stop_flag = 0
        for epoch in range(self.args.epochs):
            self.train()
            for x, y_mr in train_loader:
                optimizer.zero_grad()
                hat_ite = self.ite_estimator(x)
                loss = self.MSE(input=hat_ite, target=y_mr)  # loss func
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                # validation
            if self.early_stop:
                self.eval()
                hat_ite_val = self.ite_estimator(X_val)
                loss_val = self.MSE(input=hat_ite_val, target=Y_MR_val)  # loss func
                loss_val = torch.mean(loss_val)
                losses_valid.append(loss_val.item())
                if epoch>self.args.base_epochs and losses_valid[-1] > losses_valid[-2]:
                    self.best_sec_loss_val = loss_val.item() #.cpu().detach().numpy()
                    early_stop_flag += 1
                    if early_stop_flag >= self.early_stop_iter:
                        print('early_stop of second training in epoch:' + str(epoch))
                        self.load_state_dict(best_model)
                        return self
                else:
                    early_stop_flag = 0
                    best_model = copy.deepcopy(self.state_dict())

        return self


    def construct_YMR(self, X, T, S, Y, G, epsilon=1e-6):
        prob_G, prob_TE, prob_TO = self.forward(X)

        pG = torch.sum(G == 0) / G.shape[0]  # p(G=O)

        Y_MR = G / pG * (2 * T - 1) / (1 - T + (2 * T - 1) * prob_TE) * S * (
                    1 / prob_G - 1) + \
               (1 - G) / pG * ((2 * T - 1) / (1 - T + (2 * T - 1) * prob_TO) * (Y - S))
        if torch.isnan(torch.mean(Y_MR)) or torch.isinf(torch.mean(Y_MR)):
            raise Exception("Y_MR is nan or inf")
        return Y_MR

    def input_preparation(self, XO, TO, SO, YO, XE, TE, SE):
        YE_fill, GE, GO = torch.tensor(torch.zeros_like(TE)).to(self.device), \
            torch.ones_like(TE).to(self.device), torch.zeros_like(TO).to(self.device)
        X, T, S, Y, G = torch.cat((XO, XE), dim=0), torch.cat((TO, TE), dim=0)[:, None]*1., \
            torch.cat((SO, SE), dim=0)[:, None]*1., torch.cat((YO, YE_fill), dim=0)[:, None]*1., \
            torch.cat((GO, GE), dim=0)[:, None]*1.
        return X, T, S, Y, G


def TWO_STAGE_PROPEN(X, S, Y, T, G,
            X_val, S_val, Y_val, T_val, G_val,
            args, cross_fitting=True):
    # validation
    XO_val, SO_val, YO_val, TO_val = X_val[G_val == 0], S_val[G_val == 0], Y_val[G_val == 0], T_val[G_val == 0]
    XE_val, SE_val, TE_val = X_val[G_val == 1], S_val[G_val == 1], T_val[G_val == 1]

    # construct D1 and D2
    if cross_fitting is True:
        X, X2, S, S2, Y, Y2, T, T2, G, G2 = train_test_split(X, S, Y, T, G, test_size=0.5)

    # train on D1
    XO, SO, YO, TO = X[G == 0], S[G == 0], Y[G == 0], T[G == 0]
    XE, SE, TE = X[G == 1], S[G == 1], T[G == 1]
    mlp = PROPEN_MLP(x_dim=X.shape[1], args=args)
    mlp.fit(XO, TO, SO, YO, XE, TE, SE,
            XO_val, TO_val, SO_val, YO_val, XE_val, TE_val, SE_val)
    # cate_predict = mlp.predict(predict_X)

    # print(cate_1.shape)
    if cross_fitting is True:
        # train on D2
        XO, SO, YO, TO = X2[G2 == 0], S2[G2 == 0], Y2[G2 == 0], T2[G2 == 0]
        XE, SE, TE = X2[G2 == 1], S2[G2 == 1], T2[G2 == 1]
        mlp2 = PROPEN_MLP(x_dim=X.shape[1], args=args)
        mlp2.fit(XO, TO, SO, YO, XE, TE, SE,
                 XO_val, TO_val, SO_val, YO_val, XE_val,TE_val, SE_val)
        # cate_predict2 = mlp2.predict(predict_X)

    # average
    def predict(input):
        if cross_fitting is True:
            return (mlp2.predict(input) + mlp.predict(input)) / 2
        else:
            return mlp.predict(input)

    return predict


def IHDP(args):
    from utils.IHDP_datasets import LongTermIHDP
    from utils.dgp import set_all_seeds
    import warnings

    warnings.simplefilter('ignore')
    ite, ate = [], []

    lti = LongTermIHDP()
    for i, (train, valid, test) in enumerate(lti.get_train_valid_test()):
        set_all_seeds(args.seed)

        g_train, t_train, s_train, y_train, _, _, y1_train, y0_train, x_train, _ = train
        g_valid, t_valid, s_valid, y_valid, _, _, y1_valid, y0_valid, x_valid, _ = valid
        t_test, s_test, y_test, _, _, y1_test, y0_test, x_test, _ = test
        # ground true
        # ite_mr = np.squeeze(ite_mr)
        iteO = y1_test - y0_test

        ite_predictor = TWO_STAGE_PROPEN(x_train, np.squeeze(s_train), np.squeeze(y_train),
                                np.squeeze(t_train), np.squeeze(g_train),
                                x_valid, np.squeeze(s_valid), np.squeeze(y_valid),
                                np.squeeze(t_valid), np.squeeze(g_valid),
                                args=args, cross_fitting=True)
        ite_mr = ite_predictor(x_test)

        print(ite_mr.shape, iteO.shape)
        print('seed '+ str(i) +' MR_LTCE: ' + str(PEHE_ITE(ite_mr, iteO)) + ' ' + str(RMSE_ATE(ite_mr, iteO)))

        ite.append(PEHE_ITE(ite_mr, iteO))
        ate.append(RMSE_ATE(ite_mr, iteO))
    return ite, ate


def NEWS(args):
    from utils.NEWS_dataset import LongTermNEWS
    from utils.dgp import set_all_seeds
    import warnings

    warnings.simplefilter('ignore')
    ite_outs, ate_outs = [], []
    ite_ins, ate_ins = [], []
    ite_trains, ate_trains = [], []
    ite_valids, ate_valids = [], []

    lti = LongTermNEWS()
    for i, (train, valid, test) in enumerate(lti.get_train_valid_test()):
        set_all_seeds(args.seed)
        g_train, t_train, s_train, y_train, y1_train, y0_train, x_train = train
        g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid = valid
        t_test, s_test, y_test, y1_test, y0_test, x_test = test

        x_train_valid = np.concatenate((x_train,x_valid),axis=0)

        # ground true
        # ite_mr = np.squeeze(ite_mr)
        iteout = y1_test - y0_test
        itetrain = y1_train - y0_train
        itevalid = y1_valid - y0_valid
        ite_in = np.concatenate((itetrain,itevalid), axis=0)

        ite_predictor = TWO_STAGE_PROPEN(x_train, np.squeeze(s_train), np.squeeze(y_train),
                                np.squeeze(t_train), np.squeeze(g_train),
                                x_valid, np.squeeze(s_valid), np.squeeze(y_valid),
                                np.squeeze(t_valid), np.squeeze(g_valid),
                                args=args, cross_fitting=args.cross_fitting)
        ite_mr = ite_predictor(x_test)
        ite_mr_in = ite_predictor(x_train_valid)
        ite_mr_train = ite_predictor(x_train)
        ite_mr_valid = ite_predictor(x_valid)

        # print(ite_mr.shape, iteO.shape)
        print('seed '+ str(i) +' MR_LTCE: ' + str(PEHE_ITE(ite_mr, iteout)) + ' ' + str(RMSE_ATE(ite_mr, iteout)))

        ite_outs.append(PEHE_ITE(ite_mr, iteout))
        ate_outs.append(RMSE_ATE(ite_mr, iteout))
        ite_ins.append(PEHE_ITE(ite_mr_in, ite_in))
        ate_ins.append(RMSE_ATE(ite_mr_in, ite_in))
        ite_trains.append(PEHE_ITE(ite_mr_train, itetrain))
        ate_trains.append(RMSE_ATE(ite_mr_train, itetrain))
        ite_valids.append(PEHE_ITE(ite_mr_valid, itevalid))
        ate_valids.append(RMSE_ATE(ite_mr_valid, itevalid))

    if args.save:
        import csv
        save_dic = {}
        save_dic['method'] = 'two_stage_pro'
        save_dic['args'] = str(args)

        save_dic['ite_out'] = np.mean(ite_outs)
        save_dic['ite_out_std'] = np.std(ite_outs)
        save_dic['ate_out'] = np.mean(ate_outs)
        save_dic['ate_out_std'] = np.std(ate_outs)

        save_dic['ite_in'] = np.mean(ite_ins)
        save_dic['ite_in_std'] = np.std(ite_ins)
        save_dic['ate_in'] = np.mean(ate_ins)
        save_dic['ate_in_std'] = np.std(ate_ins)

        save_dic['ite_train'] = np.mean(ite_trains)
        save_dic['ite_train_std'] = np.std(ite_trains)
        save_dic['ate_train'] = np.mean(ate_trains)
        save_dic['ate_train_std'] = np.std(ate_trains)

        save_dic['ite_valid'] = np.mean(ite_valids)
        save_dic['ite_valid_std'] = np.std(ite_valids)
        save_dic['ate_valid'] = np.mean(ate_valids)
        save_dic['ate_valid_std'] = np.std(ate_valids)

        # 将字典保存为 CSV 文件
        field_names = ['method','args', 'ite_out', 'ite_out_std', 'ate_out', 'ate_out_std',
                       'ite_in', 'ite_in_std', 'ate_in', 'ate_in_std',
                       'ite_train', 'ite_train_std', 'ate_train', 'ate_train_std',
                       'ite_valid', 'ite_valid_std', 'ate_valid', 'ate_valid_std']
        with open('../results/NEWS'+'/newsresult.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()  # 写入字段名
            writer.writerows([save_dic])  # 写入数据

    # return ite, ate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=7, help='Use CUDA training.')
    parser.add_argument('--device', type=str, default='cuda', help='Use CUDA training.')
    parser.add_argument('--seed', type=int, default=123, help='seed all set to 123 for all methods, do NOT change')
    parser.add_argument('--cross_fitting', type=int, default=0, help='seed all set to 123 for all methods, do NOT change')
    parser.add_argument('--dataset', type=str, default='NEWS')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--base_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--normalization', type=int, default=1, help='1 means done and 0 means undone')
    parser.add_argument('--normalization_x', type=int, default=0, help='1 means done and 0 means undone')
    parser.add_argument('--early_stop', type=int, default=1, help='1 means done and 0 means undone')
    parser.add_argument('--early_stop_iter', type=int, default=5, help='stop when vali loss not decrease with num iters')
    parser.add_argument('--activation', type=str, default='lrelu', help='Number of hidden units.')
    parser.add_argument('--slope', type=int, default=1., help='param of lrelu')
    # first-step hyper-param
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size when training.')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--n_layers', type=int, default=3, help='epoch of training discriminator')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--reweight', type=float, default=1, help='using weighted regression')
    # second-step hyper-param
    parser.add_argument('--lr_2', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--weight_decay_2', type=float, default=1e-4, help='Weight decay.')
    parser.add_argument('--batch_size_2', type=int, default=128, help='batch size when training.')
    parser.add_argument('--hidden_dim_2', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('--n_layers_2', type=int, default=3, help='epoch of training discriminator')
    parser.add_argument('--dropout_2', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    # save result
    parser.add_argument('--save', type=int, default=1, help='Dropout rate (1 - keep probability).')


    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.dataset == 'IHDP':
        ite, ate = IHDP(args)
    elif args.dataset == 'NEWS':
        NEWS(args)

    # print(np.mean(ite), np.std(ite))
    # print(np.mean(ate), np.mean(ate))


    # best 效果烂，重新跑pehe最小的
    # 25.48818510320269 6.64936215550349
    # 16.29473910785234 16.29473910785234

    # pehe最小
    # 13.895107196142217 6.709357366436829
    # 4.183083299071905 4.183083299071905

    # 10 best
    # 15.742104785290516 7.344396364116518
    # 5.4554507179152 5.4554507179152