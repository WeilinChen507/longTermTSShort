import numpy as np
import torch
import os

from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split


class LongTermNEWS(object):
    def __init__(self, path_data="../datasets/news", replications=10):
        self.path_data = path_data
        self.replications = replications
        # self.x_dim = 332 #399
        # self.u_dim = 166 #99 # not use

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/data_matrix.txt', delimiter=',')
            idx_test = np.loadtxt(self.path_data + '/eval/'+str(i)+'/idx_test.txt', delimiter=',', dtype=int)
            idx_valid = np.loadtxt(self.path_data + '/eval/'+str(i)+'/idx_valid.txt', delimiter=',', dtype=int)
            idx_train = np.loadtxt(self.path_data + '/eval/'+str(i)+'/idx_train.txt', delimiter=',', dtype=int)
            g, t, s, y, y1, y0, x = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6:]
            g_train, g_valid, g_test = g[idx_train][:, np.newaxis], g[idx_valid][:, np.newaxis], g[idx_test][:, np.newaxis]
            t_train, t_valid, t_test = t[idx_train][:, np.newaxis], t[idx_valid][:, np.newaxis], t[idx_test][:, np.newaxis]
            s_train, s_valid, s_test = s[idx_train][:, np.newaxis], s[idx_valid][:, np.newaxis], s[idx_test][:, np.newaxis]
            y_train, y_valid, y_test = y[idx_train][:, np.newaxis], y[idx_valid][:, np.newaxis], y[idx_test][:, np.newaxis]
            y1_train, y1_valid, y1_test = y1[idx_train][:, np.newaxis], y1[idx_valid][:, np.newaxis], y1[idx_test][:, np.newaxis]
            y0_train, y0_valid, y0_test = y0[idx_train][:, np.newaxis], y0[idx_valid][:, np.newaxis], y0[idx_test][:, np.newaxis]
            x_train, x_valid, x_test = x[idx_train, :], x[idx_valid, :], x[idx_test, :]

            yield (g_train, t_train, s_train, y_train, y1_train, y0_train, x_train), \
                (g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid), \
                (t_test, s_test, y_test, y1_test, y0_test, x_test)


    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/data_matrix.txt', delimiter=',')
            idx_test = np.loadtxt(self.path_data + '/eval/' + str(i) + '/idx_test.txt', delimiter=',', dtype=int)
            idx_valid = np.loadtxt(self.path_data + '/eval/' + str(i) + '/idx_valid.txt', delimiter=',', dtype=int)
            idx_train = np.loadtxt(self.path_data + '/eval/' + str(i) + '/idx_train.txt', delimiter=',', dtype=int)
            g, t, s, y, y1, y0, x = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6:]
            g_train, g_valid, g_test = g[idx_train][:, np.newaxis], g[idx_valid][:, np.newaxis], g[idx_test][:, np.newaxis]
            t_train, t_valid, t_test = t[idx_train][:, np.newaxis], t[idx_valid][:, np.newaxis], t[idx_test][:, np.newaxis]
            s_train, s_valid, s_test = s[idx_train][:, np.newaxis], s[idx_valid][:, np.newaxis], s[idx_test][:, np.newaxis]
            y_train, y_valid, y_test = y[idx_train][:, np.newaxis], y[idx_valid][:, np.newaxis], y[idx_test][:, np.newaxis]
            y1_train, y1_valid, y1_test = y1[idx_train][:, np.newaxis], y1[idx_valid][:, np.newaxis], y1[idx_test][:, np.newaxis]
            y0_train, y0_valid, y0_test = y0[idx_train][:, np.newaxis], y0[idx_valid][:, np.newaxis], y0[idx_test][:, np.newaxis]
            x_train, x_valid, x_test = x[idx_train, :], x[idx_valid, :], x[idx_test, :]


            yield (g_train, t_train, s_train, y_train, y1_train, y0_train, x_train), \
                (g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid), \
                (t_test, s_test, y_test, y1_test, y0_test, x_test)

    def get_all_train_valid_test(self, i):
        data = np.loadtxt(self.path_data + '/data_matrix.txt', delimiter=',')
        idx_test = np.loadtxt(self.path_data + '/eval/' + str(i) + '/idx_test.txt', delimiter=',', dtype=int)
        idx_valid = np.loadtxt(self.path_data + '/eval/' + str(i) + '/idx_valid.txt', delimiter=',', dtype=int)
        idx_train = np.loadtxt(self.path_data + '/eval/' + str(i) + '/idx_train.txt', delimiter=',', dtype=int)
        g, t, s, y, y1, y0, x = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6:]
        g_train, g_valid, g_test = g[idx_train][:, np.newaxis], g[idx_valid][:, np.newaxis], g[idx_test][:, np.newaxis]
        t_train, t_valid, t_test = t[idx_train][:, np.newaxis], t[idx_valid][:, np.newaxis], t[idx_test][:, np.newaxis]
        s_train, s_valid, s_test = s[idx_train][:, np.newaxis], s[idx_valid][:, np.newaxis], s[idx_test][:, np.newaxis]
        y_train, y_valid, y_test = y[idx_train][:, np.newaxis], y[idx_valid][:, np.newaxis], y[idx_test][:, np.newaxis]
        y1_train, y1_valid, y1_test = y1[idx_train][:, np.newaxis], y1[idx_valid][:, np.newaxis], y1[idx_test][:,
                                                                                                  np.newaxis]
        y0_train, y0_valid, y0_test = y0[idx_train][:, np.newaxis], y0[idx_valid][:, np.newaxis], y0[idx_test][:,
                                                                                                  np.newaxis]
        x_train, x_valid, x_test = x[idx_train, :], x[idx_valid, :], x[idx_test, :]

        return (g_train, t_train, s_train, y_train, y1_train, y0_train, x_train), \
            (g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid), \
            (t_test, s_test, y_test, y1_test, y0_test, x_test)

    def get_tune_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/data_matrix.txt', delimiter=',')
            idx_test = np.loadtxt(self.path_data + '/tune/' + str(i) + '/idx_test.txt', delimiter=',', dtype=int)
            idx_valid = np.loadtxt(self.path_data + '/tune/' + str(i) + '/idx_valid.txt', delimiter=',', dtype=int)
            idx_train = np.loadtxt(self.path_data + '/tune/' + str(i) + '/idx_train.txt', delimiter=',', dtype=int)
            g, t, s, y, y1, y0, x = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6:]
            g_train, g_valid, g_test = g[idx_train][:, np.newaxis], g[idx_valid][:, np.newaxis], g[idx_test][:, np.newaxis]
            t_train, t_valid, t_test = t[idx_train][:, np.newaxis], t[idx_valid][:, np.newaxis], t[idx_test][:, np.newaxis]
            s_train, s_valid, s_test = s[idx_train][:, np.newaxis], s[idx_valid][:, np.newaxis], s[idx_test][:, np.newaxis]
            y_train, y_valid, y_test = y[idx_train][:, np.newaxis], y[idx_valid][:, np.newaxis], y[idx_test][:, np.newaxis]
            y1_train, y1_valid, y1_test = y1[idx_train][:, np.newaxis], y1[idx_valid][:, np.newaxis], y1[idx_test][:,
                                                                                                      np.newaxis]
            y0_train, y0_valid, y0_test = y0[idx_train][:, np.newaxis], y0[idx_valid][:, np.newaxis], y0[idx_test][:,
                                                                                                      np.newaxis]
            x_train, x_valid, x_test = x[idx_train, :], x[idx_valid, :], x[idx_test, :]

            yield (g_train, t_train, s_train, y_train, y1_train, y0_train, x_train), \
                (g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid), \
                (t_test, s_test, y_test, y1_test, y0_test, x_test)


    def get_all_train_valid_test(self, i=0):
        data = np.loadtxt(self.path_data + '/data_matrix.txt', delimiter=',')
        idx_test = np.loadtxt(self.path_data + '/tune/' + str(i) + '/idx_test.txt', delimiter=',', dtype=int)
        idx_valid = np.loadtxt(self.path_data + '/tune/' + str(i) + '/idx_valid.txt', delimiter=',', dtype=int)
        idx_train = np.loadtxt(self.path_data + '/tune/' + str(i) + '/idx_train.txt', delimiter=',', dtype=int)
        g, t, s, y, y1, y0, x = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6:]
        g_train, g_valid, g_test = g[idx_train][:, np.newaxis], g[idx_valid][:, np.newaxis], g[idx_test][:, np.newaxis]
        t_train, t_valid, t_test = t[idx_train][:, np.newaxis], t[idx_valid][:, np.newaxis], t[idx_test][:, np.newaxis]
        s_train, s_valid, s_test = s[idx_train][:, np.newaxis], s[idx_valid][:, np.newaxis], s[idx_test][:, np.newaxis]
        y_train, y_valid, y_test = y[idx_train][:, np.newaxis], y[idx_valid][:, np.newaxis], y[idx_test][:, np.newaxis]
        y1_train, y1_valid, y1_test = y1[idx_train][:, np.newaxis], y1[idx_valid][:, np.newaxis], y1[idx_test][:,
                                                                                                  np.newaxis]
        y0_train, y0_valid, y0_test = y0[idx_train][:, np.newaxis], y0[idx_valid][:, np.newaxis], y0[idx_test][:,
                                                                                                  np.newaxis]
        x_train, x_valid, x_test = x[idx_train, :], x[idx_valid, :], x[idx_test, :]

        return (g_train, t_train, s_train, y_train, y1_train, y0_train, x_train), \
            (g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid), \
            (t_test, s_test, y_test, y1_test, y0_test, x_test)


def generate():
    '''
    parameter fixed in 07/29/2024
    :return:
    '''
    parser = argparse.ArgumentParser(description='generate news data')
    parser.add_argument('--data_path', type=str, default='../datasets/news/news_pp.npy', help='data path')
    parser.add_argument('--save_dir', type=str, default='../datasets/news', help='dir to save generated data')
    parser.add_argument('--num_eval', type=int, default=10, help='num of dataset for evaluating the methods')
    parser.add_argument('--num_tune', type=int, default=10, help='num of dataset for tuning the parameters')

    args = parser.parse_args()
    save_path = args.save_dir

    # load data
    path = args.data_path
    news = np.load(path)
    #
    # # normalize data
    for _ in range(news.shape[1]):
        max_freq = max(news[:, _])
        news[:, _] = news[:, _] / max_freq

    num_data = news.shape[0]
    num_feature = news.shape[1]
    np.random.seed(123)
    print('before processing, dim of x', num_feature)
    print('before processing, num of data', num_data)

    # randomly choose u, ratio of u:x = 2:8
    index = [k for k in range(num_feature)]
    np.random.shuffle(index)
    ratio = 3
    u_index = index[:int(num_feature / ratio)]
    x_index = index[int(num_feature / ratio):]
    num_feature_x, num_feature_u = num_feature - int(num_feature / ratio), int(num_feature / ratio)
    print('feature dim: x,u=',num_feature_x, num_feature_u)
    # x, u = x[:, x_index], x[:,u_index]

    C1, C2, C3, C4 = 10, 10, 2, 2
    v1_xt = np.random.randn(num_feature_x)+1
    v1_xt = v1_xt/np.sqrt(np.sum(v1_xt**2))
    # v1_xt = v1_xt / C1
    v2_ut = np.random.randn(num_feature_u)+1
    v2_ut = v2_ut/np.sqrt(np.sum(v2_ut**2))
    # v2_ut = v2_ut / C2
    v3_xt_exp = np.random.randn(num_feature_x)+1
    v3_xt_exp = v3_xt_exp/np.sqrt(np.sum(v3_xt_exp**2))
    # v3_xt_exp = v3_xt_exp / C3
    v4_xg = np.random.randn(num_feature_x)
    v4_xg = v3_xt_exp/np.sqrt(np.sum(v4_xg**2))
    # v4_xg = v4_xg / C4

    v5_xs1 = np.random.randn(num_feature_x)+1
    v6_xs0 = np.random.randn(num_feature_x)+1
    v7_xy1 = np.random.randn(num_feature_x)+1
    v8_xy0 = np.random.randn(num_feature_x)+1
    v10_xusy = np.random.randn(num_feature_x)+1
    v9_usy = np.random.randn(num_feature_u)+1


    def x_t_obs(xu):
        x = xu[:, x_index]
        u = xu[:, u_index]
        offset = -2.
        p = 1 / (1 + np.exp(np.dot(x, v1_xt) +
                            np.dot(u, v2_ut) + offset))
        # p = 1 / (1 + np.exp(np.dot(x, v1_xt) + np.power(np.dot(x, v1_xt), 2) +
        #                     np.dot(u, v2_ut) + np.power(np.dot(u, v2_ut), 2) + offset))
        print(np.sum(p > 0.95) + np.sum(p < 0.15))
        print('extreme value of p(T|XU,O)', p)
        te = np.random.binomial(1, p)[:, None]
        return te

    def x_t_exp(xu):
        x = xu[:, x_index]
        # u = xu[:, u_index] no use
        # offset = np.random.uniform(-1, 1, size=1)
        offset = -0.5
        p = 1 / (1 + np.exp(np.dot(x, v3_xt_exp) + offset))
        print(np.sum(p > 0.9) + np.sum(p < 0.1))
        print('extreme value of p(T|X,E)', p)
        te = np.random.binomial(1, p)[:, None]
        return te

    def x_g(xu):
        x = xu[:, x_index]
        offset = 1.5
        p = 1 / (1 + np.exp(np.dot(x, v4_xg) + offset))
        print(np.sum(p > 0.9) + np.sum(p < 0.1))
        print('extreme value of p(G|X)', p)
        g = np.random.binomial(1, p)[:, None]
        print('ratio exp:obs', np.sum(g) / np.sum(1 - g))
        return g

    def t_x_y(xu):
        x = xu[:, x_index]
        u = xu[:, u_index]
        noise_s, noise_y = np.random.randn(x.shape[0]), np.random.randn(x.shape[0])

        s1 = 1*np.dot(x, v5_xs1) + 1*np.dot(np.power(x, 2), v5_xs1) + 1 * np.dot(u, v9_usy) + np.dot(u, v9_usy) * np.dot(x, v10_xusy) + noise_s
        s0 = 2*np.dot(x, v6_xs0) + 3*np.dot(np.power(x, 2), v6_xs0) + 1 * np.dot(u, v9_usy) + np.dot(u, v9_usy) * np.dot(x, v10_xusy) + noise_s

        y1 = 1*np.dot(x, v7_xy1) + 1*np.dot(np.power(x, 2), v7_xy1) + 4 - s1 + 2 * np.dot(u, v9_usy) + 2* np.dot(u, v9_usy) * np.dot(x, v10_xusy) + noise_y
        y0 = 2*np.dot(x, v8_xy0) + 3*np.dot(np.power(x, 2), v8_xy0) - s0 + 2 * np.dot(u, v9_usy) + 2*np.dot(u, v9_usy) * np.dot(x, v10_xusy) + noise_y
        print(np.mean(y1-y0))
        print(np.mean(y0))
        print(np.mean(y1))
        return s1, s0, y1, y0

    def news_matrix():
        xu = news
        # xu = xu + np.random.normal(0,0.5,(xu.shape[0], xu.shape[1]))
        # x2u = np.random.normal(0,1,(num_feature_x, num_feature_u))
        # xu[:, u_index] += np.dot(xu[:, x_index], x2u)
        # xu[:, u_index] = xu[:, u_index]/np.std(xu[:, u_index], axis=0)
        # print(xu[:, u_index].shape, np.std(xu[:, u_index], axis=0).shape)

        te = x_t_exp(xu)
        to = x_t_obs(xu)
        g = x_g(xu)
        t = np.zeros_like(te)
        t[g == 1] = te[g == 1]
        t[g == 0] = to[g == 0]

        x2u = np.random.normal(0,1,(num_feature_x, num_feature_u))
        # print(t.shape)
        # print( np.dot(xu[:, x_index], x2u).shape)
        # print( (t* (1-g) ).shape)
        # xu[:, u_index] += t*(1-g) * np.dot(xu[:, x_index], x2u)
        # xu[:, u_index] += t * np.dot(xu[:, x_index], x2u)
        # xu[:, u_index] = xu[:, u_index]/np.std(xu[:, u_index], axis=0)


        s1, s0, y1, y0 = t_x_y(xu)

        x = xu[:, x_index]
        u = xu[:, u_index]
        s = np.where(t == 1, s1[:, None], s0[:, None])
        y = np.where(t == 1, y1[:, None], y0[:, None])

        print('x u s y t shape', x.shape, u.shape, s.shape, y.shape, t.shape)

        data_matrix = np.concatenate((g, t, s, y, y1[:, None], y0[:, None], x), axis=1)
        print(data_matrix.shape)
        return data_matrix

    dm = news_matrix()
    # save_data_train, save_data_valid, save_data_test = dm[train_index,:], dm[valid_index,:], dm[test_index,:]
    # print('shape of saved dm', save_data_train.shape, save_data_valid.shape, save_data_test.shape)
    np.savetxt(save_path + '/data_matrix.txt', dm, delimiter=',', fmt='%.2f')
    # np.savetxt(save_path + '/data_matrix.pt', save_data_valid, delimiter=',', fmt='%.2f')
    # np.savetxt(save_path + '/data_matrix.pt', save_data_test, delimiter=',', fmt='%.2f')
    # torch.save(dm, save_path + '/data_matrix.pt')
    # torch.save(tg, save_path + '/t_grid.pt')

    # generate eval splitting
    for _ in range(args.num_eval):
        print('generating eval set: ', _)
        data_path = os.path.join(save_path, 'eval', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # randomly choose test_index / valid_index / train_index
        # ratio = 0.1 : 0.27 : 0.63
        idxtrain, test_index = train_test_split(np.arange(num_data), test_size=0.1)
        train_index, valid_index = train_test_split(idxtrain, test_size=0.3)

        print('train:valid:test=', train_index.shape, valid_index.shape, test_index.shape)
        np.savetxt(data_path + '/idx_train.txt', train_index, fmt='%d')
        np.savetxt(data_path + '/idx_valid.txt', valid_index, fmt='%d')
        np.savetxt(data_path + '/idx_test.txt', test_index, fmt='%d')
    #
    # # generate tuning splitting
    for _ in range(args.num_tune):
        print('generating eval set: ', _)
        data_path = os.path.join(save_path, 'tune', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # randomly choose test_index / valid_index / train_index
        # ratio = 0.1 : 0.27 : 0.63
        idxtrain, test_index = train_test_split(np.arange(num_data), test_size=0.1)
        train_index, valid_index = train_test_split(idxtrain, test_size=0.3)

        print('train:valid:test=', train_index.shape, valid_index.shape, test_index.shape)
        np.savetxt(data_path + '/idx_train.txt', train_index, fmt='%d')
        np.savetxt(data_path + '/idx_valid.txt', valid_index, fmt='%d')
        np.savetxt(data_path + '/idx_test.txt', test_index, fmt='%d')

if __name__ == "__main__":
    generate()
    ltn = LongTermNEWS()
    for i, (train, valid, test) in enumerate(ltn.get_train_valid_test()):
        g_train, t_train, s_train, y_train, y1_train, y0_train, x_train = train
        g_valid, t_valid, s_valid, y_valid, y1_valid, y0_valid, x_valid = valid
        t_test, s_test, y_test, y1_test, y0_test, x_test = test
        # print(g_train.shape)
