import numpy as np
from sklearn.model_selection import train_test_split


class IHDP(object):
    def __init__(self, path_data="datasets/IHDP/csv", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this binary feature is in {1, 2}
            x[:, 13] -= 1
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats

    def get_all_train_valid_test(self, i):
        data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
        t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
        # this binary feature is in {1, 2}
        x[:, 13] -= 1
        idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
        itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
        train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
        valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
        test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
        return train, valid, test, self.contfeats, self.binfeats

class LongTermIHDP(object):
    def __init__(self, path_data="../datasets/IHDP/ltcsv", replications=10):
        self.path_data = path_data
        self.replications = replications
        self.u_dim = int(25 / 3)
        self.x_dim = 25 - int(25 / 3)

    def __iter__(self):
        for i in range(self.replications):
            train_data = np.loadtxt(self.path_data + '/ihdp_lt_train_' + str(i + 1) + '.csv', delimiter=',')
            valid_data = np.loadtxt(self.path_data + '/ihdp_lt_valid_' + str(i + 1) + '.csv', delimiter=',')
            test_data = np.loadtxt(self.path_data + '/ihdp_lt_test_' + str(i + 1) + '.csv', delimiter=',')
            g_train, t_train, s_train, y_train, s1_train, s0_train, \
                y1_train, y0_train, x_train, u_train = \
                train_data[:, 0][:, np.newaxis], train_data[:, 1][:, np.newaxis], train_data[:, 2][:, np.newaxis], \
                    train_data[:, 3][:, np.newaxis], train_data[:, 4][:, np.newaxis], train_data[:, 5][:, np.newaxis], \
                    train_data[:, 6][:, np.newaxis], train_data[:, 7][:, np.newaxis], \
                    train_data[:, 8:self.x_dim + 8], train_data[:, self.x_dim + 8:self.x_dim + 8 + self.u_dim]
            g_valid, t_valid, s_valid, y_valid, s1_valid, s0_valid, \
                y1_valid, y0_valid, x_valid, u_valid = \
                valid_data[:, 0][:, np.newaxis], valid_data[:, 1][:, np.newaxis], valid_data[:, 2][:, np.newaxis], \
                    valid_data[:, 3][:, np.newaxis], valid_data[:, 4][:, np.newaxis], valid_data[:, 5][:, np.newaxis], \
                    valid_data[:, 6][:, np.newaxis], valid_data[:, 7][:, np.newaxis], \
                    valid_data[:, 8:self.x_dim + 8], valid_data[:, self.x_dim + 8:self.x_dim + 8 + self.u_dim]
            t_test, s_test, y_test, s1_test, s0_test, y1_test, y0_test, x_test, u_test = \
                test_data[:, 0][:, np.newaxis], test_data[:, 1][:, np.newaxis], test_data[:, 2][:, np.newaxis], \
                    test_data[:, 3][:, np.newaxis], test_data[:, 4][:, np.newaxis], test_data[:, 5][:, np.newaxis], \
                    test_data[:, 6][:, np.newaxis], test_data[:, 7:self.x_dim + 7], test_data[:, self.x_dim + 7:self.x_dim + 7 + self.u_dim]
            # print(x_test.shape, u_test.shape)
            yield (g_train, t_train, s_train, y_train, s1_train, s0_train, y1_train, y0_train, x_train, u_train), \
                (g_valid, t_valid, s_valid, y_valid, s1_valid, s0_valid, y1_valid, y0_valid, x_valid, u_valid), \
                (t_test, s_test, y_test, s1_test, s0_test, y1_test, y0_test, x_test, u_test)


    def get_train_valid_test(self):
        for i in range(self.replications):
            train_data = np.loadtxt(self.path_data + '/ihdp_lt_train_' + str(i + 1) + '.csv', delimiter=',')
            valid_data = np.loadtxt(self.path_data + '/ihdp_lt_valid_' + str(i + 1) + '.csv', delimiter=',')
            test_data = np.loadtxt(self.path_data + '/ihdp_lt_test_' + str(i + 1) + '.csv', delimiter=',')
            g_train, t_train, s_train, y_train, s1_train, s0_train, \
                y1_train, y0_train, x_train, u_train = \
                train_data[:, 0][:, np.newaxis], train_data[:, 1][:, np.newaxis], train_data[:, 2][:, np.newaxis], \
                    train_data[:, 3][:, np.newaxis], train_data[:, 4][:, np.newaxis], train_data[:, 5][:, np.newaxis], \
                    train_data[:, 6][:, np.newaxis], train_data[:, 7][:, np.newaxis], \
                    train_data[:, 8:self.x_dim + 8], train_data[:, self.x_dim + 8:self.x_dim + 8 + self.u_dim]
            g_valid, t_valid, s_valid, y_valid, s1_valid, s0_valid, \
                y1_valid, y0_valid, x_valid, u_valid = \
                valid_data[:, 0][:, np.newaxis], valid_data[:, 1][:, np.newaxis], valid_data[:, 2][:, np.newaxis], \
                    valid_data[:, 3][:, np.newaxis], valid_data[:, 4][:, np.newaxis], valid_data[:, 5][:, np.newaxis], \
                    valid_data[:, 6][:, np.newaxis], valid_data[:, 7][:, np.newaxis], \
                    valid_data[:, 8:self.x_dim + 8], valid_data[:, self.x_dim + 8:self.x_dim + 8 + self.u_dim]
            t_test, s_test, y_test, s1_test, s0_test, y1_test, y0_test, x_test, u_test = \
                test_data[:, 0][:, np.newaxis], test_data[:, 1][:, np.newaxis], test_data[:, 2][:, np.newaxis], \
                    test_data[:, 3][:, np.newaxis], test_data[:, 4][:, np.newaxis], test_data[:, 5][:, np.newaxis], \
                    test_data[:, 6][:, np.newaxis], test_data[:, 7:self.x_dim + 7], test_data[:, self.x_dim + 7:self.x_dim + 7 + self.u_dim]
            # print(x_test.shape, u_test.shape)
            yield (g_train, t_train, s_train, y_train, s1_train, s0_train, y1_train, y0_train, x_train, u_train), \
                (g_valid, t_valid, s_valid, y_valid, s1_valid, s0_valid, y1_valid, y0_valid, x_valid, u_valid), \
                (t_test, s_test, y_test, s1_test, s0_test, y1_test, y0_test, x_test, u_test)

    def get_all_train_valid_test(self, i):
        train_data = np.loadtxt(self.path_data + '/ihdp_lt_train_' + str(i + 1) + '.csv', delimiter=',')
        valid_data = np.loadtxt(self.path_data + '/ihdp_lt_valid_' + str(i + 1) + '.csv', delimiter=',')
        test_data = np.loadtxt(self.path_data + '/ihdp_lt_test_' + str(i + 1) + '.csv', delimiter=',')
        g_train, t_train, s_train, y_train, s1_train, s0_train, \
            y1_train, y0_train, x_train, u_train = \
            train_data[:, 0][:, np.newaxis], train_data[:, 1][:, np.newaxis], train_data[:, 2][:, np.newaxis], \
                train_data[:, 3][:, np.newaxis], train_data[:, 4][:, np.newaxis], train_data[:, 5][:, np.newaxis], \
                train_data[:, 6][:, np.newaxis], train_data[:, 7][:, np.newaxis], \
                train_data[:, 8:self.x_dim + 8], train_data[:, self.x_dim + 8:self.x_dim + 8 + self.u_dim]
        g_valid, t_valid, s_valid, y_valid, s1_valid, s0_valid, \
            y1_valid, y0_valid, x_valid, u_valid = \
            valid_data[:, 0][:, np.newaxis], valid_data[:, 1][:, np.newaxis], valid_data[:, 2][:, np.newaxis], \
                valid_data[:, 3][:, np.newaxis], valid_data[:, 4][:, np.newaxis], valid_data[:, 5][:, np.newaxis], \
                valid_data[:, 6][:, np.newaxis], valid_data[:, 7][:, np.newaxis], \
                valid_data[:, 8:self.x_dim + 8], valid_data[:, self.x_dim + 8:self.x_dim + 8 + self.u_dim]
        t_test, s_test, y_test, s1_test, s0_test, y1_test, y0_test, x_test, u_test = \
            test_data[:, 0][:, np.newaxis], test_data[:, 1][:, np.newaxis], test_data[:, 2][:, np.newaxis], \
                test_data[:, 3][:, np.newaxis], test_data[:, 4][:, np.newaxis], test_data[:, 5][:, np.newaxis], \
                test_data[:, 6][:, np.newaxis], test_data[:, 7:self.x_dim + 7], test_data[:, self.x_dim + 7:self.x_dim + 7 + self.u_dim]
        # print(x_test.shape, u_test.shape)
        return (g_train, t_train, s_train, y_train, s1_train, s0_train, y1_train, y0_train, x_train, u_train), \
            (g_valid, t_valid, s_valid, y_valid, s1_valid, s0_valid, y1_valid, y0_valid, x_valid, u_valid), \
            (t_test, s_test, y_test, s1_test, s0_test, y1_test, y0_test, x_test, u_test)


def gen_long_term_IHDP():
    '''
    Data generation follow Bayesian NonParameteric modeling
    TIME: 2025 05 12
    '''
    np.random.seed(123)
    N=747
    values = [0, 1, 2, 3, 4]
    probabilities = [0.5, 0.2, 0.15, 0.1, 0.05]

    # values = [0, 0.1, 0.2, 0.3, 0.4] # [0, 1, 2, 3, 4]
    # probabilities = [0.6, 0.1, 0.1, 0.1, 0.1]

    # binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # which features are continuous
    # contfeats = [i for i in range(25) if i not in binfeats]
    for i in range(10):
        data = np.loadtxt("../datasets/IHDP/csv/ihdp_npci_" + str(i + 1) + '.csv', delimiter=',')
        t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
        # covariate: observed : treated = 4:1
        index = [k for k in range(25)]
        np.random.shuffle(index)
        ratio = 3
        x_index = index[:25-int(25 / ratio)]
        u_index = index[25-int(25 / ratio):]
        print('num of u', len(u_index))
        print('num of x', len(x_index))
        x_dim, u_dim = len(x_index), len(u_index)
        u = x[:,u_index]
        x = x[:,x_index]
        # generate short/long term potential outcome
        # betax_s1 = np.random.choice(values, x_dim, p=probabilities)
        # betax_s0 = np.random.choice(values, x_dim, p=probabilities)
        # betax_y1 = np.random.choice(values, x_dim, p=probabilities)
        # betax_y0 = np.random.choice(values, x_dim, p=probabilities)
        # betaxu_y = np.random.choice(values, x_dim, p=probabilities)
        # betau = np.random.choice(values, u_dim, p=probabilities)
        # betaux_y = np.random.choice(values, u_dim, p=probabilities)

        betax_s1 = np.random.randn(x_dim)+1
        betax_s0 = np.random.randn(x_dim)/3
        betax_y1 = np.random.randn(x_dim)+1
        betax_y0 = np.random.randn(x_dim)/3
        betaxu_y = np.random.randn(x_dim)+1
        betau = np.random.randn(u_dim)+1
        betaux_y = np.random.randn(u_dim)+1

        noise_s = np.random.normal(0,1,size=N)
        noise_y = np.random.normal(0,1,size=N)
        # surface A
        # print(x.shape, betax_s1.shape, np.dot(x, betax_s1).shape,  np.dot(u, betau).shape, noise_s.shape, s1.shape)
        # s1 = np.dot(x, betax_s1) + np.dot(u, betau) + np.dot(u, betau)*np.dot(x, betax_s0) + 4 + noise_s
        # s0 = np.dot(x, betax_s1) + np.dot(u, betau) + np.dot(u, betau)*np.dot(x, betax_s0) + noise_s
        # y1 = np.dot(x, betax_y1) - s1 + 2 * np.dot(u, betau) + 2 * np.dot(u, betau)*np.dot(x, betax_s0) + 8 + noise_y
        # y0 = np.dot(x, betax_y1) - s0 + 2 * np.dot(u, betau) + 2 * np.dot(u, betau)*np.dot(x, betax_s0) + noise_y
        # surface B
        # print(x.shape, betax_s1.shape, np.dot(x, betax_s1).shape,  np.dot(u, betau).shape, noise_s.shape, s1.shape)
        s1 = np.dot(x, betax_s1) + np.dot(u, betau) + 4 + np.dot(u, betau)*np.dot(x, betaxu_y) + noise_s
        s0 = np.exp(np.dot(x + 0.5, betax_s0))+ np.dot(u, betau) + np.dot(u, betau)*np.dot(x, betaxu_y) + noise_s
        y1 = np.dot(x, betax_y1) - s1 + 2 * np.dot(u, betau) + 2* np.dot(u, betau)*np.dot(x, betaxu_y) + 8 + noise_y
        y0 = np.exp(np.dot(x + 0.5, betax_y0)) - s0 + 2 * np.dot(u, betau) + 2* np.dot(u, betau)*np.dot(x, betaxu_y) + noise_y
        # shape transform (747,) => (747,1)
        s1, s0, y1, y0 = s1[:,None], s0[:,None], y1[:,None], y0[:,None]

        # print(np.mean(np.dot(u, betau)))

        beta_g_exp = np.random.uniform(-0.5,0.5,size=x_dim)
        offset_g = 1.       # set offset=1 to make group ratio appriximate 1:2
        pro_g = 1 / (1+np.exp(np.dot(x, beta_g_exp) + offset_g))
        print(np.sum(pro_g<0.05) + np.sum(pro_g>0.95))
        g = np.random.binomial(1, pro_g)[:,None]
        print(np.sum(g)/g.shape[0])

        # reuse original treatment for observational data
        # use generated treatment for experimental data
        beta_t_exp = np.random.uniform(-0.5, .5,size=x_dim)
        offset = -.25
        p = 1 / (1+np.exp(np.dot(x, beta_t_exp) + offset))
        # print(np.sum(p>0.95) + np.sum(p<0.05))
        # print(beta_t_exp.shape, x.shape, p.shape)
        # exp:obs approximates 1:2
        te = np.random.binomial(1, p)[:,None]
        t[g==1] = te[g==1]

        beta_t_obs = np.random.uniform(-0.5, .5, size=x_dim+u_dim)
        offset = -1.
        po = 1 / (1 + np.exp(np.dot(np.concatenate((x,u), axis=1), beta_t_obs) + offset))
        # print(np.sum(po > 0.95) + np.sum(po < 0.05))
        # print(beta_t_exp.shape, x.shape, p.shape)
        # exp:obs approximates 1:2
        to = np.random.binomial(1, po)[:, None]
        t[g == 0] = to[g == 0]

        s = np.where(t==1, s1, s0)
        y = np.where(t==1, y1, y0)

        # split train/valid/test = 63:27:10, following TarNET
        idxtrain, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.1)
        train_index, valid_index = train_test_split(idxtrain, test_size=0.3)

        # train
        t_test, s_test, y_test, s1_test, s0_test, y1_test, y0_test, x_test, u_test =\
            t[test_index,], s[test_index,], y[test_index,], s1[test_index,], s0[test_index,],\
                y1[test_index,], y0[test_index,], x[test_index,], u[test_index,]
        g_train, t_train, s_train, y_train, s1_train, s0_train, y1_train, y0_train, x_train, u_train = \
            g[train_index, ], t[train_index, ], s[train_index, ], y[train_index, ], s1[train_index, ], s0[train_index, ], \
                y1[train_index, ], y0[train_index, ], x[train_index, ], u[train_index, ]
        g_valid, t_valid, s_valid, y_valid, s1_valid, s0_valid, y1_valid, y0_valid, x_valid, u_valid = \
            g[valid_index,], t[valid_index,], s[valid_index,], y[valid_index,], s1[valid_index,], s0[valid_index,], \
                y1[valid_index,], y0[valid_index,], x[valid_index,], u[valid_index,]
        # print(g.shape,t.shape,s.shape,y.shape,s1.shape,y1.shape,x.shape,u.shape)
        # print(t_train.shape,t_test.shape,s.shape,y.shape,s1.shape,y1.shape,x.shape,u.shape)
        print(t_test.shape, t_train.shape, t_valid.shape)


        save_data_train = np.concatenate((g_train,t_train, s_train, y_train, s1_train, s0_train,
                                          y1_train, y0_train, x_train, u_train), axis=1)
        save_data_valid = np.concatenate((g_valid, t_valid, s_valid, y_valid, s1_valid, s0_valid,
                                          y1_valid, y0_valid, x_valid, u_valid), axis=1)
        save_data_test = np.concatenate((t_test, s_test, y_test, s1_test, s0_test,
                                          y1_test, y0_test, x_test, u_test), axis=1)
        # print(save_data_train.shape)
        # print(np.mean(y1-y0))

        np.savetxt('../datasets/IHDP/ltcsv/ihdp_lt_train_' + str(i+1) + '.csv', save_data_train, delimiter=',', fmt='%.2f')
        np.savetxt('../datasets/IHDP/ltcsv/ihdp_lt_valid_' + str(i+1) + '.csv', save_data_valid, delimiter=',', fmt='%.2f')
        np.savetxt('../datasets/IHDP/ltcsv/ihdp_lt_test_' + str(i+1) + '.csv', save_data_test, delimiter=',', fmt='%.2f')


if __name__ == '__main__':
    gen_long_term_IHDP()
    lti = LongTermIHDP()
    for i, (train, valid, test) in enumerate(lti.get_train_valid_test()):
        g, t_train, s_train, y_train, s1_train, s0_train, y1_train, y0_train, x_train, u_train = train
        t_test, s_test, y_test, s1_test, s0_test, y1_test, y0_test, x_test, u_test = test
        # print(np.mean(y1_train-y0_train))
        # print(np.mean(y1_test-y0_test))