import numpy as np
import pandas as pd
from scipy.optimize import minimize

np.random.seed(1)


def outcome_ridge_bridge(Y, S2, S1, X, S3, lambda_):
    adj = np.hstack((S1, S2, X))
    adj -= np.mean(adj, axis=0)
    cond = np.hstack((S3, S2, X))
    condmean = np.mean(cond, axis=0)
    cond -= np.mean(cond, axis=0)
    vY = adj.T @ Y / len(Y)
    cov = adj.T @ cond / len(Y)

    b = np.linalg.solve(cov.T @ cov + lambda_ * np.eye(len(vY)) / len(Y), cov.T @ vY)

    def D(X):
        X = X - condmean
        return np.sum(X * b) + np.mean(Y)

    return D


def selection_ridge_bridge(obsS2, obsS1, obsX, obsS3, expS2, expS1, expX, expS3, lambda_):
    obsindicator = np.concatenate((np.ones(obsS1.shape[0], dtype=bool), np.zeros(expS1.shape[0], dtype=bool)))
    W = np.vstack((np.hstack((obsS3, obsS2, obsX)), np.hstack((expS3, expS2, expX))))
    W[~obsindicator] -= np.mean(W[obsindicator], axis=0)
    W[obsindicator] -= np.mean(W[obsindicator], axis=0)
    Z = np.hstack((obsS1, obsS2, obsX))
    Zmean = np.mean(Z, axis=0)
    Z -= np.mean(Z, axis=0)
    ZZ = np.hstack((np.ones((Z.shape[0], 1)), Z))

    def gmmobj(theta):
        eta = np.exp(ZZ @ theta)
        eta = np.clip(eta, None, 10)
        avg_moment = (W[obsindicator].T @ eta) / np.sum(obsindicator) - np.mean(W[~obsindicator], axis=0)
        avg_moment = np.append(avg_moment, np.mean(eta) - 1)
        return np.sum(avg_moment ** 2) + lambda_ / np.sum(obsindicator) * np.sum(theta[1:] ** 2)

    res = minimize(gmmobj, np.zeros(ZZ.shape[1]), method='L-BFGS-B', options={'ftol': 1e-19, 'maxiter': 20000})
    expcoef = res.x

    def D(X):
        X = X - Zmean
        X = np.insert(X, 0, 1)
        return min(np.exp(np.sum(X * expcoef)), 10)

    return D

#
# dimX = 10
# dimS = 5
# # nsim = 200
# nsim = 30
# # dimX += 1
# lambda_=0.05
#
# error_or = []
# error_dr = []
# for idx in range(1, nsim + 1):
#     print(idx)
#     obs_data = pd.read_csv(f"tmp/obs_{idx}.csv", header=None).values
#     obsX, obsS2, obsS1, obsS3, obsY, obsA = (
#         obs_data[:, :dimX],
#         obs_data[:, dimX: dimX + 1],
#         obs_data[:, dimX + 1:dimX + dimS + 1],
#         obs_data[:, dimX + dimS + 1:dimX + 2 * dimS + 1],
#         obs_data[:, dimX + 2 * dimS + 1],
#         obs_data[:, dimX + 2 * dimS + 1 + 1],
#     )
#     exp_data = pd.read_csv(f"tmp/exp_{idx}.csv", header=None).values
#     expX, expS2, expS1, expS3, expY, expA = (
#         exp_data[:, :dimX],
#         exp_data[:, dimX: dimX + 1],
#         exp_data[:, dimX + 1:dimX + dimS + 1],
#         exp_data[:, dimX + dimS + 1:dimX + 2 * dimS + 1],
#         exp_data[:, dimX + 2 * dimS + 1],
#         exp_data[:, dimX + 2 * dimS + 1 + 1],
#     )
#
#     h1 = outcome_ridge_bridge(obsY[obsA == 1], obsS2[obsA == 1], obsS1[obsA == 1], obsX[obsA == 1], obsS3[obsA == 1], lambda_)
#     exph1 = np.apply_along_axis(h1, 1, np.hstack((expS3[expA == 1], expS2[expA == 1], expX[expA == 1])))
#     h0 = outcome_ridge_bridge(obsY[obsA == 0], obsS2[obsA == 0], obsS1[obsA == 0], obsX[obsA == 0], obsS3[obsA == 0], lambda_)
#     exph0 = np.apply_along_axis(h0, 1, np.hstack((expS3[expA == 0], expS2[expA == 0], expX[expA == 0])))
#     ate_or = np.mean(exph1) - np.mean(exph0)
#
#     q1 = selection_ridge_bridge(obsS2[obsA == 1], obsS1[obsA == 1], obsX[obsA == 1], obsS3[obsA == 1],
#                                 expS2[expA == 1], expS1[expA == 1], expX[expA == 1], expS3[expA == 1], lambda_)
#     obsq1 = np.apply_along_axis(q1, 1, np.hstack((obsS2[obsA == 1], obsS1[obsA == 1], obsX[obsA == 1])))
#     q0 = selection_ridge_bridge(obsS2[obsA == 0], obsS1[obsA == 0], obsX[obsA == 0], obsS3[obsA == 0],
#                                 expS2[expA == 0], expS1[expA == 0], expX[expA == 0], expS3[expA == 0], lambda_)
#     obsq0 = np.apply_along_axis(q0, 1, np.hstack((obsS2[obsA == 0], obsS1[obsA == 0], obsX[obsA == 0])))
#
#     obsh1 = np.apply_along_axis(h1, 1, np.hstack((obsS3[obsA == 1], obsS2[obsA == 1], obsX[obsA == 1])))
#     obsh0 = np.apply_along_axis(h0, 1, np.hstack((obsS3[obsA == 0], obsS2[obsA == 0], obsX[obsA == 0])))
#     ate_dr = ate_or + np.mean(obsq1 * (obsY[obsA == 1] - obsh1)) - np.mean(obsq0 * (obsY[obsA == 0] - obsh0))
#
#     sigma = np.mean((exph1 - ate_dr) ** 2) / np.sum(expA) + np.mean(
#         obsq1 ** 2 * (obsY[obsA == 1] - obsh1) ** 2) / np.sum(obsA)
#     sd = np.sqrt(sigma)
#
#     # np.savetxt(f"tmp/result_ridge_{idx}.csv", [ate_dr, ate_or, sd], delimiter=",")
#
#     # ground truth
#     # tautrue <- 0.3 * (mean(expY[expA == 1]) - mean(expY[expA == 0])) + 0.7 * truetau
#     tautrue = np.mean(expY[expA == 1]) - np.mean(expY[expA == 0])
#     error_or.append(np.abs(ate_or-tautrue))
#     error_dr.append(np.abs(ate_dr-tautrue))
#     print(tautrue, ate_or, ate_dr)
#
# print(np.mean(error_or), np.std(error_or))
# print(np.mean(error_dr), np.std(error_dr))