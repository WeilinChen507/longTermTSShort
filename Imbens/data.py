import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# Set up dimensions
nsim = 200

dimX = 10
dimS = dimU = 5
q = 2

nobs = 2000
nexp = 2000

np.random.seed(1)


def g(lindat):
    return np.sign(lindat) * np.abs(lindat) ** q


def rnd_norm_coef(dim1, dim2, scale, min_val=0, max_val=1):
    coef = np.zeros((dim1, dim2))
    for j in range(dim2):
        vec = np.random.uniform(min_val, max_val, dim1)
        vec = scale * vec / np.linalg.norm(vec)
        coef[:, j] = vec
    return coef


def rnd_data(n, p):
    if p != 1:
        return multivariate_normal.rvs(mean=np.zeros(p), cov=np.identity(p), size=n)
    else:
        return np.expand_dims(multivariate_normal.rvs(mean=np.zeros(p), cov=np.identity(p), size=n), axis=1)


res_list = []

for i in range(1, nsim + 1):
    print(i)

    # Set up system parameters
    kappaU = rnd_norm_coef(dimU, 1, np.sqrt(0.5))
    kappaX = rnd_norm_coef(dimX, 1, np.sqrt(0.5))
    tau1 = rnd_norm_coef(1, dimS, np.sqrt(0.5))
    beta1 = rnd_norm_coef(dimX, dimS, np.sqrt(0.5))
    gamma1 = rnd_norm_coef(dimU, dimS, np.sqrt(0.5))
    tau2 = np.sqrt(0.5)
    alpha2 = rnd_norm_coef(dimS, 1, np.sqrt(0.5))
    beta2 = rnd_norm_coef(dimX, 1, np.sqrt(0.5))
    gamma2 = rnd_norm_coef(dimU, 1, np.sqrt(0.5))
    tau3 = rnd_norm_coef(1, dimS, np.sqrt(0.5))
    alpha3 = rnd_norm_coef(1, dimS, np.sqrt(0.5))
    beta3 = rnd_norm_coef(dimX, dimS, np.sqrt(0.5))
    gamma3 = rnd_norm_coef(dimU, dimS, np.sqrt(0.5))
    tauy = np.sqrt(0.5)
    alphay = rnd_norm_coef(dimS, 1, np.sqrt(0.5))
    betay = rnd_norm_coef(dimX, 1, np.sqrt(0.5))
    gammay = rnd_norm_coef(dimU, 1, np.sqrt(0.5))

    # Set up observational data
    X = rnd_data(nobs, dimX)
    U = rnd_data(nobs, dimU)
    prob = 1 / (1 + np.exp(X @ kappaX + U @ kappaU))
    A = (np.random.uniform(size=(nobs, 1)) <= prob).astype(int)
    S1 = A @ tau1 + X @ beta1 + U @ gamma1 + np.sqrt(0.5) * rnd_data(nobs, dimS)
    S2 = A * tau2 + S1 @ alpha2 + X @ beta2 + U @ gamma2 + np.sqrt(0.5) * rnd_data(nobs, 1)
    S3 = A @ tau3 + S2 @ alpha3 + X @ beta3 + U @ gamma3 + np.sqrt(0.5) * rnd_data(nobs, dimS)
    Y = A * tauy + S3 @ alphay + X @ betay + U @ gammay + np.sqrt(0.5) * rnd_data(nobs, 1)

    pd.DataFrame(np.hstack([g(X), g(S2), g(S1), g(S3), Y, A])).to_csv(f"tmp/obs_{i}.csv", index=False, header=False)

    # Set up experimental data
    X = rnd_data(nexp, dimX)
    U = rnd_data(nexp, dimU)
    A = (np.random.uniform(size=(nexp, 1)) <= 0.5).astype(int)
    S1 = A @ tau1 + X @ beta1 + U @ gamma1 + np.sqrt(0.5) * rnd_data(nexp, dimS)
    S2 = A * tau2 + S1 @ alpha2 + X @ beta2 + U @ gamma2 + np.sqrt(0.5) * rnd_data(nexp, 1)
    S3 = A @ tau3 + S2 @ alpha3 + X @ beta3 + U @ gamma3 + np.sqrt(0.5) * rnd_data(nexp, dimS)
    Y = A * tauy + S3 @ alphay + X @ betay + U @ gammay + np.sqrt(0.5) * rnd_data(nexp, 1)

    pd.DataFrame(np.hstack([g(X), g(S2), g(S1), g(S3), Y, A])).to_csv(f"tmp/exp_{i}.csv", index=False, header=False)

    S1 = tau1
    S2 = tau2 + S1 @ alpha2
    S3 = tau3 + S2 @ alpha3
    Y = tauy + S3 @ alphay

    res_list.append(Y)

np.save("tmp/result.npy", res_list)
