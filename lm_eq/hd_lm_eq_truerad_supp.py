import numpy as np
import pandas as pd
from numpy.linalg import norm
from numpy.random import seed
from scipy.linalg import toeplitz
from sklearn.linear_model import Lasso, LassoCV


def ndws(X):
    ndcv = LassoCV(cv=10, fit_intercept=False)
    lams = np.zeros(10)
    for i in np.arange(len(lams)):
        print(i)
        ndcv.fit(np.delete(X, i, axis=1), X[:, i])
        lam = ndcv.alpha_
        lams[i] = lam
    lam = np.mean(lams)

    nd = Lasso(alpha=lam, fit_intercept=False)
    inv_cov = np.full([d, d], np.nan)
    for i in np.arange(d):
        print(i)
        nd.fit(np.delete(X, i, axis=1), X[:, i])
        gam = nd.coef_
        tau2 = np.mean((X[:, i] - np.delete(X, i, axis=1).dot(gam)) ** 2) + lam * np.sum(np.abs(gam))
        inv_cov[i] = np.insert(-gam, i, 1) / tau2
    return inv_cov


B = 500
N = 2 ** 14
ds = 2 ** np.array([10])
ss = 2 ** np.array([2, 4])

ts = np.full([len(ds), len(ss), 4, 3], np.nan)

# cov_mat = [toeplitz(0.9 ** np.arange(ds[i])) for i in np.arange(len(ds))]
cov_mat = np.full((d, d), 0.8); np.fill_diagonal(cov_mat, 1)

lassocv = LassoCV(cv=10, fit_intercept=False)

l = 1
seed(l)

for i in np.arange(len(ds)):
    d = ds[i]
    X = np.random.multivariate_normal(np.zeros(d), cov_mat[i], N)
    inv_cov = ndws(X)
    for ii in np.arange(len(ss)):
        s = ss[ii]
        beta_s = np.concatenate((np.ones(s), np.zeros(d - s)))
        mu = X.dot(beta_s)
        y = np.random.normal(mu, 1)
        try:
            lassocv.fit(X, y)
            beta_h = lassocv.coef_
            g = -X.T.dot(y - X.dot(beta_h)) / N
            beta_db = beta_h - inv_cov.dot(g)
            beta_d = beta_db - beta_s
            ts[i, ii, :, 0] = np.array([norm(beta_d, np.inf), norm(beta_d, 2), norm(beta_d, 1), np.abs(beta_d[1])])
            ts[i, ii, :, 1] = np.array(
                [norm(beta_d[:s], np.inf), norm(beta_d[:s], 2), norm(beta_d[:s], 1), np.abs(beta_d[:s][1])])
            ts[i, ii, :, 2] = np.array(
                [norm(beta_d[s:], np.inf), norm(beta_d[s:], 2), norm(beta_d[s:], 1), np.abs(beta_d[s:][1])])
        except:
            continue

pd.DataFrame(np.hstack((np.ones((3 * len(ss), 1)) * l, ts[0].transpose((2, 0, 1)).reshape(3 * len(ss), 4)))).to_csv(
    str(l) + '/hd_lm_eq_truerad_supp.csv', header=False, index=False)
