"""
Simulation studies of k-grad and n+k-1-grad distributed bootstrap algorithms for simultaneous CIs in high-dimensional linear regression model.
"""

import numpy as np
from numpy.linalg import norm
from numpy.random import seed
from scipy.linalg import toeplitz
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

lassocv = LassoCV(cv=10, fit_intercept=False)


# k-grad and n+k-1-grad algorithms.
def ci(y, X, inv_cov, k, B, beta_s, t):
    cov = np.zeros((2, t))
    rad = np.zeros((2, t))

    N, d = X.shape
    n = int(N / k)

    for i in np.arange(t):
        print(i)
        if i == 0:
            lassocv.fit(X[:n, :], y[:n])
            beta_ini = lassocv.coef_
            lam = lassocv.alpha_
            print(lam)
            print(beta_ini[:10])
        else:
            beta_ini = beta_os

        psi = X.dot(beta_ini)
        q1 = y - psi
        g = -X.T.dot(q1) / N
        beta_db = beta_ini - inv_cov.dot(g)

        eps = np.repeat(np.random.normal(0, 1, (k, B)), n, 0)
        G = (-X.T.dot((eps.T * q1).T) - np.outer(g, np.sum(eps, 0))) / N

        bt = np.abs(inv_cov.dot(G))
        beta_d = beta_db - beta_s

        cd = np.percentile(np.max(bt, 0), 95)
        ts = norm(beta_d, np.inf)

        cov[0, i] = ts < cd
        rad[0, i] = cd

        eps = np.vstack(
            (np.random.normal(0, 1, (n, B)), np.repeat(np.random.normal(0, 1, (k - 1, B)) / np.sqrt(n), n, 0)))
        G = (-X.T.dot((eps.T * q1).T) - np.outer(g, np.sum(eps, 0))) / np.sqrt(N * (n + k - 1))

        bt = np.abs(inv_cov.dot(G))
        beta_d = beta_db - beta_s

        cd = np.percentile(np.max(bt, 0), 95)
        ts = norm(beta_d, np.inf)

        cov[1, i] = ts < cd
        rad[1, i] = cd

        if i < t - 1:
            def grad_n(y, x, theta):
                return -x * (np.ravel(y) - x.dot(theta))[:, None]

            g_N = grad_n(y, X, beta_ini)
            g_n = g_N[:n]
            g_k1 = np.mean(g_N[n:].reshape((k - 1, n, -1)), 1)
            lams = lam * 2 ** np.arange(-5, 6, dtype=np.float)
            kf = KFold(n_splits=min(n, k - 1, 5))
            losses = np.zeros((len(lams), kf.get_n_splits()))
            loc_spl = list(kf.split(X[:n]))
            glob_spl = list(kf.split(g_k1))
            for j in np.arange(kf.get_n_splits()):
                x_train, x_test = X[:n][loc_spl[j][0]], X[:n][loc_spl[j][1]]
                y_train, y_test = y[:n][loc_spl[j][0]], y[:n][loc_spl[j][1]]
                g_loc_train, g_loc_test = np.mean(g_n[loc_spl[j][0]], 0), np.mean(g_n[loc_spl[j][1]], 0)
                g_glob_train, g_glob_test = np.mean(np.vstack((g_loc_train, g_k1[glob_spl[j][0]])), 0), np.mean(
                    np.vstack((g_loc_test, g_k1[glob_spl[j][1]])), 0)
                for ii in np.arange(len(lams)):
                    print([j, ii])

                    def obj(w):
                        return np.mean((y_train - x_train.dot(w)) ** 2) / 2 - w.T.dot(g_loc_train - g_glob_train) + \
                               lams[ii] * norm(w, 1), \
                               -x_train.T.dot(y_train - x_train.dot(w)) / x_train.shape[0] - (
                                       g_loc_train - g_glob_train) + lams[ii] * np.sign(w)

                    beta_j = fmin_l_bfgs_b(obj, np.zeros(d))[0]
                    losses[ii, j] = np.mean((y_test - x_test.dot(beta_j)) ** 2) / 2 - beta_j.T.dot(
                        g_loc_test - g_glob_test)
            lam = lams[np.argmin(np.mean(losses, 1))]
            g_glob = -X.T.dot(y - X.dot(beta_ini)) / N
            g_loc = -X[:n, :].T.dot(y[:n] - X[:n, :].dot(beta_ini)) / n

            def obj(w):
                return np.mean((y[:n] - X[:n, :].dot(w)) ** 2) / 2 - w.T.dot(g_loc - g_glob) + lam * norm(w, 1), \
                       -X[:n, :].T.dot(y[:n] - X[:n, :].dot(w)) / n - (g_loc - g_glob) + lam * np.sign(w)

            beta_os = fmin_l_bfgs_b(obj, np.zeros(d))[0]
            print(lam)
            print(beta_os[:10])

    return cov, rad


# Nodewise Lasso algorithm.
def ndws(X):
    ndcv = LassoCV(cv=10, fit_intercept=False)
    lams = np.zeros(10)
    for i in np.arange(len(lams)):
        ndcv.fit(np.delete(X, i, axis=1), X[:, i])
        lam = ndcv.alpha_
        lams[i] = lam
    lam = np.mean(lams)

    nd = Lasso(alpha=lam, fit_intercept=False)
    inv_cov = np.full([d, d], np.nan)
    for i in np.arange(d):
        nd.fit(np.delete(X, i, axis=1), X[:, i])
        gam = nd.coef_
        tau2 = np.mean((X[:, i] - np.delete(X, i, axis=1).dot(gam)) ** 2) + lam * np.sum(np.abs(gam))
        inv_cov[i] = np.insert(-gam, i, 1) / tau2
    return inv_cov


# Number of bootstrap samples.
B = 500
# Total sample size.
N = 2 ** 14
# Dimension.
d = 2 ** 10
# Numbers of machines.
ks = 2 ** np.arange(2, 7)
# Numbers of non-zero true coefficients.
ss = 2 ** np.array([2, 4])
t = 4

cov = np.full([2, t, len(ss), len(ks)], np.nan)
rad = np.full([2, t, len(ss), len(ks)], np.nan)

# Set random seed for each replication.
l = 1
seed(l)

# Toeplitz design.
cov_mat = toeplitz(0.9 ** np.arange(d))
# For equi-corr design, use the following line instead.
# cov_mat = np.full((d, d), 0.8); np.fill_diagonal(cov_mat, 1)
X = np.random.multivariate_normal(np.zeros(d), cov_mat, N)

inv_cov = [ndws(X[:int(N / k), :]) for k in ks]

# Run k-grad/n+k-1-grad to compute CIs.
for i in np.arange(len(ss)):
    s = ss[i]
    beta_s = np.concatenate((np.ones(s), np.zeros(d - s)))
    mu = X.dot(beta_s)
    y = np.random.normal(mu, 1)
    for j in np.arange(len(ks)):
        print([l, i, j])
        k = ks[j]
        try:
            cov[:, :, i, j], rad[:, :, i, j] = ci(y, X, inv_cov[j], k, B, beta_s, t)
        except:
            continue

# Print CI results.
for i1 in np.arange(2):
    for i2 in np.arange(len(ss)):
        for i3 in np.arange(len(ks)):
            print('Coverage of {method} CI at 4 iterations for k=2^{k_exp} and s0=2^{s_exp}:'.format(
                method='k-grad' if i1 == 0 else 'n+k-1-grad',
                k_exp=int(np.log2(ks[i3])),
                s_exp=int(np.log2(ss[i2])),
            ))
            print(cov[i1, :, i2, i3])
            print('Width of {method} CI at 4 iterations for k=2^{k_exp} and s0=2^{s_exp}:'.format(
                method='k-grad' if i1 == 0 else 'n+k-1-grad',
                k_exp=int(np.log2(ks[i3])),
                s_exp=int(np.log2(ss[i2])),
            ))
            print(rad[i1, :, i2, i3] * 2)
