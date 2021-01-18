import numpy as np
import pandas as pd
from scipy.special import expit
from numpy.linalg import norm
from numpy.random import seed
from scipy.linalg import toeplitz
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

logregcv = LogisticRegressionCV(cv=5, scoring='neg_log_loss', penalty='l1', solver='saga', fit_intercept=False)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def ci_hd_glm_cv(y, X, k, B, beta_s, s, t):
    cov = np.zeros((2, 2, 4, t, 3))
    rad = np.zeros((2, 2, 4, t, 3))

    N, d = X.shape
    n = int(N / k)

    for i in np.arange(t):
        print(i)
        if i == 0:
            logregcv.fit(X[:n, :], y[:n])
            beta_ini = logregcv.coef_[0]
            lam = 1 / logregcv.C_ / n
            print(lam)
            print(beta_ini[:10])
        else:
            beta_ini = beta_os

        psi = X.dot(beta_ini)
        q1 = y - expit(psi)
        g = -X.T.dot(q1) / N
        if i == 0:
            inv_cov = ndws(X[:n, :] * np.sqrt(expit(psi[:n]) / (1 + np.exp(psi[:n])))[:, None])
        beta_db = beta_ini - inv_cov.dot(g)

        eps = np.repeat(np.random.normal(0, 1, (k, B)), n, 0)
        G = (-X.T.dot((eps.T * q1).T) - np.outer(g, np.sum(eps, 0))) / N

        bt = np.abs(inv_cov.dot(G))
        beta_d = beta_db - beta_s

        cd = np.percentile(np.vstack((np.max(bt, 0), norm(bt, 2, 0), np.sum(bt, 0), bt[1, :])), (95, 90), 1)
        ts = np.array([norm(beta_d, np.inf), norm(beta_d, 2), norm(beta_d, 1), np.abs(beta_d[1])])
        cd1 = np.percentile(np.vstack((np.max(bt[:s], 0), norm(bt[:s], 2, 0), np.sum(bt[:s], 0), bt[:s][1, :])),
                            (95, 90), 1)
        ts1 = np.array([norm(beta_d[:s], np.inf), norm(beta_d[:s], 2), norm(beta_d[:s], 1), np.abs(beta_d[:s][1])])
        cd0 = np.percentile(np.vstack((np.max(bt[s:], 0), norm(bt[s:], 2, 0), np.sum(bt[s:], 0), bt[s:][1, :])),
                            (95, 90), 1)
        ts0 = np.array([norm(beta_d[s:], np.inf), norm(beta_d[s:], 2), norm(beta_d[s:], 1), np.abs(beta_d[s:][1])])

        cov[0, :, :, i, 0] = ts < cd
        cov[0, :, :, i, 1] = ts1 < cd1
        cov[0, :, :, i, 2] = ts0 < cd0
        rad[0, :, :, i, 0] = cd
        rad[0, :, :, i, 1] = cd1
        rad[0, :, :, i, 2] = cd0

        eps = np.vstack(
            (np.random.normal(0, 1, (n, B)), np.repeat(np.random.normal(0, 1, (k - 1, B)) / np.sqrt(n), n, 0)))
        G = (-X.T.dot((eps.T * q1).T) - np.outer(g, np.sum(eps, 0))) / np.sqrt(N * (n + k - 1))

        bt = np.abs(inv_cov.dot(G))
        beta_d = beta_db - beta_s

        cd = np.percentile(np.vstack((np.max(bt, 0), norm(bt, 2, 0), np.sum(bt, 0), bt[1, :])), (95, 90), 1)
        ts = np.array([norm(beta_d, np.inf), norm(beta_d, 2), norm(beta_d, 1), np.abs(beta_d[1])])
        cd1 = np.percentile(np.vstack((np.max(bt[:s], 0), norm(bt[:s], 2, 0), np.sum(bt[:s], 0), bt[:s][1, :])),
                            (95, 90), 1)
        ts1 = np.array([norm(beta_d[:s], np.inf), norm(beta_d[:s], 2), norm(beta_d[:s], 1), np.abs(beta_d[:s][1])])
        cd0 = np.percentile(np.vstack((np.max(bt[s:], 0), norm(bt[s:], 2, 0), np.sum(bt[s:], 0), bt[s:][1, :])),
                            (95, 90), 1)
        ts0 = np.array([norm(beta_d[s:], np.inf), norm(beta_d[s:], 2), norm(beta_d[s:], 1), np.abs(beta_d[s:][1])])

        cov[1, :, :, i, 0] = ts < cd
        cov[1, :, :, i, 1] = ts1 < cd1
        cov[1, :, :, i, 2] = ts0 < cd0
        rad[1, :, :, i, 0] = cd
        rad[1, :, :, i, 1] = cd1
        rad[1, :, :, i, 2] = cd0

        if i < t - 1:
            def grad_n(y, x, theta):
                return -x * (np.ravel(y) - expit(x.dot(theta)))[:, None]

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
                        return np.mean(-x_train.dot(w) * y_train + softplus(x_train.dot(w))) - w.T.dot(
                            g_loc_train - g_glob_train) + lams[ii] * norm(w, 1), \
                               -x_train.T.dot(y_train - expit(x_train.dot(w))) / x_train.shape[0] - (
                                           g_loc_train - g_glob_train) + lams[ii] * np.sign(w)

                    beta_j = fmin_l_bfgs_b(obj, np.zeros(d))[0]
                    losses[ii, j] = np.mean(-x_test.dot(beta_j) * y_test + softplus(x_test.dot(beta_j))) - beta_j.T.dot(
                        g_loc_test - g_glob_test)
            lam = lams[np.argmin(np.mean(losses, 1))]
            g_glob = -X.T.dot(y - expit(X.dot(beta_ini))) / N
            g_loc = -X[:n].T.dot(y[:n] - expit(X[:n].dot(beta_ini))) / n

            def obj(w):
                return np.mean(-X[:n].dot(w) * y[:n] + softplus(X[:n].dot(w))) - w.T.dot(g_loc - g_glob) + lam * norm(w,
                                                                                                                      1), \
                       -X[:n].T.dot(y[:n] - expit(X[:n].dot(w))) / n - (g_loc - g_glob) + lam * np.sign(w)

            beta_os = fmin_l_bfgs_b(obj, np.zeros(d))[0]
            print(lam)
            print(beta_os[:10])

    return cov, rad


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
rep = 1000
N = 2 ** 14
ks = 2 ** np.arange(2, 7)
ds = 2 ** np.array([10])
ss = 2 ** np.array([1, 3])
t = 4

cov = np.full([2, 2, 4, t, len(ds), len(ss), len(ks), 3], np.nan)
rad = np.full([2, 2, 4, t, len(ds), len(ss), len(ks), 3], np.nan)

l = 1
seed(l)

for i in np.arange(len(ds)):
    d = ds[i]
    cov_mat = toeplitz(0.9 ** np.arange(d))
    # cov_mat = np.full((d, d), 0.8); np.fill_diagonal(cov_mat, 1)
    X = np.random.multivariate_normal(np.zeros(d), cov_mat, N)

    for ii in np.arange(len(ss)):
        s = ss[ii]
        beta_s = np.concatenate((np.ones(s), np.zeros(d - s)))
        p = expit(X.dot(beta_s))
        y = np.random.binomial(1, p)
        for j in np.arange(len(ks)):
            print([l, i, ii, j])
            with open('log_' + str(l) + '.txt', "ab") as f:
                f.write(b"\n")
                np.savetxt(f, np.array([l, i, ii, j]))
            k = ks[j]
            try:
                cov[:, :, :, :, i, ii, j, :], rad[:, :, :, :, i, ii, j, :] = ci_hd_glm_cv(y, X, k, B, beta_s, s, t)
            except:
                continue

for i1 in np.arange(2):
    for i2 in np.arange(2):
        for i3 in np.arange(4):
            for i4 in np.arange(4):
                if i1 == 0:
                    s1 = 'k'
                else:
                    s1 = 'nk1'
                if i2 == 0:
                    s2 = 'p95'
                else:
                    s2 = 'p9'
                if i3 == 0:
                    s3 = 'inf'
                elif i3 == 1:
                    s3 = '2'
                elif i3 == 2:
                    s3 = '1'
                else:
                    s3 = 'pt'
                s4 = 't' + str(i4 + 1)
                pd.DataFrame(np.hstack((np.ones((3 * len(ds) * len(ss), 1)) * l,
                                        cov[i1, i2, i3, i4, :, :, :, :].transpose((3, 0, 1, 2)).reshape(
                                            3 * len(ds) * len(ss), len(ks))))).to_csv(
                    str(l) + '/hd_glm_tp_' + s1 + '_' + s2 + '_cov_' + s3 + '_' + s4 + '.csv', header=False,
                    index=False)
                pd.DataFrame(np.hstack((np.ones((3 * len(ds) * len(ss), 1)) * l,
                                        rad[i1, i2, i3, i4, :, :, :, :].transpose((3, 0, 1, 2)).reshape(
                                            3 * len(ds) * len(ss), len(ks))))).to_csv(
                    str(l) + '/hd_glm_tp_' + s1 + '_' + s2 + '_rad_' + s3 + '_' + s4 + '.csv', header=False,
                    index=False)
