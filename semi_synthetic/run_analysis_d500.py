import pyspark
from pyspark.sql.types import *
from pyspark.sql import functions
import numpy as np
import pandas as pd
from numpy.linalg import solve, norm
from scipy.special import expit
from scipy.optimize import minimize
from scipy.linalg import toeplitz
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.model_selection import KFold


def select_dummy_factors(pdf, dummy_columns, keep_top):
    nobs = pdf.shape[0]
    factor_selected = {}
    for i in range(len(dummy_columns)):
        factor_counts = (pdf[dummy_columns[i]]).value_counts()
        factor_cum = factor_counts.cumsum() / nobs
        factor_selected[dummy_columns[i]] = sorted(list(factor_counts.index[factor_cum <= keep_top[i]]))
    dummy_info = {'factor_selected': factor_selected}
    return dummy_info


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def grad(y, x, theta):
    return -x.T.dot(y - expit(x.dot(theta))) / x.shape[0]


def grad_n(y, x, theta):
    return -x * (np.ravel(y) - expit(x.dot(theta)))[:, None]


def ndws(X):
    d = X.shape[1]
    ndcv = LassoCV(cv=10, fit_intercept=False, n_jobs=-1, random_state=93)
    inv_cov = np.full([d, d], np.nan)
    for i in np.arange(d):
        print(i)
        ndcv.fit(np.delete(X, i, axis=1), X[:, i])
        gam = ndcv.coef_
        lam = ndcv.alpha_
        tau2 = np.mean((X[:, i] - np.delete(X, i, axis=1).dot(gam)) ** 2) + lam * np.sum(np.abs(gam))
        inv_cov[i] = np.insert(-gam, i, 1) / tau2
    return inv_cov


def csl(y, x, beta, k, lam):
    d = x.shape[1]
    n0 = int(x.shape[0] / k)
    g_N = grad_n(y, x, beta)
    g_n = g_N[:n0]
    g_k1 = np.mean(g_N[n0:].reshape((k - 1, n0, -1)), 1)
    g_1 = np.mean(g_n, 0)
    g_k = np.vstack((g_1, g_k1))
    g_avg = np.mean(g_k, 0)
    g_glob = g_avg
    g_loc = g_1
    lams = lam * 2 ** np.arange(-10, 5, dtype=np.float)
    kf = KFold(n_splits=10)
    losses = np.zeros((len(lams), kf.get_n_splits()))
    loc_spl = list(kf.split(x[:n0]))
    glob_spl = list(kf.split(g_k1))
    for j in np.arange(kf.get_n_splits()):
        x_train, x_test = x[:n0][loc_spl[j][0]], x[:n0][loc_spl[j][1]]
        y_train, y_test = y[:n0][loc_spl[j][0]], y[:n0][loc_spl[j][1]]
        g_loc_train, g_loc_test = np.mean(g_n[loc_spl[j][0]], 0), np.mean(g_n[loc_spl[j][1]], 0)
        g_glob_train, g_glob_test = np.mean(np.vstack((g_loc_train, g_k1[glob_spl[j][0]])), 0), np.mean(np.vstack((g_loc_test, g_k1[glob_spl[j][1]])), 0)
        for i in np.arange(len(lams)):
            print([j, i])
            def obj(w):
                return np.mean(-x_train.dot(w) * y_train + softplus(x_train.dot(w))) - w.T.dot(g_loc_train - g_glob_train) + lams[i] * norm(w, 1)
            beta_j = minimize(obj, beta, method='BFGS', options={'maxiter': 100}).x
            losses[i, j] = np.mean(-x_test.dot(beta_j) * y_test + softplus(x_test.dot(beta_j))) - beta_j.T.dot(g_loc_test - g_glob_test)
    lam_new = lams[np.argmin(np.mean(losses, 1))]
    def obj(w):
        return np.mean(-x[:n0].dot(w) * y[:n0] + softplus(x[:n0].dot(w))) - w.T.dot(g_loc - g_glob) + lam_new * norm(w, 1)
    return minimize(obj, beta, method='BFGS').x, lam_new


def deb_lasso(y, x, beta, inv_cov):
    g_avg = grad(y, x, beta)
    return beta - inv_cov.dot(g_avg)


def nk1grad(y, x, beta, k, inv_cov):
    n0 = int(x.shape[0] / k)
    g_N = grad_n(y, x, beta)
    g_avg = np.mean(g_N, 0)
    g_n = g_N[:n0] - g_avg
    g_k1 = np.sqrt(n0) * (np.mean(g_N[n0:].reshape((k - 1, n0, -1)), 1) - g_avg)
    g_nk1 = np.vstack((g_n, g_k1))
    B = 500
    np.random.seed(93)
    eps_nk1 = np.random.normal(0, 1, (n0 + k - 1, B))
    G_nk1 = g_nk1.T.dot(eps_nk1) / np.sqrt(n0 * k * (n0 + k - 1))
    bt = np.abs(inv_cov.dot(G_nk1))
    return np.percentile(np.max(bt, 0), 95)


def kgrad(y, x, beta, k, inv_cov):
    n0 = int(x.shape[0] / k)
    g_N = grad_n(y, x, beta)
    g_avg = np.mean(g_N, 0)
    g_k = np.sqrt(n0) * (np.mean(g_N.reshape((k, n0, -1)), 1) - g_avg)
    B = 500
    np.random.seed(93)
    eps_k = np.random.normal(0, 1, (k, B))
    G_k = g_k.T.dot(eps_k) / np.sqrt(n0 * k * k)
    bt = np.abs(inv_cov.dot(G_k))
    return np.percentile(np.max(bt, 0), 95)


spark = pyspark.sql.SparkSession.builder.appName("Spark Native Logistic Regression App").getOrCreate()

fit_intercept = True

file_path = ['allfile_ordered.csv']  # HDFS file

usecols_x = ['Year', 'Month', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'Origin', 'Dest']

Y_name = "ArrDelay"
sample_size_sub = []
memsize_sub = []

data_sdf_i = spark.read.csv(file_path[0], header=True)
data_sdf_i = data_sdf_i.select(usecols_x + [Y_name])
data_sdf_i = data_sdf_i.filter(' and '.join('(%s != "NA")' % col_name for col_name in data_sdf_i.columns))

data_sdf_i = data_sdf_i.withColumn(Y_name, functions.when(data_sdf_i[Y_name] > 0, 1).otherwise(0))

data_sdf_i = data_sdf_i.sample(False, 0.01, 93)

data_sdf_i = data_sdf_i.withColumn('CRSDepTime', ((data_sdf_i['CRSDepTime'].cast('int') / 100).cast('int') % 24).cast('string'))
data_sdf_i = data_sdf_i.withColumn('CRSArrTime', ((data_sdf_i['CRSArrTime'].cast('int') / 100).cast('int') % 24).cast('string'))


d = 500
n0 = 500
k = 1000
N = n0 * k

sample_df = data_sdf_i.toPandas().sample(frac=1, random_state=93).reset_index(drop=True)[:(N * 2)]

dummy_info = select_dummy_factors(
    sample_df[:n0],
    dummy_columns=['UniqueCarrier', 'Origin', 'Dest'],
    keep_top=[0.9, 0.9, 0.9])
dummy_info['factor_selected']['CRSDepTime'] = sorted(map(str, set(range(24)) - {23, 0, 1, 2, 3, 4, 5}))
dummy_info['factor_selected']['CRSArrTime'] = sorted(map(str, set(range(24)) - {1, 2, 3, 4, 5, 6}))

for col in ['Year', 'Month', 'DayOfWeek']:
    dummy_info['factor_selected'][col] = sorted(sample_df[:n0][col].unique())

for i in dummy_info['factor_selected'].keys():
    if i in ['UniqueCarrier', 'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime']:
        sample_df.loc[~sample_df[i].isin(dummy_info['factor_selected'][i]), i] = '00_OTHERS'

convert_dummies = list(dummy_info['factor_selected'].keys())

for col in convert_dummies:
    if col in ['UniqueCarrier', 'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime']:
        sample_df[col] = pd.Categorical(sample_df[col], categories=['00_OTHERS'] + dummy_info['factor_selected'][col])
    else:
        sample_df[col] = pd.Categorical(sample_df[col], categories=dummy_info['factor_selected'][col])

X_with_dummies = pd.get_dummies(data=sample_df, drop_first=fit_intercept, columns=convert_dummies, sparse=True)
X_with_dummies['Distance'] = (X_with_dummies['Distance'] - X_with_dummies['Distance'].mean()) / X_with_dummies['Distance'].std()

x_f = X_with_dummies.drop([Y_name], axis=1)

x_f.sort_index(axis=1, inplace=True)
y_f = np.array(sample_df[Y_name]).astype('float')
x_f = sm.add_constant(x_f)
x_columns = x_f.columns
x_f = np.array(x_f).astype('float')

model = sm.Logit(y_f[N:], x_f[N:])
mdl = model.fit()

s = 5
pp = pd.DataFrame({'pvalues': mdl.pvalues, 'params': mdl.params}, index=x_columns)

if 'const' in pp.sort_values('pvalues')[:s].index:
    sig_ind = sorted(np.argpartition(mdl.pvalues, s)[:s])
else:
    sig_ind = sorted(np.argpartition(mdl.pvalues, s - 1)[:(s - 1)]) + [0]

x_f_new = x_f[:N, sig_ind]
x_columns_new = x_columns[sig_ind]

y_f_new = y_f[:N]

cov_mat = toeplitz(0.5 ** np.arange(d - s))
np.random.seed(93)
x_f_new = np.hstack((x_f_new, np.random.multivariate_normal(np.zeros(d - s), cov_mat, N)))
x_f_new[:, int(-(d - s) / 2):] = (x_f_new[:, int(-(d - s) / 2):] >= 0).astype(float)

x_m_new = x_f_new[:n0]
y_m = y_f_new[:n0]
logregcv = LogisticRegressionCV(cv=10, scoring='neg_log_loss', penalty='l1', solver='saga', fit_intercept=False, max_iter=1000, n_jobs=-1, random_state=93)
logregcv.fit(x_m_new, y_m)
beta_0 = logregcv.coef_[0]
lam_0 = 1 / logregcv.C_ / n0

psi = x_m_new.dot(beta_0)
x_proj = x_m_new * np.sqrt(expit(psi) / (1 + np.exp(psi)))[:, None]
inv_cov = ndws(x_proj)

beta_t1 = deb_lasso(y_f_new, x_f_new, beta_0, inv_cov)
cd_1 = nk1grad(y_f_new, x_f_new, beta_0, k, inv_cov)
beta_1, lam_1 = csl(y_f_new, x_f_new, beta_0, k, lam_0)
beta_t2 = deb_lasso(y_f_new, x_f_new, beta_1, inv_cov)
cd_2 = nk1grad(y_f_new, x_f_new, beta_1, k, inv_cov)
beta_2, lam_2 = csl(y_f_new, x_f_new, beta_1, k, lam_1)
beta_t3 = deb_lasso(y_f_new, x_f_new, beta_2, inv_cov)
cd_3 = nk1grad(y_f_new, x_f_new, beta_2, k, inv_cov)
beta_3, lam_3 = csl(y_f_new, x_f_new, beta_2, k, lam_2)
beta_t4 = deb_lasso(y_f_new, x_f_new, beta_3, inv_cov)
cd_4 = nk1grad(y_f_new, x_f_new, beta_3, k, inv_cov)
beta_4, lam_4 = csl(y_f_new, x_f_new, beta_3, k, lam_3)
beta_t5 = deb_lasso(y_f_new, x_f_new, beta_4, inv_cov)
cd_5 = nk1grad(y_f_new, x_f_new, beta_4, k, inv_cov)
beta_5, lam_5 = csl(y_f_new, x_f_new, beta_4, k, lam_4)
beta_t6 = deb_lasso(y_f_new, x_f_new, beta_5, inv_cov)
cd_6 = nk1grad(y_f_new, x_f_new, beta_5, k, inv_cov)

pd.DataFrame(np.transpose(np.vstack([beta_t1, beta_t2, beta_t3, beta_t4, beta_t5, beta_t6]))).to_csv('split/beta_t_d500.csv', header=False, index=False)

pd.DataFrame(np.transpose(np.vstack([cd_1, cd_2, cd_3, cd_4, cd_5, cd_6]))).to_csv('split/cd_d500.csv', header=False, index=False)

x_columns_new = x_columns_new.tolist() + ['N' + str(i) for i in np.arange(d - int((d - s) / 2) - len(x_columns)) + 1] + ['B' + str(i) for i in np.arange(int((d - s) / 2)) + 1]
pd.DataFrame(x_columns_new).to_csv('split/columns_d500.csv', header=False, index=False)