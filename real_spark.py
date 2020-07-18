import pyspark
from pyspark.sql.types import *
from pyspark.sql import functions
from pyspark.sql.functions import pandas_udf, PandasUDFType,  monotonically_increasing_id
import numpy as np
import pandas as pd
from math import ceil
from numpy.random import seed
from numpy.linalg import solve, norm
from scipy.linalg import toeplitz
from scipy.special import expit
import statsmodels.api as sm
import os, sys, time
from datetime import timedelta
import warnings
import pickle


def convert_schema(usecols_x, dummy_info, fit_intercept):
    '''Convert schema type for large data frame
    '''

    schema_fields = []
    if len(dummy_info) == 0: # No dummy is used
        for j in usecols_x:
            schema_fields.append(StructField(j, DoubleType(), True))

    else:
        # Use dummy
        convert_dummies = list(dummy_info['factor_selected'].keys())

        for x in list(set(usecols_x) - set(convert_dummies)):
            schema_fields.append(StructField(x, DoubleType(), True))

        for i in convert_dummies:
            for j in dummy_info["factor_selected_names"][i][fit_intercept:]:
                schema_fields.append(StructField(j, DoubleType(), True))


    if fit_intercept:
        schema_fields.insert(0, StructField('intercept', DoubleType(), True))

    return schema_fields


def grad(y, x, theta):
    return -x.T.dot(y - expit(x.dot(theta))) / x.shape[0]


def grad_n(y, x, theta):
    return -x * (np.ravel(y) - expit(x.dot(theta)))[:, None]


def simulate_linear_master(n, d, beta_s):
    cov_mat = toeplitz(0.9 ** np.arange(d))
    X = np.random.multivariate_normal(np.zeros(d), cov_mat, n)
    mu = X.dot(beta_s)
    y = np.random.normal(mu, 1).reshape(n, 1)
    return X, y



def logistic_model(sample_df, beta_b, Y_name, fit_intercept=False, dummy_info=[]):
    '''Run linear model on the partitioned data set
    '''

    # x_train = sample_df.drop(['label', 'row_id', 'partition_id'], axis=1)
    # sample_df = samle_df.dropna()

    # Special step to create a local dummy matrix
    if len(dummy_info) > 0:
        convert_dummies = list(dummy_info['factor_selected'].keys())

        X_with_dummies = pd.get_dummies(data=sample_df,
                                        drop_first=fit_intercept,
                                        columns=convert_dummies,
                                        sparse=True)

        x_train = X_with_dummies.drop(['partition_id', Y_name], axis = 1)

        # Check if any dummy column is not in the data chunk.
        usecols_x0 = list(set(sample_df.columns.drop(['partition_id', Y_name])) - set(convert_dummies))
        usecols_x = usecols_x0.copy()
        for i in convert_dummies:
            for j in dummy_info["factor_selected_names"][i][fit_intercept:]:
                usecols_x.append(j)
        usecols_x.sort()
        usecols_full = ['par_id', "grad"]
        usecols_full.extend(usecols_x)

        # raise Exception("usecols_full:\t" + str(len(usecols_full)))
        # raise Exception("usecols_x:\t" + str(usecols_x))

        if set(x_train.columns) != set(usecols_x):
            warnings.warn("Dummies:" + str(set(usecols_x) - set(x_train.columns))
                          + "missing in this data chunk " + str(x_train.shape)
                          + "Skip modeling this part of data.")

            # return a zero fake matrix.
            return pd.DataFrame(0,index=np.arange(len(usecols_x)),
                                columns=usecols_full)

    else:
        x_train = sample_df.drop(['partition_id', Y_name], axis=1)
        usecols_x0 = x_train.columns

    # Standardize the data with global mean and variance
    # if len(data_info) > 0:
    #     for i in usecols_x0:
    #         x_train[i]=(x_train[i] - float(data_info[i][1])) / float(data_info[i][2])


    x_train.sort_index(axis=1, inplace=True)

    # raise Exception("x_train shape:" + str(list(x_train.columns)))

    y_train = np.array(sample_df[Y_name])

    x_train = np.array(sm.add_constant(x_train)).astype('float')

    g = grad(y_train, x_train, beta_b)

    p = x_train.shape[1]


    # Assign par_id
    par_id = pd.DataFrame(np.arange(p).reshape(p, 1), columns=['par_id'])
    # par_id = pd.DataFrame(x_train.columns.to_numpy().reshape(p, 1), columns=["par_id"])

    out_pdf = pd.DataFrame(g, columns=pd.Index(["grad"]))
    par_id.reset_index(drop=True, inplace=True)
    out_pdf.reset_index(drop=True, inplace=True)
    out = pd.concat([par_id, out_pdf],1)

    if out.isna().values.any():
        warnings.warn("NAs appear in the final output")

    return out
    # return pd.DataFrame(Sig_inv)


def boots_mapred(model_mapped_sdf):
    '''MapReduce for partitioned data with given model
    '''
    # mapped_pdf = model_mapped_sdf.toPandas()
    ##----------------------------------------------------------------------------------------
    ## MERGE
    ##----------------------------------------------------------------------------------------
    groupped_sdf = model_mapped_sdf.groupby('par_id')
    groupped_sdf_sum = groupped_sdf.sum(*model_mapped_sdf.columns[1:]) #TODO: Error with Python < 3.7 for > 255 arguments. Location 0 is 'par_id'
    groupped_pdf_sum = groupped_sdf_sum.toPandas().sort_values("par_id")

    if groupped_pdf_sum.shape[0] == 0: # bad chunked models

        raise Exception("Zero-length grouped pandas DataFrame obtained, check the input.")
        # out = pd.DataFrame(columns= ["beta_byOLS", "beta_byONESHOT"] + model_mapped_sdf.columns[3:])

    else:

        Sig_invMcoef_sum = groupped_pdf_sum.iloc[:,2]
        Sig_inv_sum = groupped_pdf_sum.iloc[:,3:]

        # beta_byOLS = np.linalg.solve(Sig_inv_sum, Sig_invMcoef_sum)
        beta_byOLS = np.linalg.lstsq(Sig_inv_sum,
                                     Sig_invMcoef_sum,
                                     rcond=None)[0] # least-squares solution

        beta_byONESHOT = groupped_pdf_sum['sum(coef)'] / model_mapped_sdf.rdd.getNumPartitions()
        p = len(Sig_invMcoef_sum)

        out = pd.DataFrame(np.concatenate((beta_byOLS.reshape(p, 1),
                                           np.asarray(beta_byONESHOT).reshape(p, 1),
                                           Sig_inv_sum), 1),
                           columns= ["beta_byOLS", "beta_byONESHOT"] + model_mapped_sdf.columns[3:])

    return out


spark = pyspark.sql.SparkSession.builder.appName("Spark Native Linear Regression App").getOrCreate()

fit_intercept = True

partition_method = "systematic"

fit_intercept = False

n_files = 1  # Sequential loop to avoid Spark OUT_OF_MEM problem
partition_num_sub = 3
sample_size_sub = 2 ** 16 / 4 * 3
sample_size_per_partition = sample_size_sub / partition_num_sub
p = 2 ** 3
Y_name = "label"
dummy_info = []
data_info = []
convert_dummies = []
max_sample_size_per_sdf = sample_size_per_partition


#  Settings for using real data
#-----------------------------------------------------------------------------------------
    # file_path = ['~/running/data_raw/xa' + str(letter) + '.csv.bz2' for letter in string.ascii_lowercase[0:21]] # local file

    # file_path = ['/running/data_raw/xa' + str(letter) + '.csv' for letter in string.ascii_lowercase[0:1]] # HDFS file


file_path = ['/scratch/rice/y/yu577/dls/allfile_ordered_no_head.csv']  # HDFS file

usecols_x = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
             'CRSArrTime', 'UniqueCarrier', 'ActualElapsedTime',  # 'AirTime',
             'Origin', 'Dest', 'Distance']

schema_sdf = StructType([
    StructField('Year', IntegerType(), True),
    StructField('Month', IntegerType(), True),
    StructField('DayofMonth', IntegerType(), True),
    StructField('DayOfWeek', IntegerType(), True),
    StructField('DepTime', DoubleType(), True),
    StructField('CRSDepTime', DoubleType(), True),
    StructField('ArrTime', DoubleType(), True),
    StructField('CRSArrTime', DoubleType(), True),
    StructField('UniqueCarrier', StringType(), True),
    StructField('FlightNum', StringType(), True),
    StructField('TailNum', StringType(), True),
    StructField('ActualElapsedTime', DoubleType(), True),
    StructField('CRSElapsedTime', DoubleType(), True),
    StructField('AirTime', DoubleType(), True),
    StructField('ArrDelay', DoubleType(), True),
    StructField('DepDelay', DoubleType(), True),
    StructField('Origin', StringType(), True),
    StructField('Dest', StringType(), True),
    StructField('Distance', DoubleType(), True),
    StructField('TaxiIn', DoubleType(), True),
    StructField('TaxiOut', DoubleType(), True),
    StructField('Cancelled', IntegerType(), True),
    StructField('CancellationCode', StringType(), True),
    StructField('Diverted', IntegerType(), True),
    StructField('CarrierDelay', DoubleType(), True),
    StructField('WeatherDelay', DoubleType(), True),
    StructField('NASDelay', DoubleType(), True),
    StructField('SecurityDelay', DoubleType(), True),
    StructField('LateAircraftDelay', DoubleType(), True)
])
# s = spark.read.schema("col0 INT, col1 DOUBLE")


data_info = []

dummy_info_path = "/scratch/rice/y/yu577/dls/dummy_info_latest.pkl"
dummy_info = pickle.load(open(os.path.expanduser(dummy_info_path), "rb"))
dummy_info['factor_set']["Year"] = list(range(1987, 2009))
dummy_info['factor_selected']["Year"] = list(range(1987, 2009))
dummy_info['factor_dropped']["Year"] = []
dummy_info['factor_selected_names']["Year"] = ["Year" + '_' + str(x)
                                               for x in dummy_info['factor_selected']["Year"]]
convert_dummies = list(dummy_info['factor_selected'].keys())

n_files = len(file_path)
partition_num_sub = []
max_sample_size_per_sdf = 100000  # No effect with `real_hdfs` data
sample_size_per_partition = 100000

Y_name = "ArrDelay"
sample_size_sub = []
memsize_sub = []



## TRUE beta
seed(0)
beta_s = np.random.uniform(-0.5, 0.5, p)

time_2sdf_sub = []
time_repartition_sub = []

tic_2sdf = time.perf_counter()


n0 = int(sample_size_per_partition)
X_m, y_m = simulate_linear_master(n0, p, beta_s)
linreg = sm.OLS(y_m, X_m)
beta_b = linreg.fit(disp=0).params


loop_counter = 0
for file_no_i in range(n_files):

    isub = 0  # fixed, never changed

    # Read HDFS to Spark DataFrame
    data_sdf_i = spark.read.csv(file_path[file_no_i], header=True)
    # data_sdf_i = spark.read.csv(file_path[file_no_i], header=True, schema=schema_sdf)
    data_sdf_i = data_sdf_i.select(usecols_x + [Y_name])
    # data_sdf_i = data_sdf_i.dropna()
    data_sdf_i = data_sdf_i.filter(' and '.join('(%s != "NA")' % col_name for col_name in data_sdf_i.columns))

    # Define or transform response variable. Or use
    # https://spark.apache.org/docs/latest/ml-features.html#binarizer
    data_sdf_i = data_sdf_i.withColumn(Y_name, functions.when(data_sdf_i[Y_name] > 0, 1).otherwise(0))

    # Replace dropped factors with `00_OTHERS`. The trick of `00_` prefix will allow
    # user to drop it as the first level when intercept is used.
    for i in dummy_info['factor_dropped'].keys():
        if len(dummy_info['factor_dropped'][i]) > 0:
            data_sdf_i = data_sdf_i.replace(dummy_info['factor_dropped'][i], '00_OTHERS', i)

    sample_size_sub.append(data_sdf_i.count())
    partition_num_sub.append(ceil(sample_size_sub[file_no_i] / sample_size_per_partition))

    ## Add partition ID
    data_sdf_i = data_sdf_i.withColumn(
        "partition_id",
        monotonically_increasing_id() % partition_num_sub[file_no_i])

    ##----------------------------------------------------------------------------------------
    ## MODELING ON PARTITIONED DATA
    ##----------------------------------------------------------------------------------------

    # from pyspark.ml.feature import StandardScaler
    # scaler = StandardScaler(inputCol="Distance", outputCol="scaledDistance",
    #                         withStd=True, withMean=True)
    # scalerModel = scaler.fit(data_sdf_i)
    # scaledData = scalerModel.transform(data_sdf_i)

    tic_repartition = time.perf_counter()
    data_sdf_i = data_sdf_i.repartition(partition_num_sub[file_no_i], "partition_id")
    time_repartition_sub.append(time.perf_counter() - tic_repartition)

    data_sdf_i_master = data_sdf_i.filter(data_sdf_i.partition_id == 0)
    data_sdf_i_worker = data_sdf_i.filter(data_sdf_i.partition_id > 0)


    sample_df = data_sdf_i_master.toPandas()

    convert_dummies = list(dummy_info['factor_selected'].keys())

    X_with_dummies = pd.get_dummies(data=sample_df,
                                    drop_first=fit_intercept,
                                    columns=convert_dummies,
                                    sparse=True)

    x_m = X_with_dummies.drop(['partition_id', Y_name], axis=1)

    # Check if any dummy column is not in the data chunk.
    usecols_x0 = list(set(sample_df.columns.drop(['partition_id', Y_name])) - set(convert_dummies))
    usecols_x_m = usecols_x0.copy()
    for i in convert_dummies:
        for j in dummy_info["factor_selected_names"][i][fit_intercept:]:
            usecols_x_m.append(j)
    usecols_x_m.sort()
    usecols_full = ['par_id', "grad"]
    usecols_full.extend(usecols_x_m)

    # raise Exception("usecols_full:\t" + str(len(usecols_full)))
    # raise Exception("usecols_x:\t" + str(usecols_x))

    # Standardize the data with global mean and variance
    # if len(data_info) > 0:
    #     for i in usecols_x0:
    #         x_m[i]=(x_m[i] - float(data_info[i][1])) / float(data_info[i][2])

    x_m.sort_index(axis=1, inplace=True)
    x_m = sm.add_constant(x_m)
    x_columns = x_m.columns

    # raise Exception("x_m shape:" + str(list(x_m.columns)))

    y_m = np.array(sample_df[Y_name])

    x_m = np.array(x_m).astype('float')

    logreg = sm.Logit(y_m, x_m)
    beta_b = logreg.fit(disp=0).params


    ## Register a user defined function via the Pandas UDF
    schema_beta = StructType(
        [StructField('par_id', IntegerType(), True),
         StructField('grad', DoubleType(), True)])


    @pandas_udf(schema_beta, PandasUDFType.GROUPED_MAP)
    def logistic_model_udf(sample_df):
        return logistic_model(sample_df=sample_df,
                            beta_b=beta_b,
                            Y_name=Y_name,
                            fit_intercept=fit_intercept,
                            dummy_info=dummy_info)


    # pdb.set_trace()
    # partition the data and run the UDF
    model_mapped_sdf_i = data_sdf_i_worker.groupby("partition_id").apply(logistic_model_udf)

    # Union all sequential mapped results.
    if file_no_i == 0 & isub == 0:
        model_mapped_sdf = model_mapped_sdf_i
        # memsize_sub = sys.getsizeof(data_pdf_i)
    else:
        model_mapped_sdf = model_mapped_sdf.unionAll(model_mapped_sdf_i)


##----------------------------------------------------------------------------------------
## AGGREGATING THE MODEL ESTIMATES
##----------------------------------------------------------------------------------------


n0 = x_m.shape[0]
k = partition_num_sub[0]
g_k1 = np.array(model_mapped_sdf_i.select('grad').collect()).reshape((k - 1, -1))

g_n = grad_n(y_m, x_m, beta_b)

g_k = np.vstack((np.mean(g_n, 0), g_k1))
g_nk1 = np.vstack((g_n, g_k1))

g_avg = np.mean(g_k, 0)

psi = x_m.dot(beta_b)
q2 = expit(psi) / (1 + np.exp(psi))
x_m_w = x_m * np.sqrt(q2)[:, None]
h = x_m_w.T.dot(x_m_w) / n0
beta_t = beta_b - solve(h, g_avg)


B = 500


eps_nk1 = np.vstack((np.random.normal(0, 1, (n0, B)), np.random.normal(0, np.sqrt(n0), (k - 1, B))))
G_nk1 = g_nk1.T.dot(eps_nk1) / np.sqrt(n0 * k * (n0 + k - 1))

bt = np.abs(solve(h, G_nk1))
cd = np.percentile(np.max(bt, 0), 95)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(pd.Series(np.abs(beta_t) > cd, x_columns))



eps_k = np.random.normal(0, 1, (k, B))
G_k = g_k.T.dot(eps_k) / k

bt = np.abs(solve(h, G_k))
cd = np.percentile(np.max(bt, 0), 95)

