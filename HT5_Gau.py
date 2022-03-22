import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import sklearn.mixture as mixture
import pyclustertend 
import random
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import silhouette_score, mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from kmodes.kprototypes import KPrototypes
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from kneed import KneeLocator
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.cm as cm
import sys
import warnings
import scipy.stats as stats
import pylab
import timeit
from scipy.stats import shapiro
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot
from yellowbrick.regressor import ResidualsPlot
import statsmodels.stats.diagnostic as diag
from scipy.stats import normaltest
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import make_scorer, accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate



from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
Data = pd.read_csv('train.csv')

#Test = pd.read_csv('test.csv')

#print(Data)

normal = Data.select_dtypes(include = np.number)
CN = normal.columns.values
normal = normal.dropna()
r = ''

fig = plt.figure()
g = 0
for i in CN:
    estadistico1, p_value1 = stats.kstest(normal[i], 'norm')

    if p_value1 > 0.5:
        r = 'Es normal'
    else:
        r = 'no es normal'

    plt.subplot(7,7,g+1)
    sns.distplot(normal[i])
    plt.xlabel(i)
    g+= 1

    print(i, ": ", r)

plt.tight_layout()
plt.show()



#print(normal.describe())

normal = normal.drop(['Id', 'LowQualFinSF', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'FullBath', 'MoSold', 'YrSold', 'MSSubClass', 'OverallCond', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'WoodDeckSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'LotFrontage', 'LotArea', 'MasVnrArea', '2ndFlrSF', 'TotRmsAbvGrd', 'Fireplaces', 'OpenPorchSF' ], axis = 1)
vif_data = pd.DataFrame() 
G = normal
vif_data["feature"] = G.columns
  
vif_data["VIF"] = [variance_inflation_factor(G.values, i) 
                          for i in range(len(G.columns))] 
  
print(vif_data)
correlation_mat = normal.corr()
NC = normal.columns.values

SP = correlation_mat.iloc[-1]

SaleP = normal[['SalePrice']]


sns.heatmap(correlation_mat, annot = True)
plt.tight_layout()
plt.show()
#normal = normal.drop(['1stFlrSF', 'OverallQual', 'GarageCars'], axis = 1)

CN = normal.columns.values

H = normal
X = np.array(normal)
X.shape
print('Hopkins', pyclustertend.hopkins(X,len(X)))



random.seed(5)

km = cluster.KMeans(n_clusters=3, random_state = 5).fit(X)
centroides = km.cluster_centers_
#print(centroides)


normal = km.predict(X)
plt.scatter(X[normal == 0, -1], X[normal == 0, -1],s=100,c='red', label = "Cluster 1")
Cluster_bajo = X[normal == 0, -1]
Cluster_bajo = Cluster_bajo.tolist()
print('máximo primer cluster (rojo)', max(Cluster_bajo))
print('mínimo primer cluster (rojo)',min(Cluster_bajo))
Barato = H[((H['SalePrice']>(min(Cluster_bajo)))& (H['SalePrice']<max(Cluster_bajo)))]

plt.scatter(X[normal == 1, -1], X[normal == 1, -1],s=100,c='blue', label = "Cluster 2")
Cluster_medio = X[normal == 1, -1]
Cluster_medio = Cluster_medio.tolist()
print('máximo segundo cluster (azul)', max(Cluster_medio))
print('mínimo segundo cluster (azul)',min(Cluster_medio))
Medio = H[((H['SalePrice']>(min(Cluster_medio))) & (H['SalePrice']<max(Cluster_medio)))]

plt.scatter(X[normal == 2, -1], X[normal == 2, -1],s=100,c='green', label = "Cluster 3")
Cluster_alto = X[normal == 2, -1]
Cluster_alto = Cluster_alto.tolist()
print('máximo tercer cluster (verde)', max(Cluster_medio))
print('mínimo tercer cluster (verde)',min(Cluster_medio))
Alto =  H[((H['SalePrice']>(min(Cluster_alto))) & (H['SalePrice']<max(Cluster_alto)))]

plt.scatter(km.cluster_centers_[:,-1],km.cluster_centers_[:,-1], s=300, c="yellow",marker="*", label="Centroides")
plt.title("Grupo casa")
plt.xlabel("Precio de venta")
plt.ylabel("Precio de venta")
plt.legend()
