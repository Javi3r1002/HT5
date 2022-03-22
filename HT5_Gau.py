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
