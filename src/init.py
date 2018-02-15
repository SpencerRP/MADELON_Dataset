import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, auc, make_scorer, log_loss, precision_recall_curve
from sklearn.svm import SVC
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.cluster import KMeans
#!conda install psycopg2 --yes

import psycopg2 as pg2
from psycopg2.extras import RealDictCursor