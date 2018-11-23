import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.pylab import rcParams
# import itertools
# import seaborn as sns
# from collections import Counter
# from scipy.stats import  wilcoxon

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn import tree
from sklearn import linear_model

from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import cohen_kappa_score

from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier

import prune

class Main:
    def __init__(self, x, y):
        self.y = np.array(y)
        self.x = np.array(x)
        self.pool_size = 30
        self.n = 21

    def calc_metrics(self, y_samples, y_true):
        proc_auc_score_temp = roc_auc_score(y_samples, y_true)
        geometric_mean_score_temp = geometric_mean_score(y_samples, y_true)

        return (proc_auc_score_temp, geometric_mean_score_temp)

    def normalize(self, x_train, y_train, method_normalize, generator):
        if generator == EasyEnsembleClassifier:
            return (x_train, y_train)
        else:
            if method_normalize == SMOTE:
                return SMOTE().fit_sample(x_train, y_train)
            elif method_normalize == RandomUnderSampler:
                return RandomUnderSampler().fit_sample(x_train, y_train)

    def main(self, k_fold, n_times, generator, method_normalize, prunning):
        for i in range(n_times):
            score = (0,0)
            score_pruning = (0,0)

            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)
            for train_index, test_index in skf.split(self.x, self.y):
                x_train, x_test = self.x[train_index], self.x[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                x_train, y_train = self.normalize(x_train, y_train, method_normalize, generator)

                bag = generator(n_estimators=self.pool_size)
                bag.fit(x_train, y_train)

                bag_prune = generator(n_estimators=self.pool_size)
                bag_prune.fit(x_train, y_train)
                bag_prune.estimators_ = prunning(bag, x_train, y_train, self.pool_size, self.n)

                score = tuple(map(sum, zip(score, self.calc_metrics(bag.predict(x_test), y_test))))
                score_pruning = tuple(map(sum, zip(score_pruning, self.calc_metrics(bag_prune.predict(x_test), y_test))))

            score = tuple(map(lambda x: x/k_fold, score))
            score_pruning = tuple(map(lambda x: x/k_fold, score_pruning))

            print('---------aux/g-mean original/pruning--------')
            print(score)
            print(score_pruning, '\n')

        return

# test
data = np.genfromtxt(fname = 'data/ecoli1.dat', comments='@', delimiter=',', autostrip=True)
x = data[:,:-1]
data = pd.read_csv('data/ecoli1.dat', comment='@', header = None, delimiter=',', delim_whitespace=True)
y = data.iloc[:,-1].values

enc = LabelEncoder()
y = enc.fit_transform(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)

modelo = Main(x, y)

generator = [BaggingClassifier, AdaBoostClassifier, EasyEnsembleClassifier] # RUSBoostClassifier
samplings = [SMOTE, RandomUnderSampler]
prunning  = [prune.boosting, prune.MDM, prune.complementarity, prune.kappa, prune.reduce_error_GM]

for i in generator:
    print ('\n=======================', i, '=======================\n\n')
    for j in samplings:
        print ('\n--------------------------', j, '--------------------------\n\n')
        for k in prunning:
            print (k, '\n')
            modelo.main(3, 1, i, j, k)
            print ("\n============================================\n")
        if i == EasyEnsembleClassifier:
            break
