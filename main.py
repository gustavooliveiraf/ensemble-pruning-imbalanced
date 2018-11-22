import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import itertools
import seaborn as sns
from collections import Counter
from scipy.stats import  wilcoxon

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 

from sklearn import tree
from sklearn import linear_model

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import cohen_kappa_score

from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import AdaBoostClassifier

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

    def generators(self, generator, x_train, y_train):
        sac = generator(n_estimators=self.pool_size)

        if generator == BaggingClassifier:
            sac.fit(x_train, y_train)
        elif generator == AdaBoostClassifier:
            sample_weight = []
            y_0 = len(y_train) - sum(y_train)
            y_1 = sum(y_train)
            for i in y_train:
                if i == 1:
                    sample_weight.append(1/(y_0/2))
                else:
                    sample_weight.append(1/(y_1/2))

            sac.fit(x_train, y_train, sample_weight)

        return sac

    def main(self, k_fold, n_times, generator, prunning):
        score = (0,0)
        score_pruning = (0,0)

        for i in range(n_times):
            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)

            for train_index, test_index in skf.split(self.x, self.y):
                x_train, x_test = self.x[train_index], self.x[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                x_train, y_train = SMOTE().fit_sample(x_train, y_train) # se comentar buga!!!!!!!!!!!!!!

                bag = self.generators(generator, x_train, y_train)

                bag_prune = self.generators(generator, x_train, y_train)
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
data = pd.read_csv('./cm1Class.csv')

enc = LabelEncoder()
data.CLASS = enc.fit_transform(data.CLASS)

y = np.array(data["CLASS"])
x = np.array(data.drop(axis=1, columns = ["CLASS"]))

scaler = StandardScaler()
x = scaler.fit_transform(x)

modelo = Main(x, y)

generator = [AdaBoostClassifier, BaggingClassifier]
prunning = [prune.reduce_error_GM, prune.complementarity, prune.kappa]

for i in generator:
    print ('\n--------------------------', i, '--------------------------\n\n')
    for j in prunning:
        print (j, '\n')
        modelo.main(3, 1, i, j)
        print ("\n============================================\n")