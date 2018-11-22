import numpy as np
from sklearn.ensemble import BaggingClassifier
from imblearn.metrics import geometric_mean_score
import pandas as pd

from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

def reduce_error_GM(pool = None, X_val = None, y_val = None, pool_size = 100, n = 21):
	estim = np.zeros(len(pool.estimators_))
	for i, est in enumerate(pool.estimators_):
		y_pred = est.predict(X_val)
		estim[i] = geometric_mean_score(y_val, y_pred)
	l = np.argsort(-estim)
	aux = pool.estimators_[:]
	best = list()
	best.append(pool.estimators_[l[0]])
	l = np.delete(l, 0)

	classifier_order = 1
	while len(best) < n :
		scores = np.zeros(len(l))
		for k, j in enumerate(l):
			best.append(aux[j])
			pool.estimators_ = best
			y_pred = pool.predict(X_val)
			scores[k] = geometric_mean_score(y_val, y_pred)
			del best[classifier_order]
		best_score_index = np.argmax(scores)
		best.append(aux[l[best_score_index]])
		l = np.delete(l, best_score_index)
		classifier_order += 1
	pool.estimators_ = aux[:]
	return best

def complementarity(pool = None, X_val = None, y_val = None, pool_size = 100, n = 21):
	estim = np.zeros(len(pool.estimators_))
	for i, est in enumerate(pool.estimators_):
		y_pred = est.predict(X_val)
		estim[i] = geometric_mean_score(y_val, y_pred)
	l = np.argsort(-estim)
	aux = pool.estimators_[:]
	best = list()
	best.append(pool.estimators_[l[0]])
	l = np.delete(l, 0)
	while len(best) < n:
		pool.estimators_ = best
		y_ens = pool.predict(X_val)
		comparison_ens = (y_val == y_ens)
		counting = np.zeros(len(l))
		for k, j in enumerate(l):
			classifier = aux[j]
			y_clas = classifier.predict(X_val)
			comparison_clas = (y_val == y_clas)
			for c, cc in enumerate(comparison_ens):
				if (cc == False and comparison_clas[c]==True):
					counting[k] += 1
		best_index = np.argmax(counting)
		best.append(aux[l[best_index]])
		l = np.delete(l, best_index)
	pool.estimators_ = aux[:]
	return best

def kappa(pool = None, X_val = None, y_val = None, pool_size = 100, n = 21):
    pruning = []
    comb = combinations(range(pool_size), 2)

    for tupla in comb:
        kappa_var = cohen_kappa_score(pool.estimators_[tupla[0]].predict(X_val), pool.estimators_[tupla[1]].predict(X_val))
        pruning.append(tupla + (kappa_var,))

    pruning.sort(key=lambda tup: tup[2])

    ensemble = set()
    for j in pruning:
        ensemble.add(j[0])
        if ((len(ensemble) != n)):
            ensemble.add(j[1])
        if (len(ensemble) == n):
            break

    return [pool.estimators_[i] for i in list(ensemble)]

def MDM(pool = None, X_val = None, y_val = None, n = 21):
	estim = np.zeros(len(pool.estimators_))
	for i, est in enumerate(pool.estimators_):
		y_pred = est.predict(X_val)
		estim[i] = geometric_mean_score(y_val, y_pred)

	l = np.argsort(-estim)
	aux = pool.estimators_[:]
	best = list()
	best.append(pool.estimators_[l[0]])
	l = np.delete(l, 0)
	objective = np.ones(len(y_val))
	classifier_order = 1
	encoder = LabelEncoder()
	y = encoder.fit_transform(y_val)
	y[np.where(y==0)] = -1

	while len(best) < n:
		signature = np.zeros((len(l),len(y_val)))
		for k, j in enumerate(l):
			best.append(aux[j])
			for classif in best:
				out = encoder.fit_transform(classif.predict(X_val))
				out[np.where(out == 0)] = -1
				signature[k] += out*y
			del best[classifier_order]
		signature = signature/len(best)
		distances = np.zeros(len(l))
		for index, vector in enumerate(signature):
			distances[index] = np.linalg.norm(vector - objective)
		best_index = np.argmin(distances)
		best.append(aux[l[best_index]])
		l = np.delete(l, best_index)
		classifier_order += 1

	return best
