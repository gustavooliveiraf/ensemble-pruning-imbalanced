import numpy as np
from sklearn.ensemble import BaggingClassifier
from imblearn.metrics import geometric_mean_score
import pandas as pd

def reduce_error_GM(pool = None, X_val = None, y_val = None):
	estim = np.zeros(len(pool.estimators_))
	for i, est in enumerate(pool.estimators_):
		y_pred = est.predict(X_val)
		estim[i] = geometric_mean_score(y_val, y_pred)
	l = np.argsort(-estim)
	aux = pool.estimators_[:]
	best = list()
	best.append(pool.estimators_[l[0]])
	l = np.delete(l, 0)
	i = 1
	while len(best) < 21 :
		scores = np.zeros(len(l))
		for k, j in enumerate(l):
			best.append(aux[j])
			pool.estimators_ = best
			y_pred = pool.predict(X_val)
			scores[k] = geometric_mean_score(y_val, y_pred)
			del best[i]
		best_score_index = np.argmax(scores)
		best.append(aux[l[best_score_index]])
		l = np.delete(l, best_score_index)
		i += 1
	pool.estimators_ = aux[:]
	return best

def complementarity(pool = None, X_val = None, y_val = None, n = 21):
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