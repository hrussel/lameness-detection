import numpy as np
from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, check_random_state, _safe_indexing
from sklearn.utils.validation import _check_method_params
from sklearn.utils._joblib import delayed, Parallel

from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.model_selection._split import check_cv

def permutation_test_score(
    estimator,
    X,
    y,
    *,
    groups=None,
    cv=None,
    n_permutations=100,
    n_jobs=None,
    random_state=0,
    verbose=0,
    scoring=None,
    fit_params=None,
    test=1
):

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    score = _permutation_test_score(
        clone(estimator), X, y, groups, cv, scorer, fit_params=fit_params
    )

    if test == 1:
        permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_permutation_test_score)(
                clone(estimator),
                X,
                _shuffle_classes(y, groups, random_state),
                groups,
                cv,
                scorer,
                fit_params=fit_params,
            )
            for _ in range(n_permutations)
        )
    else:
        permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_permutation_test_score)(
                clone(estimator),
                _shuffle_features(X, y, groups, random_state),
                y,
                groups,
                cv,
                scorer,
                fit_params=fit_params,
            )
            for _ in range(n_permutations)
        )
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)
    return score, permutation_scores, pvalue


def _permutation_test_score(estimator, X, y, groups, cv, scorer, fit_params):
    """Auxiliary function for permutation_test_score"""
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    avg_score = []
    for train, test in cv.split(X, y, groups):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        fit_params = _check_method_params(X, fit_params, train)
        estimator.fit(X_train, y_train, **fit_params)
        avg_score.append(scorer(estimator, X_test, y_test))
    return np.mean(avg_score)


def _shuffle_features(X, y, groups, random_state):
    """
    Test 2 of Ojala and Garriga: Permute Data columns per class.
    :param X: The Data D
    :param y: The class labels
    :param random_state:
    :return: shuffled D'

    Let D = {(Xi, yi)}ni=1 be the data.
    A randomized version D′ of D is obtained by applying independent permutations to the columns of X within each class.
    """

    ## For each class label c ∈ Y do,
    for c in np.unique(y):
        indices = np.arange(X.shape[0])
        ## Let X(c) be the submatrix of X in class label c, that is, X(c) = {Xi | yi = c} of size lc × m
        class_mask = y == c
        if groups is None:
            ## Let π1, . . . , πm be m independent permutations of lc elements
            for m in range(np.shape(X)[1]):
                indices[class_mask] = random_state.permutation(indices[class_mask])
                X[:,m] = _safe_indexing(X[:,m], indices)
        else:
            for group in np.unique(groups):
                this_mask = np.all([groups == group, y == c], axis=0)

                for m in range(np.shape(X)[1]):
                    indices[this_mask] = random_state.permutation(indices[this_mask])
                    X[:, m] = _safe_indexing(X[:, m], indices)
    return X


def _shuffle_classes(y, groups, random_state):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = groups == group
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return _safe_indexing(y, indices)