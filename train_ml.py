import os.path
import sklearn as skl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import contextlib

import core.config
import gait_data.gait_dataset_sklearn as gait_dataset

from sklearn.model_selection import train_test_split, \
    StratifiedShuffleSplit, \
    StratifiedGroupKFold, \
    RandomizedSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from scipy.stats import loguniform
from sklearn.multiclass import OneVsRestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
import core.sk_utils
from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE




def eval_predictions(y_true, y_pred, average='weighted', verbose=False, title='', log_path=None, show=False):
    metrics = {}
    metrics['accuracy'] = sklearn.metrics.accuracy_score(y_true, y_pred)
    metrics[f'f1_{average}'] = sklearn.metrics.f1_score(y_true, y_pred, average=average)
    metrics[f'precision_{average}'] = sklearn.metrics.precision_score(y_true, y_pred, average=average)
    metrics[f'recall_{average}'] = sklearn.metrics.recall_score(y_true, y_pred, average=average)
    metrics[f'ROC_AUC_{average}'] =  sklearn.metrics.roc_auc_score(y_true, y_pred, average=average)
    metrics['mcc'] = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    metrics['cohen'] = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    metrics['cohen_w'] = sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights='linear')
    metrics['c_matrix'] = sklearn.metrics.confusion_matrix(y_true, y_pred)
    metrics['f1_binary'] = sklearn.metrics.f1_score(y_true, y_pred, average='binary', pos_label=1)
    metrics['balanced_accuracy'] = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    metrics['zero_one_loss'] = sklearn.metrics.zero_one_loss(y_true, y_pred)
    metrics['sensitivity'] = sklearn.metrics.recall_score(y_true, y_pred, pos_label=1)
    metrics['specificity'] = sklearn.metrics.recall_score(y_true, y_pred, pos_label=0)
    cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    cm.figure_.suptitle(title)
    # cm.plot()
    if log_path is None and show:
        plt.show()
    elif log_path is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'cm.png'))

    if verbose:
        for key, val in metrics.items():
            print(f"{key}:\n{val}")
    return metrics


def eval_predictions_per_fold(y_true, y_pred, average='macro', verbose=False, title='', log_path=None):
    all_metrics = None
    stacked_y_true = []
    stacked_y_pred = []


    for f in range(len(y_true)):

        fold_metrics = eval_predictions(y_true[f], y_pred[f],  average, verbose=False, title=title, log_path=None, show=False)

        if all_metrics is None:
            all_metrics = dict((key, []) for key in fold_metrics.keys())

        for key, value in fold_metrics.items():
            all_metrics[key].append(value)

        stacked_y_true.extend(y_true[f])
        stacked_y_pred.extend(y_pred[f])

    mean_metrics = dict((key, np.mean(all_metrics[key])) for key in all_metrics.keys())
    std_metrics = dict((key, np.std(all_metrics[key])) for key in all_metrics.keys())

    stacked_metrics = eval_predictions(stacked_y_true, stacked_y_pred, average=average, verbose=False, title=title, log_path=None, show=False)

    cm = confusion_matrix(stacked_y_true, stacked_y_pred)

    if verbose:
        print('Metrics average per fold:')
        print('-------------------------')
        for key in mean_metrics.keys():
            print(f"{key}:\n{mean_metrics[key]} (+/-) {std_metrics[key]}")

        print('\n********\n')
        print('Metrics stacked for all folds:')
        print('------------------------------')
        for key in stacked_metrics.keys():
            print(f"{key}:\n{stacked_metrics[key]}")
        print(f'Confusion matrix:\n{cm}')

    return mean_metrics, std_metrics, cm

def plot_perm_importance(rs, feature_names, title='', filename='feat_imp.png', log_path=None):
    # Loop through each fold
    importances_mean = []
    importances_std = []
    importances = []
    for fold in range(len(rs)):
        importances_mean.append(rs[fold]['importances_mean'])
        importances_std.append(rs[fold]['importances_std'])
        importances.append(rs[fold]['importances'])

    importances_mean = np.array(importances_mean)
    importances_std = np.array(importances_std)
    importances = np.array(importances)
    feature_names = np.array(feature_names)
    imp_mean = np.mean(importances_mean, axis=0)
    imp_std = np.mean(importances_std, axis=0)
    imp = np.mean(importances, axis=0)

    perm_sorted_idx = imp_mean.argsort()
    indices = np.arange(0, perm_sorted_idx.shape[0]) + 0.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(indices, imp_mean[perm_sorted_idx], height=0.7)
    ax1.errorbar(imp_mean[perm_sorted_idx], indices, xerr=imp_std[perm_sorted_idx], c='k', fmt='o')
    ax1.set_yticks(indices)
    ax1.set_yticklabels(feature_names[perm_sorted_idx])
    ax1.set_ylim((0, perm_sorted_idx.shape[0]))

    ax2.boxplot(imp[perm_sorted_idx].T, vert=False, labels=feature_names[perm_sorted_idx] )
    plt.suptitle(title)
    fig.tight_layout()

    if log_path is None:
        plt.show()
    else:
        imp_mean_std = np.stack([feature_names, imp_mean, imp_std]).T
        df = pd.DataFrame(imp_mean_std, columns=['feat', 'mean', 'std'])
        df.to_csv(os.path.join(log_path, f'{filename}.csv'), index=False)

        df = pd.DataFrame(imp.T, columns=feature_names)
        df.to_csv(os.path.join(log_path, f'{filename}_all.csv'), index=False)

        plt.savefig(os.path.join(log_path, filename))


# def nested_cross_validation(dataset, model, params, k_inner, k_outer, seed, scoring_in='f1_weighted', scoring_out=None, n_iter=100, njobs=1):
#     X, y, g = dataset.data, dataset.labels, dataset.ids
#     inner_cv = StratifiedShuffleSplit(n_splits=k_inner, random_state=seed)
#     outer_cv = StratifiedGroupKFold(n_splits=k_outer, shuffle=True, random_state=seed)
#
#     if scoring_out is None:
#         scoring_out = scoring_in
#
#     clf = RandomizedSearchCV(estimator=model,
#                              param_distributions=params,
#                              cv=inner_cv,
#                              scoring=scoring_in,
#                              verbose=1,
#                              n_jobs=njobs,
#                              error_score=0,
#                              refit=True,
#                              n_iter=n_iter,
#                              random_state=seed
#                              )
#
#     nested_score = cross_validate(clf, X=X, y=y, groups=g, cv=outer_cv, scoring=scoring_out, verbose=1, n_jobs=njobs)
#
#     return nested_score


def cv_grid_search(dataset, model, params, cv, dataset_index=None, scoring='f1_weighted', n_iter=100, seed=42, njobs=1):
    if dataset_index is None:
        X, y, g = dataset.data, dataset.labels, dataset.ids

    else:
        X, y, g = dataset.data[dataset_index], dataset.labels[dataset_index], dataset.ids[dataset_index]

    clf = RandomizedSearchCV(estimator=model,
                             param_distributions=params,
                             cv=cv, scoring=scoring,
                             verbose=1,
                             n_jobs=njobs,
                             error_score=0,
                             refit=True,
                             n_iter=n_iter,
                             random_state=seed
                             )
    clf.fit(X,y, groups=g)
    print(clf.best_params_, clf.best_score_)
    return clf


def manual_nested_cv(dataset, model_type, k_inner, k_outer, seed, scoring_in='f1_macro', feat_importance=0, n_iter=100, njobs=-1):
    ## see https://github.com/rosscleung/Projects/blob/b9abc20db545d9f483e90a9b046ea50c74f25718/Tutorial%20notebooks/Nested%20Cross%20Validation%20Example.ipynb
    rng = np.random.RandomState(seed)

    # Following kf is the outer loop
    outer_kf = StratifiedGroupKFold(n_splits=k_outer, shuffle=True, random_state=rng)
    # inner_kf = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=rng)

    outer_loop_y_pred = []
    outer_loop_y_true = []
    test_folds = [["videos_test", "id_test", "y_test", "y_pred"]]

    r_s_train = []
    r_s_test = []

    # Looping through the outer loop, feeding each training set into a GSCV as the inner loop
    for o_i, (train_index, test_index) in enumerate(outer_kf.split(dataset.data, dataset.labels, dataset.ids)):

        # RSCV is looping through the training data to find the best parameters. This is the inner loop
        RSCV = main_simple_cv_search(model_type, dataset, seed, k_inner, scoring_in, n_iter=n_iter, dataset_index=train_index)

        # refit best model on outer training data
        RSCV.fit(dataset.data[train_index], dataset.labels[train_index])

        # predict on outer test data
        pred = RSCV.predict(dataset.data[test_index])

        if feat_importance:
            r_tain = permutation_importance(RSCV, dataset.data[train_index], dataset.labels[train_index], scoring=scoring_in, n_repeats=feat_importance, random_state=seed, n_jobs=njobs)
            r_s_train.append(r_tain)
            r_test = permutation_importance(RSCV, dataset.data[test_index], dataset.labels[test_index], scoring=scoring_in, n_repeats=feat_importance, random_state=seed, n_jobs=njobs)
            r_s_test.append(r_test)
            print_feat_importance(r_tain['importances'], dataset.feature_names)
            print_feat_importance(r_test['importances'], dataset.feature_names)

        outer_loop_y_pred.append(pred)
        outer_loop_y_true.append(dataset.labels[test_index])

        test_folds.append([dataset.video_names[test_index], dataset.ids[test_index], dataset.labels[test_index], pred])

    return outer_loop_y_true, outer_loop_y_pred, r_s_train, r_s_test, test_folds


def train_test_K_fold(dataset, model, n_folds, seed, feat_importance=0):

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    test_evals = None
    y_true = []
    y_preds = []
    r_s_train = []
    r_s_test = []
    test_folds = [["videos_test", "id_test", "y_test", "y_pred"]]


    for i, (train_index, test_index) in enumerate(sgkf.split(dataset.data, dataset.labels, dataset.ids)):

        X_train, y_train, id_train = dataset.data[train_index], dataset.labels[train_index], dataset.ids[train_index]
        X_test, y_test, id_test = dataset.data[test_index], dataset.labels[test_index], dataset.ids[test_index]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        y_preds.append(preds)
        y_true.append(y_test)
        if feat_importance:
            r = permutation_importance(model, X_test, y_test, scoring='f1_macro', n_repeats=feat_importance, random_state=seed, n_jobs=1)
            r_s_test.append(r)
            r = permutation_importance(model, X_train, y_train, scoring='f1_macro', n_repeats=feat_importance, random_state=seed, n_jobs=1)
            r_s_train.append(r)

        videos_test = dataset.video_names[test_index]
        test_folds.append([videos_test, id_test, y_test, preds])

    return y_true, y_preds, r_s_train, r_s_test, test_folds


def get_hyper_params_range(model_type):
    p_random = None

    if model_type == 'DT':
        p_random = {'model__estimator__max_depth': [*np.linspace(10, 1000, num=100, dtype=int), None],
                    'model__estimator__min_samples_split': np.linspace(2, 10, num=10, dtype=int),
                    'model__estimator__min_samples_leaf': np.linspace(1, 10, num=10, dtype=int),
                    'model__estimator__min_weight_fraction_leaf': np.linspace(0.0, 0.5, num=10, dtype=float)

                    }

    elif model_type == 'RF':
        p_random = {'model__estimator__n_estimators': np.linspace(10, 1000, num=100, dtype=int),
                    'model__estimator__max_depth': [*np.linspace(10, 1000, num=100, dtype=int), None],
                    'model__estimator__min_samples_leaf': np.linspace(1, 10, num=10, dtype=int),
                    'model__estimator__min_samples_split': np.linspace(2, 10, num=10, dtype=int)
                    }

    elif model_type == 'SVC_L':
        p_random = {'model__C': loguniform(1e0, 1e3),
                    'model__penalty': ['l1', 'l2']
                    }

    elif model_type == 'SVC_R':
        p_random = {'model__estimator__C': loguniform(1e0, 1e4),
                    'model__estimator__gamma': loguniform(1e-6, 1e-1)}

    elif model_type == 'GB':
        p_random = {'model__estimator__loss': ['log_loss', 'exponential'],
                    'model__estimator__learning_rate': loguniform(1e-5, 1e-1),
                    'model__estimator__n_estimators': np.linspace(1, 10000, num=500, dtype=int)
                    }

    elif model_type == 'MLP':
        p_random = {'model__estimator__hidden_layer_sizes': [(56,56), (56,56,56),
                                                             (128,128), (128, 128, 128),
                                                             (256,256), (256, 256,256),
                                                             ],
                    'model__estimator__solver': ["lbfgs", "adam"],
                    'model__estimator__activation': ["identity", "logistic", "relu"],
                    'model__estimator__learning_rate_init': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5 ],
                    'model__estimator__max_iter': np.linspace(10000, 1000000, num=10, dtype=int),
                    'model__estimator__batch_size': [4, 8, 16, 'auto']
                    # 'model__estimator__max_iter': np.linspace(2000, 100000, num=50, dtype=int)
                    # 'model__estimator__momentum': np.linspace(0,1, num=20, dtype=float)
        }

    elif model_type == 'LR':
        p_random = {'model__solver': ['lbfgs', 'saga'],
                    'model__penalty': ['l2'],
                    'model__C': loguniform(1e0, 1e4),
                    'model__max_iter': np.linspace(100, 1000, num=10, dtype=int)
        }

    return p_random


def main_simple_cv_search(model_type, dataset, seed, folds, scoring, dataset_index=None, n_iter=100):
    rng = np.random.RandomState(seed)

    if model_type == 'DT':
        print("Decision Tree:")
        model = DecisionTreeClassifier(random_state=rng)
        model = OneVsRestClassifier(model)

    elif model_type == 'RF':
        print("Random Forest:")
        model = RandomForestClassifier(random_state=rng)
        model = OneVsRestClassifier(model)

    elif model_type == 'SVC_L':
        print("SVC Linear:")
        model = LinearSVC(class_weight='balanced', dual=False, random_state=rng, max_iter=1e5)

    elif model_type == 'SVC_R':
        print("SVC Radial:")
        model = SVC(class_weight='balanced', kernel='rbf', random_state=rng, max_iter=1e5)
        model = OneVsRestClassifier(model)

    elif model_type == 'GB':
        print("Gradient Boosting:")
        model = GradientBoostingClassifier(random_state=rng)
        model = OneVsRestClassifier(model)

    elif model_type == 'MLP':
        print("MLP CLassifier")
        model = MLPClassifier(random_state=rng)
        model = OneVsRestClassifier(model)

    elif model_type == 'LR':
        print('Logistic Regression')
        model = LogisticRegression(multi_class='ovr', class_weight='balanced', random_state=rng)

    else:
        print(f'ERROR: Unknown model type {model_type}')
        return None

    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)  # re-create the cv splitter for consistent splits.

    sampler = SMOTE(random_state=seed)
    scaler = RobustScaler()
    # pipe = Pipeline(steps=[('sampler', sampler), ('scaler', scaler), ('model', model)])
    pipe = Pipeline(steps=[('scaler', scaler), ('model', model)])

    p_random = get_hyper_params_range(model_type)

    ## before nested
    n_iter = int(n_iter / folds)
    clf = cv_grid_search(dataset, pipe, p_random, cv, dataset_index=dataset_index, scoring=scoring, n_iter=n_iter,seed=seed, njobs=-1)

    best_clone = skl.clone(skl.clone(clf.best_estimator_).set_params(**clf.best_params_))

    return best_clone


def load_model(model_path):
    if not model_path or not os.path.exists(model_path):
        return None, None
    else:
        print("Loading model from ", model_path)
        config, model = joblib.load(model_path)

    return config, model


def save_model(model, config):
    flat = 'FLAT' if config.flat_cv else 'NEST'
    save_fname = f"{config.model_type}_{flat}_K{config.n_folds}_{config.merging}_"
    for f in config.features:
        save_fname += f"{f}_"
    save_fname = save_fname + (config.gait_scores_csv.split('/')[-1]).split('.')[0] + '_'
    save_fname = save_fname + datetime.now().strftime('%b%d') + '.joblib'

    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)

    save_path = os.path.join(config.save_path, save_fname)
    if model is not None:
        joblib.dump([config, model], save_path)

    return save_path


def print_feat_importance(rs, feature_names):
    mean_rs = np.mean(rs, axis=1)
    std_rs = np.std(rs, axis=1)

    for i in mean_rs.argsort():
        print(f"{feature_names[i]:<8}"
              f"{mean_rs[i]:.3f}"
              f" +/- {std_rs[i]:.3f}")


if __name__ == '__main__':

    config, _ = core.config.parse_args('Train ML')

    '''
    Notes on the Random State.
    - It is better to pass a rng INSTANCE to a classifier/regressor.
    - It is better to pass a INTEGER to a cross-validation.
    - For CV, make sure that the split is called ONCE, or reset the cv instance with the random state integer.
    For a full overview, see https://scikit-learn.org/dev/common_pitfalls.html
    '''
    seed = 42
    rng = np.random.RandomState(seed)

    use_wandb = False
    dataset = gait_dataset.load_gait_dataset(config)

    FLAT = config.flat_cv
    print("FLAT CV?", FLAT)
    feat_importance = 100  # 100
    n_iter = 100

    model_config, model = load_model(config.load_model)
    if model is None:
        if FLAT:
            print("No model loaded, searching best params")
            model = main_simple_cv_search(config.model_type, dataset, seed, config.n_folds, scoring='f1_macro', n_iter=n_iter)
            print("Saved model at ", config.load_model)

        config.load_model = save_model(model, config)
    #
    log_path = config.load_model.split('.joblib')[0]
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    out_path = os.path.join(log_path, 'out.txt')
    print(out_path)
    with open(out_path, 'w') as f:
        with contextlib.redirect_stdout(f):
            print(config)

            if FLAT: # FLAT CV
                print(config.model_type, model.get_params())
                print("FLAT CV?", FLAT)

                y_true, y_preds, rs_train, rs_test, test_folds = train_test_K_fold(dataset, model, config.n_folds, seed, feat_importance=feat_importance)

            else: # NESTED CV
                print("FLAT CV?", FLAT)
                y_true, y_preds, rs_train, rs_test, test_folds = manual_nested_cv(dataset, config.model_type, k_inner=config.n_folds*2, k_outer=config.n_folds, seed=seed, feat_importance=feat_importance, n_iter=n_iter)

            eval_predictions_per_fold(y_true, y_preds, average='macro', verbose=True, title=f"{config.model_type} {config.merging}", log_path=log_path)

            header = np.array(test_folds[0])
            for i, fold in enumerate(test_folds[1:]):
                res_path = os.path.join(log_path, f'test_fold_{i}.csv')
                fold = np.array(fold).T
                fold = np.vstack((header, fold))
                np.savetxt(res_path, fold, fmt='%s', delimiter=',')

            if feat_importance:
                plot_perm_importance(rs_train, dataset.feature_names, title=f"{config.model_type} {config.merging} - Train", filename=f"{config.model_type}_train_feat_imp.png", log_path=log_path)
                plot_perm_importance(rs_test, dataset.feature_names, title=f"{config.model_type} {config.merging} - Test loop", filename=f"{config.model_type}_test_feat_imp.png", log_path=log_path)
