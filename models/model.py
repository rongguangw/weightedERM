import os
import math
import shutil
import numpy as np
import pandas as pd
import autogluon.core as ag
from datetime import datetime
from fastai.tabular.all import *
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


class WeightedERM():
    def __init__(
        self,
        label_name='Diagnosis',
        random_seed=42,
        refit=True,
        task_type='binary', # 'binary', 'multiclass', or 'regression'
        metric=None, # 'accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'precision', 'recall', 'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', or 'r2'
        time_limit=None,
        save_dir='./logs',
        model_name=None,
        alpha_range=[1,10]
        ):
        super(WeightedERM, self).__init__()
        self.label_name = label_name
        self.random_seed = random_seed
        self.refit = refit
        self.task_type = task_type
        self.metric = metric
        self.time_limit = time_limit
        self.save_dir = save_dir
        self.model_name = model_name
        self.alpha_range = alpha_range

        if self.metric is None:
            if self.task_type == 'binary' or 'multiclass':
                self.metric = 'accuracy'
            elif self.task_type == 'regression':
                self.metric = 'root_mean_squared_error'

    def model_path(self, alpha=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if alpha is not None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
            path = os.path.join(self.save_dir, "weighted-erm-{}-alpha-{}{}".format(timestamp, str(alpha), os.path.sep))
        elif self.model_name is not None:
            path = os.path.join(self.save_dir, "weighted-erm-{}{}".format(str(self.model_name), os.path.sep))
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
            path = os.path.join(self.save_dir, "weighted-erm-{}{}".format(timestamp, os.path.sep))
        return path

    def hyperparameter_tuning(self, X_s, X_t):
        best = {'score': -np.inf, 'alpha': 0, 'path': None}
        for alpha in range(self.alpha_range[0], self.alpha_range[1]+1):
            print(f"\n=> Training with hyperparameter alpha = {alpha}\n")
            X_s['Weight'] = 1.0
            X_t['Weight'] = alpha * X_s.shape[0]/X_t.shape[0]
            X = X_s.append(X_t, ignore_index=True)
            current = {'alpha': alpha, 'path': self.model_path(alpha)}
            model = TabularPredictor(label=self.label_name, sample_weight='Weight', problem_type=self.task_type, eval_metric=self.metric, path=current['path'])
            model.fit(X_s, tuning_data=X_t, presets='best_quality', refit_full=self.refit, use_bag_holdout=True, time_limit=self.time_limit)
            current['score'] = model.leaderboard(silent=True).score_val[0]
            if current['score'] > best['score']:
                if best['path'] is not None:
                    shutil.rmtree(best['path'])
                best['score'] = current['score']
                best['alpha'] = current['alpha']
                best['path'] = current['path']
            else:
                shutil.rmtree(current['path'])
        if self.model_name is not None:
            path = self.model_path()
            os.rename(best['path'], path)
            best['path'] = path
        predictor = TabularPredictor.load(best['path'])
        return predictor, best

    def fit(self, data_source, data_target=None):
        if self.label_name not in data_source.columns.tolist():
            raise ValueError(f"'{self.label_name}' is not included in the source data.")
        if data_target is None:
            path = self.model_path()
            self.predictor = TabularPredictor(label=self.label_name, problem_type=self.task_type, eval_metric=self.metric, path=path)
            self.predictor.fit(data_source, presets='best_quality', refit_full=self.refit, use_bag_holdout=True, time_limit=self.time_limit)
            print("\n========  Model Perforamnce on Source Data ========\n")
            print("{}: {}".format(self.metric, self.predictor.leaderboard(data_source, silent=True).score_test[0]))
            if 'error' in self.metric:
                print("Note: Scores are always higher_is_better. This metric score can be multiplied by -1 to get the metric value.")
            print("\n=> Model has been saved at: {}.".format(path))
        else:
            if self.label_name not in data_target.columns.tolist():
                raise ValueError(f"'{self.label_name}' is not included in the target data.")
            if data_source.shape[1] != data_target.shape[1]:
                raise ValueError(f"Source data and Target data must have the same columns.")
            self.predictor, params = self.hyperparameter_tuning(data_source, data_target)
            print("\n========  Model Perforamnce on Source Data ========\n")
            print("{}: {}".format(self.metric, self.predictor.leaderboard(data_source, silent=True).score_test[0]))
            if 'error' in self.metric:
                print("Note: Scores are always higher_is_better. This metric score can be multiplied by -1 to get the metric value.")
            print("\n========  Model Perforamnce on Target Data ========\n")
            print("{}: {}".format(self.metric, self.predictor.leaderboard(data_target, silent=True).score_test[0]))
            if 'error' in self.metric:
                print("Note: Scores are always higher_is_better. This metric score can be multiplied by -1 to get the metric value.")
            print("\n=> The optimal hyperparameter alpha is: {}.".format(str(params['alpha'])))
            print("\n=> Model has been saved at: {}.".format(params['path']))

    def predict(self, data_test, model_path=None, with_prob=False):
        if not self.predictor and model_path is None:
            raise ValueError(f"Please specify 'model_path' or fit a model using the 'fit()' function.")
        elif model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(f"Directory {model_path} does not exists.")
            else:
                self.predictor = TabularPredictor.load(model_path)
        if self.task_type == 'regression':
            with_prob = False
        y_pred = self.predictor.predict(data_test)
        if with_prob:
            y_prob = self.predictor.predict_proba(data_test)
            return y_pred, y_prob
        else:
            return y_pred

    def eval(self, y_pred, y_test):
        self.predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
