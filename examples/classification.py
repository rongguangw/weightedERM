import pandas as pd
from sklearn.model_selection import train_test_split

from .models import WeightedERM


# load in data
data = pd.read_csv('./datasets/neurosynth_subset.csv', low_memory=False)
# remove irrelavant columns (participant id)
drop_cols = ['PTID']
data = data.drop(columns=drop_cols)
# prepare source and target data
# here we use female as source data and male as target data
source_data = data[data.Sex == 'F'].copy()
target_data = data[data.Sex == 'M'].copy()
# split train and test data
# for target data, we use 20% as train data and 80% as test data
source_train, source_test = train_test_split(source_data, test_size=0.2, random_state=42)
target_train, target_test = train_test_split(target_data, test_size=0.8, random_state=42)

# 1. train on source data only, and test on target data
# initialize model with label column name, learning task and evaluation metric
# the choices for 'task_type' are ['binary', 'multiclass', or 'regression']
# the choices for 'metric' are ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'precision', or 'recall']
# you can add argument 'time_limit' to control training time in terms of second
# you can also specify model name using 'model_name' argument
# the saved model directory will be printed after finished training
model = WeightedERM(label_name='Race', task_type='multiclass', metric='accuracy', time_limit=None, model_name=None)
model.fit(source_train)
# evaluate the performance on test set
# setting 'with_prob=True' to output prediction probability
# you can specify 'model_path' to load in a desired model, otherwise load in the model just trained
y_pred, y_prob = model.predict(source_test, with_prob=True, model_path=None)
model.eval(y_prob, source_test.Race)

# 2. train on source data and a small amount of target data, and test on target data
model = WeightedERM(label_name='Race', task_type='multiclass', metric='accuracy', time_limit=None)
model.fit(source_data, target_train)
y_pred, y_prob = model.predict(target_test, with_prob=True)
model.eval(y_prob, target_test.Race)
