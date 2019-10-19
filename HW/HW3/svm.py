# Starting code for UVA CS 4501 ML- SVM

from sklearn.svm import SVC
import random
import numpy as np
import pandas
import sklearn
np.random.seed(37)

# Att: You're not allowed to use modules other than SVC in sklearn, i.e., model_selection.

# Dataset information
# the column names (names of the features) in the data files
# you can use this information to preprocess the features
col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']

# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing.
# For example, as a start you can use one hot encoding for the categorical variables and normalization
# for the continuous variables.


def cross_val_score(clf, x_train, y_train, cv):
    train_scores = []
    test_scores = []
    num_per_fold = x_train.shape[0]//cv
    for i in range(cv):
        #         x_train.iloc[np.r_[0:i*num_per_fold, (i+1)*num_per_fold:]]
        x_train_folds = x_train.iloc[np.r_[
            0:i*num_per_fold, (i+1)*num_per_fold:]]
        y_train_folds = y_train.iloc[np.r_[
            0:i*num_per_fold, (i+1)*num_per_fold:]]
        x_test_fold = x_train.iloc[i*num_per_fold:(i+1)*num_per_fold]
        y_test_fold = y_train.iloc[i*num_per_fold:(i+1)*num_per_fold]
        clf.fit(x_train_folds, y_train_folds)
        # Calculate test accuracy
        y_predict = clf.predict(x_test_fold)
        count = 0
        for yi_train, yi_predict in zip(y_test_fold, y_predict):
            if yi_train == yi_predict:
                count += 1
        test_scores.append(count/num_per_fold)
        # Calculate test fold accuracy
        y_predict = clf.predict(x_train_folds)
        count = 0
        for yi_train_folds, yi_predict in zip(y_train_folds, y_predict):
            if yi_train_folds == yi_predict:
                count += 1
        train_scores.append(count/x_train_folds.shape[0])
    return train_scores, test_scores


def load_data(csv_file_path):
    # your code here
    df = pandas.read_csv("salary.labeled.csv", names=col_names_x+col_names_y)
    # One-hot encoding
    one_hot_df = pandas.get_dummies(df, columns=[
                                    'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
    # normalize
    for col in ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']:
        x, y = df[col].min(), df[col].max()
        one_hot_df[col] = (one_hot_df[col] - x) / (y - x)
    one_hot_df.loc[:, 'label'] = one_hot_df.loc[:, 'label'].str.strip()
    one_hot_df.loc[one_hot_df.label == '<=50K', 'label'] = 0
    one_hot_df.loc[one_hot_df.label == '>50K', 'label'] = 1

    return one_hot_df.loc[:, one_hot_df.columns != 'label'], one_hot_df[col_names_y[0]]

# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.


def train_and_select_model(training_csv):
    # load data and preprocess from filename training_csv
    x_train, y_train = load_data(training_csv)
    # hard code hyperparameter configurations, an example:
    param_set = [
        {'kernel': 'rbf', 'C': 1, 'degree': 1},
        {'kernel': 'rbf', 'C': 1, 'degree': 3},
        {'kernel': 'rbf', 'C': 1, 'degree': 5},
        {'kernel': 'linear', 'C': 1, 'degree': 1},
        {'kernel': 'linear', 'C': 1, 'degree': 3},
        {'kernel': 'linear', 'C': 1, 'degree': 5},
        {'kernel': 'linear', 'C': 1, 'degree': 7},
        {'kernel': 'poly', 'C': 1, 'degree': 1},
        {'kernel': 'poly', 'C': 1, 'degree': 3},
        {'kernel': 'poly', 'C': 1, 'degree': 5},
        {'kernel': 'poly', 'C': 1, 'degree': 7},
        {'kernel': 'sigmoid', 'C': 1, 'degree': 3},
        {'kernel': 'sigmoid', 'C': 1, 'degree': 5},
        {'kernel': 'sigmoid', 'C': 1, 'degree': 7},
    ]
    # your code here
    scores = []
    best_params = None
    best_score = 0
    for params in param_set:
        clf = sklearn.svm.SVC(
            kernel=params['kernel'], C=params['C'], degree=params['degree'], gamma='auto')
        # iterate over all hyperparameter configurations
        # perform 3 FOLD cross validation
        cv_train_scores, cv_test_scores = cross_val_score(
            clf, x_train, y_train, cv=3)
        print(params)
        print('cv_train_scores: {}'.format(cv_train_scores))
        print('cv_test_scores: {}'.format(cv_test_scores))
        mean_score = np.mean(cv_test_scores)
        if mean_score > best_score:
            best_params = params
            best_score = mean_score
        # print cv scores for every hyperparameter and include in pdf report
    # select best hyperparameter from cv scores, retrain model
    best_model = sklearn.svm.SVC(
        kernel=best_params['kernel'], C=best_params['C'], degree=best_params['degree'], gamma='auto')
    best_model.fit(x_train, y_train)
    return best_model, best_score

# predict for data in filename test_csv using trained model


def predict(test_csv, trained_model):
    x_test, _ = load_data(test_csv)
    predictions = trained_model.predict(x_test)
    return predictions

# save predictions on test data in desired format


def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')


if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    # fill in train_and_select_model(training_csv) to
    # return a trained model with best hyperparameter from 3-FOLD
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter.
    # hardcode hyperparameter configurations as part of train_and_select_model(training_csv)
    trained_model, cv_score = train_and_select_model(training_csv)

    print("The best model was scored %.2f" % cv_score)
    # use trained SVC model to generate predictions
    predictions = predict(testing_csv, trained_model)
    # Don't archive the files or change the file names for the automated grading.
    # Do not shuffle the test dataset
    output_results(predictions)
    # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
