import numpy as np
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
parameters = {'depth'         : [4,5,6,7,8,9, 10],
                 'learning_rate' : [0.01,0.02,0.03,0.04,0.08,0.1],
                  'iterations'    : [ 100,120,130,150,160,170,180]
                 }

def catboost_training(train, test, features, cat_features, targets, learning_rate=0.01, iterations=150, depth=4, classif=False):
    """
    Train a Catboost model

    inputs:
        train (pandas DataFrame): Training DataFrame
        test (pandas DataFrame): Testing DataFrame
        features (list[str]): List of columns to consider as variables during the training
        cat_features (list[int]): List of categorical labels
        targets (list[str]): Targets
        learning_rate (float): Learning rate
        iterations (int): Number of iterations
        depth (int): Depth of model
        classif (bool): If true, use a classification model, use regression otherwise

    output:
        model (CatBoost): CatBoost fit model

    """
    models = []
    for i, target in enumerate(targets):
        print("==== Target ", target, " ====")
        print("Iteration ", i + 1, "/", len(targets))

        relevant = train[features + [target]].dropna()

        valid = test[features + [target]].dropna()

        # Training model

        train_dataset = Pool(data=relevant[features],
                             label=relevant[target],
                             cat_features=cat_features)
        valid_dataset = Pool(data=valid[features],
                             label=valid[target],
                             cat_features=cat_features)
        if classif:
            clf = CatBoostClassifier(iterations=iterations,
                                     learning_rate=learning_rate,
                                     depth=depth,
                                     loss_function="MultiClass")
            clf.fit(train_dataset, eval_set=valid_dataset)
        else:
            clf = CatBoostRegressor(iterations=150,
                                    learning_rate=0.1,
                                    depth=6,
                                    loss_function="MAE")
            clf.fit(train_dataset, eval_set=valid_dataset)
        models.append(clf)
    return models


def catboost_test_score(model):
    """
    Test score of the Catboost model

    input:
        model (CatBoost): CatBoost model

    output:
        (float): Score
    """
    return round(model.get_best_score()['validation']['MAE'], 3)


def catboost_prediction(models, test, features, targets, level_col):
    """
    Prediction using CatBoost model

    inputs:
        models (list[CatBoost]): List of catboost models
        test (pandas DataFrame): Testing DataFrame
        features (list[str]): List of columns to consider as variables during the training
        targets (list[str]): Targets
        level_col ():
    """

    relevant = test[features + ['date']
                    ].dropna().reset_index(drop=True)

    for i, _ in enumerate(models):
        print("==== Target ", targets[i], " ====")
        print("Iteration ", i+1, "/", len(models))

        # Getting Predictions

        relevant[targets[i]] = models[i].predict(relevant).round()

    if level_col == 'Station':
        relevant = relevant.merge(test[['Station', 'area']].value_counts(
        ).reset_index().drop(0, axis=1), on=['Station', 'area'], how='left')
    elif level_col == 'area':
        relevant = relevant.merge(test[['area']].value_counts(
        ).reset_index().drop(0, axis=1), on=['area'], how='left')
    return relevant


def mae(y_true, y_pred, N):
    """
    Mean absolute error

    inputs:
        y_true (NumpyArray): ground truth
        y_pred (NumpyArray): prediction
        N (int): Number of samples
    """
    return round(sum(abs(y_true-y_pred))/N, 1)


def sae(y_true, y_pred):
    """
    Sum of absolute errors

    inputs:
        y_true (NumpyArray): ground truth
        y_pred (NumpyArray): prediction
    """
    return round(np.sum(abs(y_true-y_pred)), 3)
