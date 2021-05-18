import catboost
import pandas as pd
import numpy as np
import pickle as pck
from sklearn import metrics
from catboost import Pool, CatBoost
from sklearn.model_selection import train_test_split
import nlp_part_shadow as nlp
import studies_checks as test
import io_part as io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")


def Catboost(data):

    train, test = train_test_split(
        data, test_size=0.3, random_state=22)

    train_pool = Pool(train.drop(columns=["texts", "mark"]), train['mark'])
    test_pool = Pool(test.drop(
        columns=["texts", "mark"]), test['mark'])

    params = {'iterations': 300,
              'depth': 5,
              'loss_function': 'Logloss',
              'verbose': True,
              'use_best_model': True}

    clf = CatBoost(params)
    clf.fit(train_pool, eval_set=test_pool)

    clf.save_model("model_update.mdl")

    return (clf, test.drop(
        columns=["texts", "mark"]), test['mark'], test['texts'].values)


def RF(data):

    train, test = train_test_split(
        data, test_size=0.3, random_state=32)

    pipeline = Pipeline([('clf', SGDClassifier(loss="modified_huber"))])
    # pipeline = Pipeline([('clf', RandomForestClassifier())])
    parameters = {
        # 'clf__n_estimators': ([1000]),
        'clf__epsilon': ([0.01]),
        'clf__penalty': (['l2']),
        'clf__max_iter': ([5000]),

    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=2)

    grid_search.fit(list(train['vector'].values),  train['mark'])

    res = []

    return (grid_search, list(test['vector'].values), test['mark'], test['text'].values)


def Catboost_blind(data):

    train_pool = Pool(data.drop(columns=["texts", "mark"]), data['mark'])

    params = {'iterations': 200,
              'depth': 5,
              'loss_function': 'Logloss',
              'verbose': True}

    clf = CatBoost(params)
    clf.fit(train_pool)

    clf.save_model("model_update_blind.mdl")

    return (clf)


if __name__ == '__main__':

    #data = io.load_pandas("training_data.csv")
    #(di, data) = nlp.vectorize(data)

    #(model, x, y, text) = Catboost(data)
    # (model, x, y, text) = RF(data)

    #test.test_model(model, x, y, text)
    #test.importance(model, x, y, text)

    from_file = CatBoost()
    from_file = from_file.load_model(fname="model_update.mdl")

    test.test_selected(from_file)
