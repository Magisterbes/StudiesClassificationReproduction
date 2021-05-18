from operator import itemgetter
from numpy.lib.shape_base import apply_along_axis
import pandas as pd
import numpy as np
import pickle as pck
from catboost import Pool
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import io_part as io
import nlp_part_shadow as nlp
from tqdm import tqdm
import Levenshtein
import re

exclude = io.load_pandas("exclude.csv")
whitelist = set('abcdefghijklmnopqrstuvwxyz ')


def find_reason(imp, testx, testy, preds_proba):
    res = []
    imp = imp.drop(columns=["real_class", "pred"])
    cols = imp.columns
    dci = {}
    for (i, row) in enumerate(imp.values):
        temp = dict(pd.DataFrame(row, cols))[0]
        srt = sorted(temp.items(), key=itemgetter(1), reverse=True)
        srt_best = sorted(srt[:3], key=itemgetter(0), reverse=True)
        chosen = tuple(map(lambda t: t[0], srt_best[:3]))
        values = list(map(lambda t: np.round(t[1], 2), srt_best[:3]))

        bad_chosen = tuple(map(lambda t: t[0], srt[-3:]))
        bad_values = list(map(lambda t: np.round(t[1], 2), srt[-3:]))

        res.append([chosen, values, np.round(
            np.sum(values), 2), bad_chosen, bad_values,  testy.values[i]])

    pdd = pd.DataFrame(res, columns=[
        "best", "imp values", "sum imp", "worst", "words imp", "class"])

    pdd.to_csv("pdd.csv", sep=";", index=False)


def importance(model, x, y, txt):

    test_pool = Pool(x, y)

    preds_proba = model.predict(test_pool, prediction_type='Probability')[:, 1]

    cols = list(x.columns)
    cols.append("baseline")

    imp = model.get_feature_importance(type='ShapValues', data=test_pool)
    df = pd.DataFrame(imp, columns=cols)

    df["real_class"] = y
    df["pred"] = preds_proba

    df.to_csv("full_imp.csv", sep=";", index=False)

    find_reason(df, x, y, preds_proba)


def exclude_verdict(text, old_verdict):
    summ = 0
    for (j, w) in enumerate(exclude['words'].values):
        if w in text:
            summ += exclude['weight'].values[j]

    if summ > 2:
        return 0

    return old_verdict


def test_model(model, x, y, txt):
    # classify test data
    test_pool = Pool(x, y)

    preds_proba = model.predict(test_pool, prediction_type='Probability')[:, 1]
    # preds_proba = model.predict_proba(x)[:, 1]

    for (i, pred) in enumerate(preds_proba):
        count = []
        summ = 0

        for (j, w) in enumerate(exclude['words'].values):
            if w in txt[i]:
                count.append(w)
                summ += exclude['weight'].values[j]

        if summ > 2:
            preds_proba[i] = 0

        if preds_proba[i] >= 0.3 and y.values[i] == 0:
            print('FP: ' + txt[i][:100])

        if preds_proba[i] <= 0.2 and y.values[i] == 1:
            print('FN: ' + txt[i][:100])
    # AUC ROC
    acc = roc_auc_score(y, preds_proba)

    # F1
    f1 = metrics.f1_score(y, np.round(preds_proba))

    # ROC curve saved to file
    fpr, tpr, thresholds = metrics.roc_curve(y, preds_proba, pos_label=1)
    fr2 = pd.DataFrame(
        list(zip(*[fpr, tpr, thresholds])), columns=["fpr", "tpr", "thresholds"])
    fr2.to_csv("roc.csv", sep=";", index=False)

    # Build classifier efficiency table and save to file
    df = []
    for i in range(20):
        preds = (preds_proba > float(i+1)/20)*1

        bots = (np.sum((y == 1)*1))
        not_bots = (np.sum((y == 0)*1))
        tp = np.sum((preds*y == 1)*1)
        fp = np.sum((preds == 1)*(y == 0))
        fn = np.sum((preds == 0)*(y == 1))
        tn = np.sum((preds == 0)*(y == 0))

        f1 = metrics.f1_score(y, preds)

        tpr = tp/bots
        fpr = fp/not_bots

        df.append([np.round(float(i+1)/20, 2), bots,
                   not_bots, tp, fp, tn, fn, tpr, fpr, f1])
    df = pd.DataFrame(df, columns=[
        "Threshold", "In class", "Not in class", "TP", "FP", "TN", "FN", "TPR", "FPR", "F1"])
    df.to_csv("table.csv", sep=";")

    # Print basic results
    print(str([acc, f1]))

# Train model on 90% of data and use 10% as test.


def title_change(title):

    return ''.join(filter(whitelist.__contains__, title.lower()))


def test_selected(model):

    TPFPTNFN = [0, 0, 0, 0]

    full_data = io.load_pandas('full_record.csv')  # .head(100)
    full_data['title'] = full_data['title'].apply(lambda t: t.lower())
    full_data = full_data.drop_duplicates(subset="title")
    full_data.to_csv('full_record.csv', sep=";", index=False)

    selected_data = io.load_pandas('selected.csv')

    di = io.load_from_file(io.dir+"VocForModel.dmp")
    full_data['texts'] = full_data.apply(nlp.unite_cells, axis=1)

    x = []
    for index, row in tqdm(full_data.iterrows()):
        x.append(np.transpose(nlp.get_vector_by_dic(
            di, nlp.just_stem(row['texts']), len(di.keys())))[0])

    full_data["preds"] = model.predict(
        x, prediction_type='Probability')[:, 1]
    full_data["preds"] = (full_data["preds"] > 0.35)*1

    new_pred = []
    for findex, frow in full_data.iterrows():
        new_pred.append(exclude_verdict(frow["texts"], frow["preds"]))

    full_data["preds"] = new_pred

    names = list(selected_data['Name'].apply(title_change))
    full_data["title"] = list(full_data['title'].apply(title_change))
    wasfound = ""
    for findex, frow in tqdm(full_data.iterrows()):
        found = False

        if wasfound != "":
            names.remove(wasfound)
            wasfound = ""

        for name in names:
            dis = Levenshtein.distance(
                frow['title'].lower(), name)
            l = min([len(frow['title']), len(name)])

            if dis < 5 or float(dis)/float(l+1) < 0.05:
                found = True
                wasfound = name
                break

        if found and frow["preds"] == 1:
            TPFPTNFN[0] += 1

        if not found and frow["preds"] == 1:
            TPFPTNFN[1] += 1

        if not found and frow["preds"] == 0:
            TPFPTNFN[2] += 1

        if found and frow["preds"] == 0:
            TPFPTNFN[3] += 1

    print(TPFPTNFN)
    print("\n".join(names))
