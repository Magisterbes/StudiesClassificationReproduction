import pickle as pck
import pandas as pd

dir = '''M:/GD/PubMedSearch/PubMedSearch/Check_full_list/'''


def dump_to_file(val, filepath):
    with open(filepath, 'wb') as pickle_file:
        pck.dump(val, pickle_file)


def load_from_file(filepath):
    with open(filepath, 'rb') as pickle_file:
        data = pck.load(pickle_file)
    return data


def load_pandas(filepath):
    return pd.read_csv(filepath, sep=";")


def save_pandas(data, filepath):
    data.to_csv(filepath, sep=";", index=False)


def save_arr(data, filepath):
    pdd = pd.DataFrame(data)
    save_pandas(pdd, filepath)
