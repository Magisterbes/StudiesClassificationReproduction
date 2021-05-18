import itertools
import math
from sklearn import feature_extraction
from tqdm import tqdm
import nltk
from nltk.stem.snowball import SnowballStemmer
import sys
import re
import numpy as np
import pandas as pd
from operator import itemgetter
import io_part as io
from collections import Counter
import flatten_json

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['abstract', 'methods', 'method', 'conclusion',
                 'conclusions', 'results', 'discussion', 'more', 'than', 'that', 'use', 'these', 'which', 'not', 'and', 'are', 'for', 'can', 'the', 'may', 'among', 'but', 'use', 'this', 'has', 'with', 'most', 'been', 'had', 'our', 'other', 'when', 'who', 'both', 'there', 'howev', 'their', 'could', 'each', 'those', 'through', 'would', 'have', 'should', 'some', 'such', 'will', 'while', 'they', 'then', 'make', 'where', 'how'])
stopwords.extend(['but', 'and', 'by'])

stemmerRu = SnowballStemmer("russian")
stemmerEn = SnowballStemmer("english")


#stopwords = nltk.corpus.stopwords.words('russian')


def tag_pos(tokens):
    return nltk.tag.pos_tag(tokens, lang='eng')


def stemm(word):

    if word.lower() in stopwords:
        return ''

    if re.search('[a-z]', word):
        return stemmerEn.stem(word)

    return str(word)


def just_stem_ignorestop(text):
    tokens = re.findall(u"[a-z]{2,}", text.lower())

    stems = []
    for word in tokens:
        if word.strip() == "":
            continue
        sword = stemm(word)
        stems.append(sword)

    return ' '.join(stems)


def just_stem(text):
    tokens = re.findall(u"[a-z#\+\-]{2,}", text.lower())

    stems = []
    for word in tokens:
        if word.lower() in stopwords or word.strip() == "":
            continue
        sword = stemm(word)
        if sword.lower() in stopwords:
            continue
        if len(sword) > 2:
            stems.append(sword)

    return list(stems)


def make_numbered_dictionary(data):
    filter_di = {}

    stemmed = data["texts"].apply(just_stem)
    joined = list(itertools.chain(*stemmed))
    counted = Counter(joined)
    counted = dict(filter(lambda t:  t[1] >= 10, counted.items()))

    words_list = list(counted.keys())

    rng = 4

    for (i, row) in tqdm(enumerate(stemmed)):
        words = list(filter(lambda t:  t in words_list, row))

        for (j, word) in enumerate(words):
            if word not in filter_di:
                filter_di[word] = np.zeros(shape=[len(words_list), 1])

            for k in range(0, 2*rng+1):
                if j+k-rng < len(words) and j+k-rng > 0:
                    idx = words_list.index(words[j+k-rng])
                    filter_di[word][idx] += 1

    return (len(words_list), filter_di, words_list)


def get_vector_by_dic(Di, Row, vecsize):
    vec = np.zeros([vecsize, 1])

    for (i, w) in enumerate(Row):
        if w in Di.keys():
            vec += Di[w]/np.sum(Di[w])

    return list(vec/np.sum(vec))


def unite_cells(row):

    # return "{0} {1} {2}".format(row["title"], row["abstract"], row["pubmed_keywords"])
    return "{0} {1}".format(row["title"], row["abstract"], row["authors"])


def vectorize(data):

    data['texts'] = data.apply(unite_cells, axis=1)

    (vecsize, di, names) = make_numbered_dictionary(data)

    io.dump_to_file(di, io.dir+"VocForModel.dmp")

    y = []
    x = []
    texts = []
    for index, row in data.iterrows():
        x.append(get_vector_by_dic(
            di, just_stem(row['texts']), vecsize))
        texts.append(row['texts'].lower())

        if row["mark"] == 1:
            y.append(1)
        else:
            y.append(0)

    x = pd.DataFrame(x, columns=names)
    x["mark"] = y
    x["texts"] = texts
    return (di, x)
