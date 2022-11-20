import numpy as np
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import os
import jieba
import re

input_f = open("./Statmt-newstest_zhen-2022-zho-eng.zho", "r", encoding="utf-8")

lines = input_f.readlines()
docs_num = len(lines)
docs_words = []

for line in lines:
    clean_line = re.sub('[^\u4e00-\u9fa5]+', '', line)
    cut_line = jieba.cut(clean_line, cut_all=False)
    save_line = " ".join(cut_line).strip("\n").rstrip(" ").split(" ")
    docs_words.append(save_line)

vocab = set(itertools.chain(*docs_words))

v2i = {v: i for i, v in enumerate(vocab)}
i2v = {i: v for v, i in v2i.items()}


def safe_log(x):
    mask = x != 0
    x[mask] = np.log(x[mask])
    return x

def show_tfidf(tfidf, vocab, filename):
    plt.imshow(tfidf, cmap="YlGn", vmin=tfidf.min(), vmax=tfidf.max())
    plt.xticks(np.arange(20), vocab[0:20], fontsize=6, rotation=90)
    plt.yticks(np.arange(10), np.arange(1, tfidf.shape[0]+1)[:10], fontsize=6)
    plt.tight_layout()
    output_folder = './visual/'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, '%s.png') % filename, format="png", dpi=500)
    plt.show()

tf_methods = {
        "ori": lambda x: x,
        "log": lambda x: np.log(1+x),
        "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
        "boolean": lambda x: np.minimum(x, 1),
        "log_avg": lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True))),
    }
idf_methods = {
        "log": lambda x: 1 + np.log(docs_num / (x+1)),
        "prob": lambda x: np.maximum(0, np.log((docs_num - x) / (x+1))),
        "len_norm": lambda x: x / (np.sum(np.square(x))+1),
    }


def get_tf(method="log"):
    _tf = np.zeros((len(vocab), docs_num), dtype=np.float64)    # [n_vocab, n_doc]
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        for v in counter.keys():
            _tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]

    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    return weighted_tf(_tf)


def get_idf(method="log"):
    df = np.zeros((len(i2v), 1))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i, 0] = d_count

    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df)


def cosine_similarity(q, _tf_idf):
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity


def docs_score(q, len_norm=False):
    q_words = q.replace(",", "").split(" ")

    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i)-1] = v
            unknown_v += 1
    if unknown_v > 0:
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype=np.float)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=np.float)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float)     # [n_vocab, 1]
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]

    q_vec = q_tf * _idf            # [n_vocab, 1]
    q_scores = cosine_similarity(q_vec, _tf_idf)
    if len_norm:
        len_docs = [len(d) for d in docs_words]
        q_scores = q_scores / np.array(len_docs)
    return q_scores


def get_keywords(n=2):
    for c in range(3):
        col = tf_idf[:, c]
        idx = np.argsort(col)[-n:]
        print("doc{}, top{} keywords {}".format(c, n, [i2v[i] for i in idx]))

if not os.path.exists("./tf_idf_numpy.txt"):
    tf = get_tf("ori")  # [n_vocab, n_doc]
    idf = get_idf()  # [n_vocab, 1]
    tf_idf = tf * idf  # [n_vocab, n_doc]
    np.savetxt("./tf_numpy.txt", tf, fmt="%.10e", delimiter=",", encoding="utf-8")
    np.savetxt("./idf_numpy.txt", idf, fmt="%.10e", delimiter=",", encoding="utf-8")
    np.savetxt("./tf_idf_numpy.txt", tf_idf, fmt="%.10e", delimiter=",", encoding="utf-8")
else:
    tf = np.loadtxt("./tf_numpy.txt", dtype=np.float64, delimiter=",", encoding="utf-8")
    idf = np.loadtxt("./idf_numpy.txt", dtype=np.float64, delimiter=",", encoding="utf-8")
    tf_idf = np.loadtxt("./tf_idf_numpy.txt", dtype=np.float64, delimiter=",", encoding="utf-8")

print("tf shape(vecb in each docs): ", tf.shape)
print("\ntf samples:\n", tf[:2])
print("\nidf shape(vecb in all docs): ", idf.shape)
print("\nidf samples:\n", idf[:2])
print("\ntf_idf shape: ", tf_idf.shape)
print("\ntf_idf sample:\n", tf_idf[:2])


n_num = int(tf_idf.shape[1] / 5)
def cluster_by_tfidf(tfidf, c_num, centers=None):
    if centers is None:
        centers = tfidf[:c_num]
    tfidf_copy = tfidf
    for i in range(centers.shape[0]):
        neighbor = np.argsort(((tfidf_copy - centers[i]) ** 2).sum(axis=1), axis=0)
        if i != centers.shape[0] - 1:
            nearest = tfidf_copy[neighbor[:n_num]]
        else:
            nearest = tfidf_copy
        if i == 0:
            new_centers = np.mean(nearest, axis=0).reshape(1, nearest.shape[1])
        else:
            temp_center = np.mean(nearest, axis=0).reshape(1, nearest.shape[1])
            new_centers = np.concatenate((new_centers, temp_center), axis=0)
        tfidf_copy = tfidf_copy[neighbor[n_num + 1:]]
    distance = np.sum(((new_centers - centers) ** 2).reshape(1, -1)[0], axis=0)
    return new_centers, distance

threshold = 0.5
dis = 1
center = None
while dis > threshold:
    center, dis = cluster_by_tfidf(tf_idf.T, 5, center)
    print(dis)

class_list = []
tfidf_copy = tf_idf
flag = np.ones((docs_num))
for i in range(center.shape[0]):
    neighbor = np.argsort(((tf_idf.T - center[i]) ** 2).sum(axis=1), axis=0)
    if i != center.shape[0] - 1:
        copy_flag = flag[neighbor]
        copy_flag = np.cumsum(copy_flag, axis=0)
        max_num = np.argwhere(copy_flag == n_num)[0][0]
    else:
        max_num = docs_num
    real_n = neighbor[:max_num]
    copy_flag = flag[neighbor]
    real_n = real_n[np.array(copy_flag[:max_num], dtype=bool)]
    flag[real_n] = 0
    class_list.append(real_n)

output_f = open("./cluster_file.txt", "w", encoding="utf-8")

for doc_class in class_list:
    for idx in doc_class:
        output_f.write(" ".join(docs_words[idx]) + "\n")


input_f.close()
output_f.close()
