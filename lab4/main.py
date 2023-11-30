import glob
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sklearn
import sklearn.svm
import sklearn.tree

rng = np.random.default_rng()
Q_VALUES = (0.1, 0.3, 0.5, 0.7, 1)
P = 2

def create_pairs(img):
    return [[(ele, sub[i+1]) for i, ele in enumerate(sub[:-1])] for sub in img]

def sp(image):
    u = image[:, :-1].reshape((-1,))
    v = image[:, 1:].reshape((-1,))

    M, N = image.shape
    Xc = np.count_nonzero(
        np.array(u % 2 == 0).astype(int) & np.array(u < v).astype(int) | np.array(u % 2 == 1).astype(int) & np.array(
            u > v).astype(int))
    Zc = np.count_nonzero(u == v)
    Wc = np.count_nonzero(np.floor(u / 2) == np.floor(v / 2)) - Zc
    Vc = M * (N - 1) - (Xc + Zc + Wc)

    a = (Wc + Zc) / 2
    b = 2 * Xc - M * (N - 1)
    c = Vc + Wc - Xc

    D = b ** 2 - 4 * a * c
    if a > 0:
        p1 = (-b + sqrt(D)) / (2 * a)
        p2 = (-b - sqrt(D)) / (2 * a)
    else:
        p1 = -1
        p2 = -1

    p = np.real(min(p1, p2))
    return np.array([p])

def get_bit_plane(image, num_plane):
    cont_image = image % 2 ** num_plane
    if num_plane <= 1:
        return cont_image
    else:
        return (cont_image - get_bit_plane(cont_image, num_plane - 1)) // (
            2 ** (num_plane - 1)
        )

def embed_plus_minus_one(q, C, W):
    C = C.ravel()
    N = int(C.size * q)

    embed_mask = np.concatenate([np.ones(N), np.zeros(C.size - N)]).astype(bool)
    np.random.shuffle(embed_mask)

    C_P = get_bit_plane(C, P)
    embed_mask &= C_P != W
    sign_mask = np.random.random(C.size) < 0.5

    C[embed_mask & (C < 255) & ((C == 0) | sign_mask)] += 1 * (2 ** P - 1)
    C[embed_mask & (C > 0) & ((C == 255) | ~sign_mask)] -= 1 * (2 ** P - 1)

def feature_histogram_full(img):
    return np.bincount(img.flat, minlength=256)

def calc_pairs(h_e):
    Pth_bit = 1 << 0
    pair_low = np.arange(256) & ~Pth_bit
    pair_high = pair_low + Pth_bit
    h_t = (h_e[pair_low] + h_e[pair_high]) / 2

    pair_base = np.unique(pair_low)
    return h_e[pair_base], h_t[pair_base]

def feature_chi_square(img):
    h_e = feature_histogram_full(img)
    h_e, h_t = calc_pairs(h_e)
    h_t[h_t == 0] = 1
    chi = np.sum((h_e - h_t)**2 / h_t)
    return np.array([chi])

def calc_histogram_diff(CW):
    # he - эмпирическая гистограмма
    # ht - теоретическая гистограмма
    he = np.bincount(CW.flat, minlength=256)
    i1 = np.arange(256) // 2
    i2 = i1 + 1
    ht = (he[i1] + he[i2]) / 2

    i3 = np.arange(128) * 2
    diff = np.abs(he[i3] - ht[i3])
    return diff

def calc_features(q, dataset):
    K = len(dataset)

    X = []
    Y = []

    for i in range(K):
        img = dataset[i].copy()
        W = np.random.randint(0, 2, img.ravel().size)
        if i < (K // 2):
            embed_plus_minus_one(q, img, W)
            cls = 1  # есть встраивание
        else:
            cls = 0  # нет встраивания

        X.append(sp(img))
        Y.append(cls)

    return X, Y

if __name__ == '__main__':
    dataset = [plt.imread(file) for file in glob.glob("BOWS2/*.tif")][:300]
    K = len(dataset)

    accuracy = []
    f1_scores = []
    for q in Q_VALUES:
        X, Y = calc_features(q, dataset)
        X_cls1, Y_cls1 = X[: K // 2], Y[: K // 2]
        X_cls0, Y_cls0 = X[K // 2:], Y[K // 2:]
        X_cls1_train, X_cls1_test, y_cls1_train, y_cls1_test = sklearn.model_selection.train_test_split(
            X_cls1,
            Y_cls1,
            random_state=228
        )
        X_cls0_train, X_cls0_test, y_cls0_train, y_cls0_test = sklearn.model_selection.train_test_split(
            X_cls0,
            Y_cls0,
            random_state=228
        )

        clf = sklearn.svm.LinearSVC()
        X_train = np.concatenate([X_cls1_train, X_cls0_train])
        y_train = np.concatenate([y_cls1_train, y_cls0_train])

        X_test = np.concatenate([X_cls1_test, X_cls0_test])
        y_test =  np.concatenate([y_cls1_test, y_cls0_test])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy.append(sklearn.metrics.accuracy_score(y_pred, y_test))
        f1_scores.append(sklearn.metrics.f1_score(y_pred, y_test))
        print(f"q value: {q} accuracy: {accuracy[-1]}")
        print(f"q value: {q} f1_score: {f1_scores[-1]}")

    plt.plot(Q_VALUES, accuracy)
    plt.show()
    plt.plot(Q_VALUES, f1_scores)
    plt.show()