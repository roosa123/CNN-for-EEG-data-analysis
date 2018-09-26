# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:30:35 2017

@author: Bartek
"""
# from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
import scipy.stats as stats
import os
import pandas as pd
from numpy.core.multiarray import ndarray

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv, DataFrame
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from keras.utils import to_categorical
from keras import backend as K


# TODO:
# 1 zastąpić CSP FBCSP: najpierw porwnać z oryginalnym csp, potem dać domylne pasma...done
# 2 zastąpić SelectKBest na RFE...pass
# 3 dopisać więcej cech...ale jakich??
# 4 przykład działania dla kNN, SVM linear, SVN rbf, RandomTrees...done
# 5 usuwanie jakich podobnych predyktorw?...pass
# 6 normalizowanie cech wczeniej1!!...done


class Log:
    counter = 0

    def LogInc(self, object):
        fileName = "c:\\test%d.txt" % Log.counter
        pd.DataFrame(object).to_csv(fileName)
        Log.counter = Log.counter + 1

    def Log(self, object):
        fileName = "c:\\test24.txt"
        pd.DataFrame(object).to_csv(fileName)


class CSP:
    def __init__(self):
        self.cspmat = []

    def transform(self, signal):
        return signal.transpose().dot(self.cspmat[0, :, :])

    def fit(self, class1, class2):
        self.val, self.cspmat = np.linalg.eig((np.cov(class1), np.cov(class1) + np.cov(class2)))


class FBCSP:
    def __init__(self, noBands):
        self.cspmat = []
        self.noBands = noBands

    def transform(self, signal):
        cspsig = np.zeros((signal.shape[1], signal.shape[0] * self.noBands))
        # print(np.shape(cspsig))
        for fid in range(0, self.noBands):
            # print("****")
            # print(fid)
            # print(fid*signal.shape[0])
            ##print(fid*signal.shape[0]+signal.shape[0])
            # print(np.shape(signal[:,:,fid].transpose()))
            # print(np.shape(self.cspmat[:,:,fid]))
            temp = signal[:, :, fid].transpose().dot(self.cspmat[:, :, fid])
            # print(np.shape(temp))
            cspsig[:, range(fid * signal.shape[0], fid * signal.shape[0] + signal.shape[0])] = temp
        return cspsig

    def fit(self, class1, class2):
        self.cspmat = np.zeros((signal.shape[1], signal.shape[1], self.noBands))
        for fid in range(0, self.noBands):
            C1 = class1[:, :, fid]
            C2 = class2[:, :, fid]
            # print(np.shape(self.cspmat))
            # print(fid)
            self.val, temp = np.linalg.eig((np.cov(C1), np.cov(C1) + np.cov(C2)))
            # print(np.shape(temp))
            self.cspmat[:, :, fid] = temp[0, :, :]


class featureExtractor:
    def __init__(self, name):
        self.name = name

    def transform(self, signal, labels1, labels2, labels3):
        noEvents = int(labels1.max())
        noRepetitedSignals = int(labels3.max()) + 1

        labels = np.zeros((2 * noEvents * noRepetitedSignals, 1))
        features = np.zeros((2 * noEvents * noRepetitedSignals, signal.shape[1]))
        if self.name == "all":
            features = np.zeros((2 * noEvents * noRepetitedSignals, 2 * signal.shape[1]))
        # Log.Log(signal)
        for tid in range(0, noEvents):
            for rid in range(0, noRepetitedSignals):
                # tid+1
                if self.name == "logvar":
                    features[rid + tid * noRepetitedSignals, :] = np.log(
                        np.var(signal[((labels1 == tid + 1) & (labels3 == rid)), :], axis=0))
                    # Log.Log(features[:, range(0, 4)])
                    features[rid + tid * noRepetitedSignals + noEvents * noRepetitedSignals, :] = np.log(
                        np.var(signal[((labels2 == tid + 1) & (labels3 == rid)), :], axis=0))
                    # Log.Log(features[:, range(0, 4)])
                    dataA = signal[((labels1 == tid + 1) & (labels3 == rid)), :]
                    # Log.LogInc(dataA)
                    dataA = signal[((labels2 == tid + 1) & (labels3 == rid)), :]
                    # Log.LogInc(dataA)

                if self.name == "pearsonr":
                    features[tid, :] = stats.pearsonr(signal[labels1 == tid + 1, :], signal[labels1 == tid + 1, :])
                    features[tid + noEvents, :] = stats.pearsonr(signal[labels2 == tid + 1, :],
                                                                 signal[labels2 == tid + 1, :])
                if self.name == "lyapunov":
                    features[tid, :] = 0  # (signal[labels1==tid+1,:],axis=0)
                    features[tid + noEvents, :] = 0  # np.log(np.var(signal[labels2==tid+1,:],axis=0))
                if self.name == "all":
                    # wyznacz wszystkie i złącz...kolumnami w jedną duż cechę
                    a1 = np.log(np.var(signal[labels1 == tid + 1, :], axis=0))
                    b1 = np.log(np.var(signal[labels2 == tid + 1, :], axis=0))
                    a2 = stats.entropy(signal[labels1 == tid + 1, :])
                    b2 = stats.entropy(signal[labels2 == tid + 1, :])

                    features[tid, :] = np.hstack((a1, a2))
                    features[tid + noEvents * noRepetitedSignals, :] = np.hstack((b1, b2))

                labels[tid + rid * noEvents] = 0;
                labels[tid + rid * noEvents + noEvents * noRepetitedSignals] = 1;
        # Log.Log(features)
        # Log.Log(labels)
        return features, labels


def build_cnn(kernel_size=(3,), dropout_rate=0.75, n=2688):
    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=kernel_size, activation='relu', input_shape=(n,1)))
    # model.add(Conv1D(filters=32, kernel_size=kernel_size, activation='relu'))

    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=(2,)))

    # model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    model.add(Conv1D(filters=16, kernel_size=kernel_size, strides=2, activation='relu'))

    model.add(Dropout(dropout_rate))
    # model.add(MaxPooling1D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    return model


def augment_data(x, y, shape_y):
    x_len = len(x)

    aug_x = x
    aug_y = y

    for i in range(10 * x_len):
        noise = np.random.normal(0.0, 1.0, shape_y)
        aug = x[i % x_len] * noise
        aug = np.expand_dims(aug, axis=0)

        label = np.expand_dims(y[i % x_len], axis=0)

        aug_x = np.append(aug_x, aug, axis=0)
        aug_y = np.append(aug_y, label, axis=0)

    return (aug_x, aug_y)


if __name__ == '__main__':
    plt.close("all")

    # USTAWIENIA:
    featureType = "logvar"  # typ cechy do wyboru: logvar, pearsonr, lyapunov, all
    classifierID = "CNN"  # zdefiniowane przez użytkownika
    f_bands = np.array([[1, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [32, 36], [36, 40], [8, 30]])
    # PARAMETRY PRZESZUKIWANIA
    maxfFeatures = 9  # max liczba cech do wybrania...można wszystkie sprawdzić, ale za mało przykładow i nie wyjdzie
    if not os.path.exists(("%s\%s") % (classifierID, featureType)):
        os.makedirs(("%s\%s") % (classifierID, featureType))
    # WŁACIWY PROGRAM
    noSubjects = 11
    finalAcc = []

    val1 = 1
    val2 = 0
    val3 = 0.05

    # for subject_id in range(1, noSubjects + 1):
    for subject_id in range(9 * 20 * 19):
        # filename = "subjectNoiseRefVsWrk%d.npy" % (subject_id)  # subjectNoise_1_repNo_0_amp_0.050000.csv
        # filename = "subjectNoise_1_repNo_0_amp_0.050000.npy"
        
        filename = "subjectNoise_%d_repNo_%d_amp_%f.npy" % (val1, val2, val3)
        
        if val1 >= 10:
            break
        
        if val3 >= 0.95:
            val3 = 0.05
            val2 += 1

            if val2 >= 20:
                val2 = 0
                val1 += 1
        else:
            val3 += 0.05

        # data = read_csv(filename, header=0)
        try:
            data = pd.DataFrame(np.load(filename))
        except FileNotFoundError:
            print("File: %s not found. Skipping." % (filename))
            continue

        #  wydobądz co potrzeba
        signal = data.values[:, range(0, 14)]
        reflab = data.values[:, 14]
        wrklab = data.values[:, 15]
        sampleId = data.values[:, 16]

        noEvents = int(reflab.max())

        # podziel caly sygnal na probki (tylko dla sieci konwolucyjnej)
        train_conv = []
        labels_conv = []

        if classifierID == "CNN":
            for i in range(noEvents):
                sample1 = data.loc[(data[14] == i + 1) & (data[16] == int(sampleId.max())), range(0, 14)]    # kolumna 14 - czas referencyjny (klasa 1)
                sample2 = data.loc[(data[15] == i + 1) & (data[16] == int(sampleId.max())), range(0, 14)]    # kolumna 15 - rekcja na bodziec (klasa 2)

                sample1 = np.transpose(sample1.values)
                sample2 = np.transpose(sample2.values)

                (nx1, ny1) = np.shape(sample1)
                (nx2, ny2) = np.shape(sample2)

                train_conv.append(np.reshape(sample1, nx1 * ny1))
                train_conv.append(np.reshape(sample2, nx2 * ny2))

                labels_conv.append(0)
                labels_conv.append(1)

            train_conv = np.array(train_conv)
            labels_conv = np.array(labels_conv)

            ss = StandardScaler()
            train_conv = ss.fit_transform(train_conv)

            (x, y) = np.shape(train_conv)
            (train_conv, labels_conv) = augment_data(train_conv, labels_conv, y)

            train_conv = np.expand_dims(train_conv, axis=2)

            labels_conv_categorical = to_categorical(labels_conv, 2)

        # wybrać jeden z przykładow lub dopisać jakis własny
        if classifierID == "LDA":  # 1
            estimators = [('prep', StandardScaler()), ('fsel', SelectKBest()),
                          ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))]
            params = {'fsel__score_func': [mutual_info_classif], 'fsel__k': range(1, maxfFeatures)}
        if classifierID == "kNN":  # 2
            estimators = [('prep', StandardScaler()), ('fsel', SelectKBest()), ('clf', KNeighborsClassifier())]
            params = {'fsel__score_func': [mutual_info_classif], 'fsel__k': range(1, maxfFeatures),
                      'clf__n_neighbors': range(1, round(4 * (noEvents - 1) / 3) - 1),
                      'clf__weights': ('uniform', 'distance'),
                      'clf__metric': ['minkowski'], 'clf__p': range(1, 5)}
        if classifierID == "SVMlin":  # 3
            estimators = [('prep', StandardScaler()), ('fsel', SelectKBest()), ('clf',
                                                                                SVC(kernel='linear', cache_size=2000,
                                                                                    probability=False))]  # klasy są zbalansowane, nie trzeba nic tu korygować
            params = {'fsel__score_func': [mutual_info_classif], 'fsel__k': range(1, maxfFeatures),
                       # 'clf__C': np.logspace(0.001, 100, 20),
                      # 'clf__shrinking': [True, False], 'clf__tol': [1e-1, 1e-3, 1e-5],
                      'clf__C': np.logspace(0.001, 0.001, 1),
                      'clf__shrinking': [True], 'clf__tol': [1e-1],

                      }

        if classifierID == "SVMrbf":  # 4
            estimators = [('prep', StandardScaler()), ('fsel', SelectKBest()), ('clf',
                                                                                SVC(kernel='rbf', cache_size=2000,
                                                                                    probability=False))]  # klasy są zbalansowane, nie trzeba nic tu korygować
            params = {'fsel__score_func': [mutual_info_classif], 'fsel__k': range(1, maxfFeatures),
                      'clf__C': np.logspace(0.001, 100, 20), 'clf__gamma': np.logspace(0.001, 100, 20),
                      'clf__shrinking': [True, False], 'clf__tol': [1e-1, 1e-3, 1e-5],
                      }
        if classifierID == "RF":  # 5
            estimators = [('prep', StandardScaler()), ('fsel', SelectKBest()),
                          ('clf', RandomForestClassifier(max_depth=None, bootstrap=True))]
            params = {'fsel__score_func': [mutual_info_classif], 'fsel__k': range(1, maxfFeatures),
                      'clf__n_estimators': np.arange(1, 100, 5), 'clf__criterion': ['gini', 'entropy'],
                      'clf__max_features': np.arange(1, 140, 10), 'clf__min_samples_split': np.arange(1, 15, 3),
                      'clf__oob_score': [True, False]}

        if classifierID == "CNN":
            (x, y, z) = np.shape(train_conv)
            estimators = [('prep', None), ('fsel', None), ('clf', KerasClassifier(build_fn=build_cnn))]
            params = {'clf__kernel_size': [(4,)], 'clf__dropout_rate': [0.6], 'clf__n': [y],
                        'clf__batch_size': [6], 'clf__epochs': [6, 16],'clf__verbose': [1]}


        # filtruj sygnał: DOCELOWO BĘDZIE ZASZYTE W FBCSP
        SPS = 128
        noTaps = 466  # iir
        noBands = f_bands.shape[0]
        fsig = np.zeros((signal.shape[1], signal.shape[0], noBands))
        for fid in range(0, noBands):
            filt = dsp.firwin(noTaps, [f_bands[fid, 0], f_bands[fid, 1]], pass_zero=False, window=('kaiser', 5.6533),
                              nyq=64)
            fsig[:, :, fid] = dsp.filtfilt(filt, 1, signal.transpose())
        # WALIDACJA LEAVE-ONE-OUT (STRATIFIED)
        noRepetitedSignals = int(sampleId.max()) + 1
        acc2 = np.zeros((noEvents, noRepetitedSignals *2 ))
        dfSelFeat = DataFrame([]).transpose()
        dfBestParams = DataFrame([]).transpose()
        for val_id in range(0, noEvents):

            # podziel dane na trening i test
            test_id = np.array(list(range(val_id * noRepetitedSignals, val_id * noRepetitedSignals + noRepetitedSignals)))

            train_id = np.setdiff1d(range(0, noEvents * noRepetitedSignals), test_id)
            # wyznacz CSP (ale tylko na treningu)
            oCSP = FBCSP(noBands);
            oCSP.fit(fsig[:, (reflab > 0) & (reflab != test_id), :], fsig[:, (wrklab > 0) & (wrklab != test_id), :])
            cspsig = oCSP.transform(fsig)
            # print(np.shape(cspsig))
            # print("yo")

            # dla każdego triala wyznacz energie/ceche
            oFE = featureExtractor(featureType)
            features, labels = oFE.transform(cspsig, reflab, wrklab, sampleId)  # type: (ndarray, ndarray)

            # print(np.shape(features))
            # klasyfikuj - > jako pipeline
            indTrain = np.hstack((train_id, train_id + noEvents * noRepetitedSignals))
            indTest = np.hstack((test_id, test_id + noEvents * noRepetitedSignals))
            print(np.shape(indTrain))

            pipe = Pipeline(estimators)
            tuner = GridSearchCV(pipe, params,
                                 scoring=None, fit_params=None,
                                 n_jobs=1, iid=True, refit=True, cv=None,
                                 verbose=0, pre_dispatch='2*n_jobs',
                                 error_score=np.NaN, return_train_score=True)
            
            if classifierID == "CNN":
                tuner.fit(train_conv[indTrain, :], labels_conv_categorical[indTrain])

                acc1 = tuner.score(train_conv[indTrain, :], labels_conv_categorical[indTrain])
                acc2[val_id, :] = (tuner.predict(train_conv[indTest, :]) == labels_conv[indTest])
            else:
                tuner.fit(features[indTrain, :], labels[indTrain].ravel())
                acc1 = tuner.score(features[indTrain, :], labels[indTrain].ravel())
                acc2[val_id, :] = (tuner.predict(features[indTest, :]) == labels[indTest].ravel())

            # ZAPIS NAJLEPSZYCH USTAWIAN I WYBRANYCH CECH
            dfParams = DataFrame.from_dict(tuner.best_params_, orient="index")
            dfBestParams = dfBestParams.append(dfParams.transpose())

            if classifierID != "CNN":
                dfSelFeat = dfSelFeat.append(
                    DataFrame(tuner.best_estimator_.named_steps['fsel'].get_support(indices=True)).transpose())
                    # print(tuner.best_estimator_.named_steps['fsel'].get_support(indices=True))
                dfSelFeat.index = range(0, noEvents * noRepetitedSignals)
                dfSelFeat.to_csv(".\%s\%s\%s_sid_%d_%s_bestfeatures.csv" % (
                    classifierID, featureType, classifierID, subject_id, featureType))

            K.clear_session()

        dfBestParams.index = range(0, noEvents)# * noRepetitedSignals)
        dfBestParams.to_csv(
            ".\%s\%s\%s_sid_%d_%s_bestParams.csv" % (classifierID, featureType, classifierID, subject_id, featureType))

        # ZAPIS KOŃCOWEGO WYNIKU
        finalAcc.append(acc2.mean())
        print("Srednia skutecznosc dla badanego %d: %f" % (subject_id, finalAcc[subject_id - 1]))
        dfRES = DataFrame(np.array(finalAcc))
        dfRES.to_csv(".\%s\%s\%s_%s_final_acc.csv" % (classifierID, featureType, classifierID, featureType))
    print("Skutecznosc calkowita: %f" % (np.array(finalAcc).mean()))