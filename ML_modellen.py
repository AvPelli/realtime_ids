import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical, np_utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import optimizers

import sklearn as sk
from sklearn import preprocessing, preprocessing, tree
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, make_scorer, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

import imblearn #solve imbalance
from imblearn.over_sampling import SMOTE
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import random
from itertools import zip_longest
from time import time


############### Constanten ###############

#Verander pad
database_location = "C:/Users/Arthu/Desktop/thesis_realtime_ids/csecicids2018-clean/"

DATA_0302 = database_location + "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
DATA_0301 = database_location + "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
DATA_0228 = database_location + "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0223 = database_location + "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0222 = database_location + "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0221 = database_location + "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0220 = database_location + "Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0216 = database_location + "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0215 = database_location + "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0214 = database_location + "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"

datasets = [DATA_0302, DATA_0301, DATA_0228, DATA_0222, DATA_0221, DATA_0220, DATA_0216, DATA_0215, DATA_0214]
dataset_names = ["DATA_0302", "DATA_0301", "DATA_0228", "DATA_0223", "DATA_0222", "DATA_0221", "DATA_0220", "DATA_0216",
 "DATA_0215", "DATA_0214"]

 
#Herlabellen aanvals categoriëen (15) tot 6 klassen
dict_category = {
    "Benign"                : "Benign",
    "Bot"                   : "Bot", #02/03 
    "Infilteration"         : "Infilteration", #28/02,01/03
    "SQL Injection"         : "SQL Injection", #22/02,23/02

    "Brute Force -Web"      : "Brute Force", #22/02,23/02
    "Brute Force -XSS"      : "Brute Force", #22/02,23/02
    "FTP-BruteForce"        : "Brute Force", #14/02 
    "SSH-Bruteforce"        : "Brute Force",  #14/02

    "DDOS attack-LOIC-UDP"  : "DOS",#20/02,21/02 
    "DDOS attack-HOIC"      : "DOS",  #21/02 
    "DDoS attacks-LOIC-HTTP" : "DOS", #20/02 
    "DoS attacks-SlowHTTPTest" : "DOS", #16/02 
    "DoS attacks-Hulk"      : "DOS",    #16/02 
    "DoS attacks-Slowloris" : "DOS", #15/02
    "DoS attacks-GoldenEye" : "DOS" #15/02 
}

#Kies de gewenste target klasse
dict_binary = {
    "Benign"                : 0, 
    "Bot"                   : 1, 
    "Infilteration"         : 0, 
    "SQL Injection"         : 0, 
    "Brute Force"           : 0,
    "DOS"                   : 0
}


############### Hulp Functies ###############

def read_random(filename,sample_size):
    if sample_size is None:
        df = pd.read_csv(filename)
    else:
        n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
        skip = sorted(random.sample(range(1,n+1),n-sample_size)) #the 0-indexed header will not be included in the skip list
        df = pd.read_csv(filename, skiprows=skip)
    return df

def draw_histogram_attacks(dataframe):
    count = dataframe.groupby(['Label'])['Label'].count().sort_values(ascending=False) 
    attacks = count.keys().tolist()
    amount = count.values.tolist()
    
    y_pos = np.arange(len(attacks))
    plt.bar(y_pos, amount, align='center', alpha=0.5)
    plt.xticks(y_pos, attacks, rotation=20, horizontalalignment='right')
    plt.title('Attacks')
    plt.tight_layout()
    plt.show()

def relabel_minorities(labels):
    relabelled = []
    for i in labels:
        relabelled.append(dict_category[i]) 
    
    #Numpy array
    return np.array(relabelled)

def encode_to_binary_classification(y_train,y_test):
    #Encode output labels
    y_train_encoded = []
    y_test_encoded = []
    for i,j in zip_longest(y_train,y_test, fillvalue="end"):
        if i != "end":
            y_train_encoded.append(dict_binary[i])
        if j != "end":
            y_test_encoded.append(dict_binary[j])
    return (y_train_encoded,y_test_encoded)
    




############### Implementatie ###############
print(tf.executing_eagerly())

df = None #dataframe met samples van alle dagen
df_next = None #dataframe met samples van 1 dag
df_array = [] #tuples (dag,dataframe)

for i in datasets:
    if df is None:
        df = read_random(i,20000)
        df_next = df
    else:
        df_next = read_random(i,20000)
        df = pd.concat([df,df_next])

df_next = pd.read_csv(DATA_0223,skiprows=range(1,1500),nrows=20000)
df = pd.concat([df, df_next])

#Drop timestamp
df = df.drop(df.columns[2],axis=1)

print(df.head())

labels = df.iloc[:,-1].values
labels = relabel_minorities(labels)

#Omzetten naar binary classification
labels_to_binary = []
for i in labels:
    labels_to_binary.append(dict_binary[i])

unique, counts = np.unique(labels_to_binary, return_counts=True)
print("before oversampling:")
print(dict(zip(unique,counts)))

unique_df = pd.DataFrame(data=labels[1:], columns=["Label"])

y = labels_to_binary
X = df.iloc[:,:-1]

#k-fold crossvalidation met k = 4 => 25% train, 75% test
kfold = StratifiedKFold(n_splits=4, shuffle=False, random_state=None)

accuracies = []
f_scores = []
precisions = []
recalls = []
time_array = []

#Decision tree
def decision_tree(X_train,Y_train,bool_oversample):
    #Split in K folds
    for train_index,test_index in kfold.split(X_train, Y_train):
        x_train,x_test = X_train[train_index],X_train[test_index]
        y_train,y_test = Y_train[train_index],Y_train[test_index]
        
        if bool_oversample:
            #Oversample 
            oversample = SMOTE(sampling_strategy=1)
            x_train, y_train = oversample.fit_resample(x_train,y_train)

            unique, counts = np.unique(y_train, return_counts=True)
            print("after oversampling:")
            print(dict(zip(unique,counts)))

        #Decision tree
        DT = DecisionTreeClassifier()

        #Train network
        t0 = time()
        DT.fit(np.array(x_train), np.array(y_train))
        t1 = time()
        
        #Metrics
        y_pred = DT.predict(x_test)
        accuracies.append(DT.score(x_test,y_test))
        f_scores.append(f1_score(y_test,y_pred))
        precisions.append(precision_score(y_test,y_pred))
        recalls.append(recall_score(y_test,y_pred))
        time_array.append(t1-t0)

#K nearest neighbors
def KNN(X_train,Y_train,time_array,k,distance_power,bool_oversample):

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)

    #Split in K folds
    for train_index,test_index in kfold.split(X_scaled, Y_train):
        x_train,x_test = X_scaled[train_index],X_scaled[test_index]
        y_train,y_test = Y_train[train_index],Y_train[test_index]
        
        if bool_oversample:
            #Oversample 
            oversample = SMOTE(sampling_strategy=1)
            x_train, y_train = oversample.fit_resample(x_train,y_train)

            unique, counts = np.unique(y_train, return_counts=True)
            print("after oversampling:")
            print(dict(zip(unique,counts)))

        KNN = KNeighborsClassifier(n_neighbors=k, p=distance_power, n_jobs=-1)

        t0 = time()
        KNN.fit(np.array(x_train),np.array(y_train))
        t1 = time()

        #Metrics
        y_pred = KNN.predict(x_test)
        accuracies.append(KNN.score(x_test,y_test))
        f_scores.append(f1_score(y_test,y_pred))
        precisions.append(precision_score(y_test,y_pred))
        recalls.append(recall_score(y_test,y_pred))
        time_array.append(t1-t0)

#SVM
def SVM_rbf(X_train,Y_train,time_array, bool_oversample):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)

    #Split in K folds
    for train_index,test_index in kfold.split(X_scaled, Y_train):
        x_train,x_test = X_scaled[train_index],X_scaled[test_index]
        y_train,y_test = Y_train[train_index],Y_train[test_index]
        
        if bool_oversample:
            #Oversample 
            oversample = SMOTE(sampling_strategy=1)
            x_train, y_train = oversample.fit_resample(x_train,y_train)

            unique, counts = np.unique(y_train, return_counts=True)
            print("after oversampling:")
            print(dict(zip(unique,counts)))

        SVM = SVC() #Use this when using rbf kernel

        t0 = time()
        print("fitting")
        SVM.fit(x_train,y_train)
        t1 = time()
        
        #Metrics
        y_pred = SVM.predict(x_test)
        accuracies.append(SVM.score(x_test,y_test))
        f_scores.append(f1_score(y_test,y_pred))
        precisions.append(precision_score(y_test,y_pred))
        recalls.append(recall_score(y_test,y_pred))
        time_array.append(t1-t0)

#SVM
def SVM_linear(X_train,Y_train,time_array, bool_oversample):
    #Misschien standardscaler nodig
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)

    #Split in K folds
    for train_index,test_index in kfold.split(X_scaled, Y_train):
        x_train,x_test = X_scaled[train_index],X_scaled[test_index]
        y_train,y_test = Y_train[train_index],Y_train[test_index]
        
        if bool_oversample:
            #Oversample 
            oversample = SMOTE(sampling_strategy=1)
            x_train, y_train = oversample.fit_resample(x_train,y_train)

            unique, counts = np.unique(y_train, return_counts=True)
            print("after oversampling:")
            print(dict(zip(unique,counts)))

        #Two options: linear or rbf
        SVM = LinearSVC(dual=False) #Use this when using linear kernel

        t0 = time()
        print("fitting")
        SVM.fit(x_train,y_train)
        t1 = time()
        
        #Metrics
        y_pred = SVM.predict(x_test)
        accuracies.append(SVM.score(x_test,y_test))
        f_scores.append(f1_score(y_test,y_pred))
        precisions.append(precision_score(y_test,y_pred))
        recalls.append(recall_score(y_test,y_pred))
        time_array.append(t1-t0)


def create_MLP(X_train,Y_train):
    dropout_rate = 0.2
    regularize_rate = 0.01

    #Create neural network
    model = Sequential()
    model.add(Dense(9, input_dim=np.array(X_train).shape[1], activation='relu', kernel_regularizer=regularizers.l2(l=regularize_rate)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(l=regularize_rate)))
    model.add(Dropout(dropout_rate))

    #Output node
    model.add(Dense(1, activation='sigmoid'))
    
    #Set learning rate 
    opt_adam = optimizers.Adam(learning_rate=0.00001)

    model.compile(loss='binary_crossentropy',optimizer=opt_adam,metrics=['accuracy',f1_m, precision_m, recall_m])
    
    model.summary()
    return model

def MLP(X_train,Y_train,time_array,bool_oversample):
    
    #Standardize training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)

    #Split in K folds
    for train_index,test_index in kfold.split(X_scaled, Y_train):
        x_train,x_test=X_scaled[train_index],X_scaled[test_index]
        y_train,y_test=Y_train[train_index],Y_train[test_index]
        
        if bool_oversample:
            #Oversample 
            oversample = SMOTE(sampling_strategy=1)
            x_train, y_train = oversample.fit_resample(x_train,y_train)

            unique, counts = np.unique(y_train, return_counts=True)
            print("after oversampling:")
            print(dict(zip(unique,counts)))
        
        #Create network for each fold
        model=create_MLP(x_train,y_train)

        #Train network
        t0 = time()
        model.fit(np.array(x_train), np.array(y_train),epochs=20,verbose=0)
        t1 = time()

        loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
        
        #Metrics
        accuracies.append(accuracy)
        f_scores.append(f1_score)
        precisions.append(precision)
        recalls.append(recall)
        time_array.append(t1-t0)

def calculate_metrics(time_array,accuracies,f_scores,precisions,recalls):
    avg_time = sum(time_array)/len(time_array)
    avg_acc = sum(accuracies)/len(accuracies)
    avg_f = sum(f_scores)/len(f_scores)
    avg_prec = sum(precisions)/len(precisions)
    avg_rec = sum(recalls)/len(recalls)

    print("Avg time: " +str(avg_time))
    print("Avg accuracy: " + str(avg_acc))
    print("Avg f score: " + str(avg_f))
    print("Avg precision:" + str(avg_prec))
    print("Avg recall: " + str(avg_rec))

    accuracies = []
    f_scores = []
    precisions = []
    recalls = []
    time_array = []


#Algorithm parameters: (training_set, training_labels, [algorithm specific args...], boolean_oversampling)

#decision tree
decision_tree(np.array(X),np.array(y),True)
print("Decision tree:")
calculate_metrics(time_array,accuracies,f_scores,precisions,recalls)

#KNN
KNN = KNN(np.array(X),np.array(y), time_array, 5, 2, True)
print("KNN")
calculate_metrics(time_array,accuracies,f_scores,precisions,recalls)

#MLP
MLP(np.array(X),np.array(y),time_array,True)
print("MLP")
calculate_metrics(time_array,accuracies,f_scores,precisions,recalls)

#SVM
SVM_rbf(np.array(X),np.array(y), time_array, True)
print("SVM_rbf:")
calculate_metrics(time_array,accuracies,f_scores,precisions,recalls)

SVM_linear(np.array(X),np.array(y), time_array, True)
print("SVM_linear")
calculate_metrics(time_array,accuracies,f_scores,precisions,recalls)

