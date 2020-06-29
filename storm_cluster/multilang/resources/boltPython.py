import sklearn
from sklearn import preprocessing, tree
from sklearn.tree import DecisionTreeClassifier

import imblearn #solve imbalance
from imblearn.over_sampling import SMOTE
import numpy as np 
import pandas as pd
import random
from itertools import zip_longest

import storm

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
DATA_0214 = database_location + "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"µ

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

class BoltPython(storm.BasicBolt):

    def initialize(self, conf, context):
        self._conf = conf
        self._context = context
        storm.logInfo("Bolt starting...")
        
        #Read dataset
        df = None 
        df_next = None

        for i in datasets:
            if df is None:
                df = read_random(i,20000)
                df_next = df
            else:
                df_next = read_random(i,20000)
                df = pd.concat([df,df_next])

        df_next = pd.read_csv(DATA_0223,skiprows=range(1,1500),nrows=20000)
        df = pd.concat([df, df_next])

        df = df.drop(df.columns[2],axis=1) #drop timestamp

        labels = df.iloc[:,-1].values
        labels = relabel_minorities(labels)

        #Binary labels
        labels_to_binary = []
        for i in labels:
            labels_to_binary.append(dict_binary[i])

        unique_df = pd.DataFrame(data=labels[1:], columns=["Label"])

        y = labels_to_binary
        X = df.iloc[:,:-1]

        #Oversampling + decision tree        
        oversample = SMOTE(sampling_strategy=1)
        x_train, y_train = oversample.fit_resample(X,y)

        DT = DecisionTreeClassifier()
        DT.fit(np.array(x_train), np.array(y_train))

        self._DT = DT
        storm.logInfo("Bolt ready...")

    def process(self, tuple):

        #Read spout tuple
        network_line = tuple.values[0]
        storm.logInfo("Processing tuple: " + network_line)

        features = np.array(network_line.split(','))
        features = np.delete(features,[2,79],None)

        #Decision tree prediction
        prediction = self._DT.predict(features.reshape(1,-1))
        
        #Convert prediction to json serializable list
        storm.emit(prediction.tolist())

BoltPython().run()