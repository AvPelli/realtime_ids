# Master's thesis 2020

## Introduction

In today's networking landscape, parties rely on internal networks to exchange confidential information. Searching efficient and effective ways to protect data over the network is subject to intense research focus. 
One way to detect suspicious activity is using a network intrusion detection system (NIDS). This system needs to be reliable in the face of (un)known attacks. Furthermore, intrusions are ideally detected sooner rather than later to mitigate potential damage. 
In search to satisfy these conditions, this thesis investigated the use of several machine learning models on the novel cse-cic-ids2018 dataset, as well as the incorporation of one of those models in a stream processing system using the Apache Storm framework.


## General

The models are trained by using the cse-cic-ids2018 dataset, these csv files are not included in this repository because of their size (Gigabytes).

These csv files are used in the following python files:

* ML_modellen.py 
* storm_cluster/multilang/resources/spoutPython.py
* storm_cluster/multilang/resources/boltPython.py

In these files the paths to the csv files have to be adjusted based on their location.


The python file "ML_modellen" contains all code for the various machine learning models. The dependencies are listed in requirements.txt

## Dependencies

The cluster uses the following software:

| Dependency | version  |   
| ------- | --- |
| [Apache Kafka](https://kafka.apache.org/downloads) | 2.5.0 |
| [Apache Storm](https://storm.apache.org/2019/10/31/storm210-released.html) | 2.1.0 |
| [Java](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html)  | 8 |

The cluster does not work with newer java versions, its important to verify if java jdk 8 is used.

## Apache storm 

After installation of [Apache Storm](https://storm.apache.org/2019/10/31/storm210-released.html) the framework's configuration resides in .../apache-storm-2.1.0/conf, the standard settings for all possible settings are located in defaults.yml. 

The same directory contains storm.yaml, where the location of nimbus (localhost) has to be specified. The default settings from default.yml can be overridden in this file. The settings used for this thesis are to be found in the storm.yaml file in this github repository. 


## Cluster

The /storm_cluster directory contains the eclipse project with source code. To start the cluster, the code has to be compiled to a jar file. 

The following commands are used in this order to run the cluster locally:
```
zookeeper-server-start .../zookeeper.properties
storm nimbus (in console met admin rechten)
storm supervisor (in console met admin rechten)
storm ui
```

When this is done, the topology can be run in a new console by using the following command. Replace <cluster_jar_name> with the name of the compiled jar:

```
cd {project pad}/storm_cluster 
storm jar target/<cluster_jar_name>.jar org.apache.storm.flux.Flux --local topology.yaml
```

The statistics of the cluster are obtained on the storm UI at localhost:8080

The logs are available in .../apache-storm-2.1.0/logs

## Implementation & configuration of the topology

The code of the spout and bolt components resides in the storm_cluster/multilang/resources directory. 
The topology of the stream processing network can be changed in the topology.yml file. 




