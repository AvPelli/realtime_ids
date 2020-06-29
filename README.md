# Algemeen

De modellen worden steeds opgezet door de cse-cic-ids2018 dataset te gebruiken, deze csv files staan er niet bij doordat ze te groot zijn om op github up te loaden. 

De python files die deze csv bestanden gebruiken zijn:

* ML_modellen.py 
* storm_cluster/multilang/resources/spoutPython.py
* storm_cluster/multilang/resources/boltPython.py

In deze files zal men de paden naar deze csv files moeten aanpassen.


# 1. machine learning

De python file "ML_modellen" bevat alle code voor de verschillende modellen op te stellen. De dependencies hiervoor zijn opgelijst in requirements.txt

# 2. realtime cluster

## Dependencies

Voor de cluster is volgende software nodig:

| Dependency | version  |   
| ------- | --- |
| [Apache Kafka](https://kafka.apache.org/downloads) | 2.5.0 |
| [Apache Storm](https://storm.apache.org/2019/10/31/storm210-released.html) | 2.1.0 |
| [Java](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html)  | 8 |

De cluster start niet op met nieuwere java versies, het is dus belangrijk om te verifiëren of versie 8 gebruikt wordt.

## Apache storm 

Na installatie van [Apache Storm](https://storm.apache.org/2019/10/31/storm210-released.html)  vindt men de configuratie van het framework zelf in .../apache-storm-2.1.0/conf, daar vindt men in defaults.yml de standaard waarden voor alle mogelijke instellingen. 

In diezelfde map zit storm.yaml, daar moet men de locatie van de nimbus opgeven (localhost) en kan men de instellingen vanuit default.yml overschrijven. De instellingen die gebruikt zijn tijdens de thesis kunt u vinden in storm.yaml hierboven in de github repository.


## Cluster

De map /storm_cluster bevat het eclipse project met de source code. Om de cluster op te starten moet de code gecompileerd worden tot een jar file.

Om de cluster lokaal te runnen, moeten enkele zaken eerst opgestart worden (in verschillende consoles), er wordt hier verondersteld dat de storm folder tot het pad toegevoegd is:
```
zookeeper-server-start .../zookeeper.properties
storm nimbus (in console met admin rechten)
storm supervisor (in console met admin rechten)
storm ui
```

Daarna kan de topologie opgestart worden in een nieuwe console, vervang "cluster_jar_name" door de naam van de gecompileerde jar:

```
cd {project pad}/storm_cluster 
storm jar target/cluster_jar_name.jar org.apache.storm.flux.Flux --local topology.yaml
```

De statistieken van de cluster kunnen geraadpleegd worden in de browser op localhost:8080

De logs zijn te vinden in .../apache-storm-2.1.0/logs

## Implementatie & configuratie topologie

Er is geen enkele lijn java code geschreven, de code voor de spouts en bolts zijn te vinden in storm_cluster/multilang/resources folder. 
Indien men de topologie wil veranderen moet men enkel topology.yml aanpassen. 




