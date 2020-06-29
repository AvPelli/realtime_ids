# 1. machine learning

De python file "ML_modellen" bevat alle code voor de verschillende modellen op te stellen.

# 2. realtime cluster

Voor de cluster is volgende software nodig:
| Dependency | version  |   
| ------- | --- |
| Apache Kafka | 2.5.0 |
| Apache Storm | 2.1.0 |
| Java  | 8 |

De map /storm_cluster bevat het eclipse project met de source code. Om de cluster op te starten moet de code gecompileerd worden tot een jar file.

Om de cluster lokaal te runnen, moeten enkele zaken eerst opgestart worden (in verschillende consoles):
```
zookeeper-server-start .../zookeeper.properties
storm nimbus (in console met admin rechten)
storm supervisor (in console met admin rechten)
storm ui
```

Daarna kan de topologie opgestart worden in een nieuwe console:

```
cd .../storm_cluster (project pad)
storm jar cluster_jar_name.jar org.apache.storm.flux.Flux --local topology.yaml
```

De statistieken van de cluster kunnen geraadpleegd worden in de browser op localhost:8080

