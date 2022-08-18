#!/bin/bash
#export SPARK_SUBMIT_OPTS=-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=7777
K8S_SERVER=$(minikube kubectl -- config view --output=jsonpath='{.clusters[].cluster.server}')

/home/ducha/Workspace/Tools/spark-3.2.1-bin-hadoop3.2/bin/spark-submit \
--master k8s://"$K8S_SERVER" \
--deploy-mode cluster \
--name word-embeddings \
--conf spark.kubernetes.container.image=spark:v3.2.1 \
--conf spark.kubernetes.context=minikube \
--conf spark.driver.memory=3g \
--conf spark.executor.memory=2g \
--conf spark.executor.instances=4 \
--conf spark.executor.cores=1 \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
--class "Main" \
--verbose \
local:///app/spark-nlp-starter-assembly-3.3.4.jar