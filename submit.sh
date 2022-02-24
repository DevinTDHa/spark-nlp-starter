#!/bin/bash
/home/ducha/Workspace/Tools/spark-3.1.3-bin-hadoop3.2/bin/spark-submit \
--driver-memory 12g \
--class "Main" \
target/scala-2.12/spark-nlp-starter-assembly-3.3.4.jar