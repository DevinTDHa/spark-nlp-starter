# spark-nlp-starter

This is just a simple demo as how to use Spark NLP in a SBT project in Scala. Usually, once you have your application you want to package it and run it via `spark-submit`.

## spark-submit
After your executed `sbt assembly` to get a Fat JAR 
(without Apache Spark since your environment has Apache Spark already), you can use `spark-submit` like this:
```shell
export SPARK_NLP_LICENSE=$YOUR_SPARK_NLP_LICENSE; \
export AWS_ACCESS_KEY_ID=$YOUR_AWS_ACCESS_KEY_ID; \
export AWS_SECRET_ACCESS_KEY=$YOUR_AWS_SECRET_ACCESS_KEY; \
./bin/spark-submit \
--conf spark.driver.extraJavaOptions=-Dconfig.file=$PATH_TO/application.conf \
--conf spark.executor.extraJavaOptions=-Dconfig.file=$PATH_TO/application.conf \
--class "Main" $PATH_TO/target/scala-2.12/mt-sinai-demo-assembly-1.0.jar
```
Note that the environment variables need to be set on each node.

## Getting Dependencies for Spark NLP Healthcare
Link to Spark NLP Healthcare jar:
https://pypi.johnsnowlabs.com/$SECRET/spark-nlp-jsl-$JSL_VERSION.jar

SECRET and JSL_VERSION are the keys to the values in the license.json file.