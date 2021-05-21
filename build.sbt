import com.typesafe.sbt.packager.archetypes.JavaAppPackaging

enablePlugins(JavaServerAppPackaging)
enablePlugins(JavaAppPackaging)

val scalaTestVersion = "3.0.0"

name := "mt-sinai-demo"

version := "1.0"

scalaVersion := "2.12.10"

javacOptions ++= Seq("-source", "1.8", "-target", "1.8")

libraryDependencies ++= {
  val sparkVer = "3.0.2"
  val sparkNLP = "3.0.0"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer % Provided,
    "org.apache.spark" %% "spark-sql" % sparkVer % Provided,
    "org.apache.spark" %% "spark-mllib" % sparkVer % Provided,
    "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
    "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNLP,
    "com.johnsnowlabs.nlp" % "spark-nlp-jsl" % "3.0.0" from "file:///home/ducha/Workspace/JSL/setup/spark-nlp-jsl-3.0.0.jar"
  )
}

/** Disables tests in assembly */
test in assembly := {}

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs@_*) => MergeStrategy.discard
  case x if x.startsWith("NativeLibrary") => MergeStrategy.last
  case x if x.startsWith("aws") => MergeStrategy.last
  case _ => MergeStrategy.last
}

/*
* If you wish to make a Uber JAR (Fat JAR) without Spark NLP
* because your environment already has Spark NLP included same as Apache Spark
**/
//assemblyExcludedJars in assembly := {
//  val cp = (fullClasspath in assembly).value
//  cp filter {
//    j => {
//        j.data.getName.startsWith("spark-nlp")
//    }
//  }
//}