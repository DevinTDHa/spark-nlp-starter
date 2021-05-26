import com.typesafe.sbt.packager.archetypes.JavaAppPackaging

enablePlugins(JavaServerAppPackaging)
enablePlugins(JavaAppPackaging)

val scalaTestVersion = "3.0.0"

name := "mt-sinai-demo"

version := "1.1"

scalaVersion := "2.11.12"

javacOptions ++= Seq("-source", "1.8", "-target", "1.8")

libraryDependencies ++= {
  val sparkVer = "2.4.5"
  val sparkNLP = "3.0.2"
  val sparkNlpJsl = "3.0.2"

  // The secret is read from a environment variable. Alternatively can be defined explicitly here as well.
  //  val jslSecret = "$YOURSECRET"
  val jslSecret = sys.env("SECRET")
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer % Provided,
    "org.apache.spark" %% "spark-sql" % sparkVer % Provided,
    "org.apache.spark" %% "spark-mllib" % sparkVer % Provided,
    "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
    "com.johnsnowlabs.nlp" %% "spark-nlp-spark24" % sparkNLP,
    "com.johnsnowlabs.nlp" % "spark-nlp-jsl-spark24" % sparkNLP
      from s"https://pypi.johnsnowlabs.com/$jslSecret/spark-nlp-jsl-$sparkNlpJsl-spark24.jar"
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