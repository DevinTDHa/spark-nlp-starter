import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.pretrained.{PretrainedPipeline, ResourceDownloader}
import com.johnsnowlabs.nlp.training.CoNLL

object Main {
  val spark: SparkSession = SparkSession.builder
    .appName("spark-nlp-starter")
    .getOrCreate

  def main(args: Array[String]): Unit = {

    spark.sparkContext.setLogLevel("ERROR")
    //
    //    val documentAssembler = new DocumentAssembler()
    //      .setInputCol("text")
    //      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setEnableInMemoryStorage(true)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(
        //        documentAssembler,
        tokenizer, embeddings, embeddingsFinisher))

    val data = CoNLL().readDataset(spark, "/data/eng.train")

    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(20, 80)
  }

  def pretrainedPipeline(args: Array[String]): Unit = {

    spark.sparkContext.setLogLevel("ERROR")

    val testData = spark.createDataFrame(Seq(
      (1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
      (2, "The Paris metro will soon enter the 21st century, ditching single-use paper tickets for rechargeable electronic cards.")
    )).toDF("id", "text")

    val pipeline = new PretrainedPipeline("explain_document_dl", lang = "en")
    pipeline.annotate("Google has announced the release of a beta version of the popular TensorFlow machine learning library")
    pipeline.transform(testData).select("entities").show(false)

    val pipelineML = new PretrainedPipeline("explain_document_ml", lang = "en")
    pipelineML.annotate("Google has announced the release of a beta version of the popular TensorFlow machine learning library")
    pipelineML.transform(testData).select("pos").show(false)

  }
}
