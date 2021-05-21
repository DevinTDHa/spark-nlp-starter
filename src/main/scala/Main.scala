import com.johnsnowlabs.nlp.DocumentAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.ner.{MedicalNerModel, NerConverter}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel

object Main {
  def main(args: Array[String]) {
    val spark: SparkSession = SparkSession.builder
      .appName("Spark Demo")
      .master("local[*]")
      .config("spark.driver.memory", "12G")
      .config("spark.driver.maxResultSize", "0")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "1000M")
      .getOrCreate()

    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("tokens")

    val sentencer = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val embeddings = WordEmbeddingsModel
      .pretrained("embeddings_clinical", "en", "clinical/models")
      .setInputCols(Array("sentences", "tokens"))
      .setOutputCol("embeddings")

    val posTagger = PerceptronModel
      .pretrained("pos_clinical", "en", "clinical/models")
      .setInputCols(Array("sentences", "tokens"))
      .setOutputCol("posTags")

    val nerTagger = MedicalNerModel
      .pretrained("ner_clinical", "en", "clinical/models")
      .setInputCols(Array("sentences", "tokens", "embeddings"))
      .setOutputCol("nerTags")

    val nerConverter = new NerConverter()
      .setInputCols(Array("sentences", "tokens", "nerTags"))
      .setOutputCol("nerChunks")

    //set up the pipeline
    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentencer,
        tokenizer,
        embeddings,
        posTagger,
        nerTagger,
        nerConverter
      ))


    val text = Seq(("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to" +
      " presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced " +
      "pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body " +
      "mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , " +
      "and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a " +
      "respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin " +
      "and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical" +
      " examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination " +
      "was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : " +
      "serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL" +
      " , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was " +
      "normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to " +
      "significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral " +
      "intake for three days prior to admission . However , serum chemistry obtained six hours after presentation " +
      "revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L ," +
      " triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and " +
      "found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed " +
      "prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an " +
      "insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL ," +
      " within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of " +
      "SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of " +
      "insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It " +
      "was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with " +
      "endocrinology post discharge .")).toDF("text")

    val model = pipeline.fit(text)
    val results = model.transform(text)

    results
      .selectExpr("explode(nerChunks) as ner_chunks")
      .selectExpr("ner_chunks.result", "ner_chunks.metadata.entity")
      .show(false)

  }
}
