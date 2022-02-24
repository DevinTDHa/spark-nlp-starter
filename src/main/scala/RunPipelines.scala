import Main.spark
import Main.spark.implicits._
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
import com.johnsnowlabs.nlp.base.{DocumentAssembler, Finisher, RecursivePipeline}
import com.johnsnowlabs.nlp.embeddings.{AlbertEmbeddings, BertSentenceEmbeddings, UniversalSentenceEncoder, WordEmbeddingsModel, XlmRoBertaEmbeddings}
import org.apache.spark.ml.Pipeline

object RunPipelines {
  def classifierDl(): Unit = {
    // First extract the prerequisites for the NerDLModel
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    // Use the transformer embeddings
    val embeddings = BertSentenceEmbeddings
      .pretrained("sent_bert_multi_cased", "xx")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    // This pretrained model requires those specific transformer embeddings
    val document_classifier = ClassifierDLModel.pretrained("classifierdl_bert_news", "de")
      .setInputCols("sentence_embeddings")
      .setOutputCol("class")


    val pipeline = new Pipeline().setStages(Array(
      documentAssembler,
      sentence,
      tokenizer,
      embeddings,
      document_classifier
    ))

    val data = Seq("The Grand Budapest Hotel is a 2014 comedy-drama film written and directed by Wes Anderson").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("class.result").show(false)
  }

  def nerDl(): Unit = {
    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val token = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val posTagger = PerceptronModel.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val wordEmbeddings = WordEmbeddingsModel.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("word_embeddings")

    val ner = NerDLModel.pretrained("ner_dl", "en")
      .setInputCols("token", "sentence", "word_embeddings")
      .setOutputCol("ner")

    val nerConverter = new NerConverter()
      .setInputCols("sentence", "token", "ner")
      .setOutputCol("ner_converter")

    val finisher = new Finisher()
      .setInputCols("ner", "ner_converter")
      .setCleanAnnotations(false)

    val pipeline = new Pipeline().setStages(
      Array(
        document,
        sentenceDetector,
        token,
        posTagger,
        wordEmbeddings,
        ner,
        nerConverter,
        finisher))

    val testData = spark.createDataFrame(Seq(
      (1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
      (2, "The Paris metro will soon enter the 21st century, ditching single-use paper tickets for rechargeable electronic cards.")
    )).toDF("id", "text")

    val predicion = pipeline.fit(testData).transform(testData)
    predicion.select("ner_converter.result").show(false)
    predicion.select("pos.result").show(false)
  }

  def sentimentDl(): Unit = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val useEmbeddings = UniversalSentenceEncoder.pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val sentiment = SentimentDLModel.pretrained("sentimentdl_use_twitter")
      .setInputCols("sentence_embeddings")
      .setThreshold(0.7F)
      .setThresholdLabel("neutral")
      .setOutputCol("sentiment")

    val pipeline = new Pipeline().setStages(Array(
      documentAssembler,
      useEmbeddings,
      sentiment
    ))

    val data = Seq(
      "Wow, the new video is awesome!",
      "bruh this sucks what a damn waste of time"
    ).toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("text", "sentiment.result").show(false)
  }

  def albert(): Unit = {


    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings.pretrained()
      .setInputCols("token", "document")
      .setOutputCol("embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline().setStages(Array(
      documentAssembler,
      tokenizer,
      embeddings,
      embeddingsFinisher
    ))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }
  def albertForTok(): Unit = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
    val tokenClassifier = AlbertForTokenClassification.pretrained()
      .setInputCols("token", "document")
      .setOutputCol("label")
      .setCaseSensitive(true)
    val pipeline = new Pipeline().setStages(Array(
      documentAssembler,
      tokenizer,
      tokenClassifier
    ))
    val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("label.result").show(false)
  }

  def t5(): Unit = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer.pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setMaxOutputLength(200)
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val data = Seq(
      "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a " +
        "downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness" +
        " of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this " +
        "paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework " +
        "that converts all text-based language problems into a text-to-text format. Our systematic study compares " +
        "pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens " +
        "of language understanding tasks. By combining the insights from our exploration with scale and our new " +
        "Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many benchmarks covering " +
        "summarization, question answering, text classification, and more. To facilitate future work on transfer " +
        "learning for NLP, we release our data set, pre-trained models, and code."
    ).toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("summaries.result").show(false)
  }

  def useMultiL(): Unit = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val useEmbeddings = UniversalSentenceEncoder
      .pretrained("tfhub_use_multi", "xx")
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        useEmbeddings
      ))

    val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("sentence_embeddings.result").show(false)
  }

  def xlmRoberta(): Unit = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = XlmRoBertaEmbeddings.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        embeddings,
        embeddingsFinisher
      ))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }
}
