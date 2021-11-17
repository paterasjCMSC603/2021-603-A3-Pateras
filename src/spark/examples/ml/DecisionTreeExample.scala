package spark.examples.ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession

object DecisionTreeExample {
  
	def main(args: Array[String]): Unit = {
	  
			val spark = SparkSession
					.builder
					.appName("DecisionTree")
					.config("spark.master", "local[4]")
					.getOrCreate()

			// Load the data stored in LIBSVM format as a DataFrame.
			val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

			// Split the data into training and test sets (30% held out for testing).
			val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

			// Train a DecisionTree model.
			val dt = new DecisionTreeClassifier()
			.setLabelCol("label")
			.setFeaturesCol("features")

			// Train model. This also runs the indexers.
			val model = dt.fit(trainingData)

			// Make predictions.
			val predictions = model.transform(testData)

			// Select (prediction, true label) and compute test error.
			val evaluator = new MulticlassClassificationEvaluator()
			.setLabelCol("label")
			.setPredictionCol("prediction")
			.setMetricName("accuracy")
			
			val accuracy = evaluator.evaluate(predictions)
			println(s"Test Error = ${(1.0 - accuracy)}")

			val treeModel = model.asInstanceOf[DecisionTreeClassificationModel]
			println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

			spark.stop()
	}
}