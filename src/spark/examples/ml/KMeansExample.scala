package spark.examples.ml

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession

object KMeansExample {

	def main(args: Array[String]): Unit = {
			val spark = SparkSession
					.builder
					.appName("KMeansExample")
					.config("spark.master", "local[4]")
					.getOrCreate()

			// Load training data
			val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

			// Trains a k-means model.
			val kmeans = new KMeans().setK(2).setSeed(1L)
			val model = kmeans.fit(dataset)

			// Make predictions
			val predictions = model.transform(dataset)

			// Evaluate clustering by computing Silhouette score
			val evaluator = new ClusteringEvaluator()

			val silhouette = evaluator.evaluate(predictions)
			println(s"Silhouette with squared euclidean distance = $silhouette")

			// Shows the result.
			println("Cluster Centers: ")
			model.clusterCenters.foreach(println)

			spark.stop()
	}
}