package spark.examples.ml

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object LinearRegressionWithElasticNetExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearRegressionExample")
      .config("spark.master", "local[4]")
      .getOrCreate()

    // Load training data
    val training = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    spark.stop()
  }
}