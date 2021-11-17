package spark.examples.ml

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.DataFrame

object kNN {
  
  def mapper(train: DataFrame, test: DataFrame) {
  // convert the assigned chunk of train instances to an array?
    
  //for i in length test
      //calculate distance from ith test instance to each train in trainChunk
      //trainChunk(class, features) => (test index, (class, distance))
      
      //sortByKey(distance, class) and choose 3 shortest classes.
    
  }
  
  def main(args: Array[String]) {
    
  // set context
	val spark = SparkSession
					.builder
					.appName("AssignmentThree")
					.config("spark.master", "local[4]")
					.getOrCreate()
					
	import spark.implicits._
	
  // load training and test data
	// small
  val trainData = spark.read.format("libsvm").load("data/mllib/small-train.libsvm")
  val testData = spark.read.format("libsvm").load("data/mllib/small-test.libsvm")
  
  // medium 
  // val trainData = spark.read.format("libsvm").load("data/mllib/medium-train.libsvm")
  // val testData = spark.read.format("libsvm").load("data/mllib/medium-test.libsvm")
  
  // large
  // val trainData = spark.read.format("libsvm").load("data/mllib/large-train.libsvm")
  // val testData = spark.read.format("libsvm").load("data/mllib/large-test.libsvm")
   
  // give each test instance a query index
  trainData.show()
  testData.show()  // need to broadcast test data so all distributions of train can see
  
  // calculate the distance between each instance in training chunk and broadcasted test 
  // map trainer chunk to executor, but each executor must see whole test set (broadcast)
  
  trainData.map(partial(mapper,testData))
  
  // reduce mapper output
  // reduceByKey(minimum distance) and use three shortest to vote for classification
  // return array of test instances and predictions
   
  // test accuracy by comparing to ground truth
  
  }
}