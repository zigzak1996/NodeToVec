import java.io.File

import breeze.io.CSVWriter
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import breeze.linalg.{DenseMatrix, DenseVector, argsort, convert, csvread, csvwrite, min, norm}
import breeze.numerics.{exp, log, sigmoid}
import org.apache.spark.{HashPartitioner, RangePartitioner, SparkConf, SparkContext}
import org.apache.spark.mllib.random.UniformGenerator

import scala.collection.immutable
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

object NodeToVec {
	
  val r = new Random()

  val total_nodes = 40334
  val emb_dim = 100
  val learning_rate = 0.001f
  val epochs = 30


  def read_data(path: String, spark: SparkSession): RDD[(Int, Int)] = {
    spark.read.format("csv")
      // the original data is store in CSV format
      // header: source_node, destination_node
      // here we read the data from CSV and export it as RDD[(Int, Int)],
      // i.e. as RDD of edges
      .option("header", "true")
      // State that the header is present in the file
      .schema(StructType(Array(
      StructField("source_node", IntegerType, false),
      StructField("destination_node", IntegerType, false)
    )))
      // Define schema of the input data
      .load(path)
      // Read the file as DataFrame
      .rdd.map(row => (row.getAs[Int](0), row.getAs[Int](1)))
    // Interpret DF as RDD
  }

  def create_batches(data: RDD[(Int, Int)], batch_size: Int) = {
  	val indexed_data = data.zipWithIndex().map(x => ((x._2 / batch_size).toInt, x._1))
    val batches_count: Int = (data.count() / batch_size).toInt
    for (i <- 0 to batches_count)
      yield indexed_data.filter(_._1 == i).map(_._2)
      
  }

  def create_embedding_matrix(emb_dim: Int, total_nodes: Int) = {
    val uni = new UniformGenerator()
    new DenseMatrix(emb_dim, total_nodes, Array.fill(total_nodes*emb_dim)(uni.nextValue().toFloat - 0.5f))
  }


  def estimate_loss(
                source: Int,
                destination: Int,
                emb_in: DenseMatrix[Float],
                emb_out: DenseMatrix[Float]
               ) = {
    
    val in = emb_in(::, source)
    val out = emb_out(::, destination)

    -log(sigmoid.apply(in.dot(out)))


  }

  def estimate_gradients_for_edge(
                                   source: Int,
                                   destination: Int,
                                   emb_in: DenseMatrix[Float],
                                   emb_out: DenseMatrix[Float]
                                 ): ((Int, DenseVector[Float]), (Int, DenseVector[Float])) = {

    val in = emb_in(::, source)
    val out = emb_out(::, destination)

    val neg_num = 20

    var in_grads = out * -sigmoid.apply(-in.dot(out))
    var out_grads = in * -sigmoid.apply(-in.dot(out))

    for (_ <- 1 to neg_num) {
      val negative_vec = emb_out(::, r.nextInt(total_nodes))
      in_grads += negative_vec * sigmoid.apply(in.dot(negative_vec))
    }


    /*
     * Estimate gradients
     */

    // return a tuple
    // Tuple((Int, DenseVector), (Int, DenseVector))
    // this tuple contains sparse gradients
    // in_grads is vector of gradients for
    // a node with id = source



    ((source, in_grads), (destination, out_grads))
  }

  def top_k(
             source: Int,
             k: Int,
             emb_in: DenseMatrix[Float],
             emb_out: DenseMatrix[Float],
             vals_to_remove: Array[Int]
           ) = {

      val vec: DenseVector[Float] = emb_out.t * emb_in(::, source)

      vals_to_remove.foreach(x => {
        vec(x) = Float.NegativeInfinity
      })
      vec(source) = Float.NegativeInfinity
      val sorted_indxs = argsort(vec)
      val slice = sorted_indxs.slice(total_nodes-11, total_nodes - 1 ).reverse

      slice
  }


  def averagePrecision(pred:Array[Int], lab:Array[Int]): Double = {


        var i:Double = 0.0
        var cnt = 0
        var precSum = 0.0
        val n = lab.length
        for (value <- pred){
          if (lab.contains(value)) {
              cnt += 1
              precSum += cnt.toDouble / (i + 1.0)
          }

            i += 1.0
        }
        precSum / min(lab.length, 10)

    }


  def fit(emb_in: DenseMatrix[Float], emb_out: DenseMatrix[Float], batches: immutable.IndexedSeq[RDD[(Int, Int)]], test_data: RDD[(Int, Int)], sc: SparkContext) = {
    for (i <- 1 to epochs) {
      var emb_in_broadcast = sc.broadcast(emb_in)
      var emb_out_broadcast = sc.broadcast(emb_out)
      var train_loss = 0.0f
      for (batch <- batches) {
        emb_in_broadcast = sc.broadcast(emb_in)
        emb_out_broadcast = sc.broadcast(emb_out)

        train_loss += batch.map(
          edge => estimate_loss(
            edge._1,
            edge._2,
            emb_in_broadcast.value,
            emb_out_broadcast.value
          )
        ).reduce(_ + _) / batch.count()


        val derivative = batch.map(
          edge => estimate_gradients_for_edge(
            edge._1,
            edge._2,
            emb_in_broadcast.value,
            emb_out_broadcast.value
          )
        )

        val inGrads = derivative
          .map(x => (x._1._1, (x._1._2, 1)))
          .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2 ))
          .collectAsMap()

        val outGrads = derivative
          .map(x => (x._2._1, (x._2._2, 1)))
          .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2 ))
          .collectAsMap()

        inGrads.foreach(x => {
          emb_in(::, x._1) -= learning_rate * (x._2._1 / x._2._2.toFloat)
        })
        outGrads.foreach(x => {
          emb_out(::, x._1) -= learning_rate * (x._2._1 / x._2._2.toFloat)
        })
      }
      emb_in_broadcast = sc.broadcast(emb_in)
      emb_out_broadcast = sc.broadcast(emb_out)


      if (i % 10 == 0) {
        CSVWriter.writeFile(
          new File("emb_in.csv"),
          IndexedSeq.tabulate(emb_in.rows, emb_in.cols)(emb_in(_, _).toString),
          ';',
          '\u0000',
          '\\'
        )
        CSVWriter.writeFile(
          new File("emb_out.csv"),
          IndexedSeq.tabulate(emb_in.rows, emb_in.cols)(emb_out(_, _).toString),
          ';',
          '\u0000',
          '\\'
        )
      }

      var test_loss = test_data
        .map(grad => estimate_loss(grad._1, grad._2, emb_in_broadcast.value, emb_out_broadcast.value))
        .reduce(_ + _)

      val loss = test_loss / test_data.count()

      println(s"epoch : $i test_loss : ${loss} train_loss : ${train_loss / 35f}")

    }
    CSVWriter.writeFile(
      new File("emb_in.csv"),
      IndexedSeq.tabulate(emb_in.rows, emb_in.cols)(emb_in(_, _).toString),
      ';',
      '\u0000',
      '\\'
    )
    CSVWriter.writeFile(
      new File("emb_out.csv"),
      IndexedSeq.tabulate(emb_in.rows, emb_in.cols)(emb_out(_, _).toString),
      ';',
      '\u0000',
      '\\'
    )
  }

  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)


    val spark = SparkSession
      .builder()
      .appName("NodeToVec")
      .config("spark.master", "local")
      .getOrCreate()

    val sc = spark.sparkContext
    val train_path = args(0)
    val test_path = args(1)
    val batch_size = 10000

    val train_data = read_data(train_path, spark)
    val test_data = read_data(test_path, spark)

    var emb_in = DenseMatrix.zeros[Float](emb_dim, total_nodes)
    var emb_out = DenseMatrix.zeros[Float](emb_dim, total_nodes)

    if (args.length < 4) {
      val batches = create_batches(train_data, batch_size)


      emb_in = create_embedding_matrix(emb_dim, total_nodes)
      emb_out = create_embedding_matrix(emb_dim, total_nodes)

      fit(emb_in, emb_out, batches, test_data, sc)


    }
    else {
      val batches = create_batches(train_data, batch_size)

      emb_in = csvread(new File(args(2)),';').map(x=>x.toFloat)
      emb_out = csvread(new File(args(3)), ';').map(x=>x.toFloat)
      fit(emb_in, emb_out, batches, test_data, sc)

    }
    //csvwrite(new File("emb_in.csv"), emb_in, separator = ';')
    //csvwrite(new File("emb_out.csv"), emb_out[Float], separator = ';')
//    val list = new DenseVector[Double](total_nodes)
//    (0 to total_nodes - 1).foreach(source => {
//        val test_nodes: Array[(Int, Int)] = test_data.filter(_._1 == source)
//          .join(train_data)
//          .map(x => (x._2._1, x._2._2))
//          .collect()
//        if (test_nodes.length > 0) {
//          val pred_nodes: Array[Int] = top_k(source, 10, emb_in, emb_out, rem_vals).toArray
//
//          list(source) = averagePrecision(pred_nodes, test_nodes)
//        }
//        if (source % 10 == 0){
//          println(mean(list))
//        }
//      }
//    )
//    println(mean(list))
  }
}