from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sparktorch import SparkTorch, serialize_torch_obj
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper
import torch
import torch.nn as nn
from mynet import MyNet


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("p8_b64_itr50") \
        .master("local[8]") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", "file:///home/rexma/local/spark-3.1.2-bin-hadoop3.2/spark-log") \
        .config("spark.history.fs.logDirectory", "file:///home/rexma/local/spark-3.1.2-bin-hadoop3.2/spark-log") \
        .getOrCreate()

    # Read in mnist_train.csv dataset
    df = spark.read.option("inferSchema", "true").csv('hdfs://a4.wanglab.ml:9000/csc2222/mnist_train.csv').orderBy(rand()).repartition(8)

    network = MyNet()

    # Build the pytorch object
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD,
        lr=0.001
    )

    # Setup features
    vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')

    # Demonstration of some options. Not all are required
    # Note: This uses the barrier execution mode, which is sensitive to the number of partitions
    spark_model = SparkTorch(
        inputCol='features',
        labelCol='_c0',
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=50,
        verbose=1,
        validationPct=0.2,
        miniBatch=64
    )

    # Create and save the Pipeline
    p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
    p.write().overwrite().save('simple_cnn')

    # Example of loading the pipeline
    loaded_pipeline = PysparkPipelineWrapper.unwrap(PipelineModel.load('simple_cnn'))

    # Run predictions and evaluation
    predictions = loaded_pipeline.transform(df).persist()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="_c0", predictionCol="predictions", metricName="accuracy")

    accuracy = evaluator.evaluate(predictions)
    print("Train accuracy = %g" % accuracy)
