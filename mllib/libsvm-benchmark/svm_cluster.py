from pyspark.ml.classification import LinearSVC, LinearSVCSummary
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("svm-cluster").master('spark://a4.wanglab.ml:7077') \
    .config('spark.executor.memory','1G') \
    .config('spark.executor.cores','2') \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "file:///home/rexma/local/spark-3.1.2-bin-hadoop3.2/spark-log") \
    .config("spark.history.fs.logDirectory", "file:///home/rexma/local/spark-3.1.2-bin-hadoop3.2/spark-log") \
    .getOrCreate()

# Load training data
training = spark.read.format("libsvm").load("hdfs://a4.wanglab.ml:9000/csc2222/covtype.libsvm.binary").repartition(12)

lsvc = LinearSVC(maxIter=20, regParam=0.1)

# Fit the model
lsvcModel = lsvc.fit(training)

# Print the coefficients and intercept for linear SVC
print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lsvcModel.summary()
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
# trainingSummary.residuals.show()
# print("MSE: %f" % trainingSummary.meanSquaredError)
# print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
# print("r2: %f" % trainingSummary.r2)
