from pyspark.ml.classification import LinearSVC, LinearSVCSummary
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("svm-sample").getOrCreate()

# Load training data
training = spark.read.format("libsvm").load("covtype.libsvm.binary")

lsvc = LinearSVC(maxIter=10, regParam=0.1)

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
