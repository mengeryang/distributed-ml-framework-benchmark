from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("linear-regression-house").getOrCreate()

# Load training data
data = spark.read.csv("E:\\Git\\csc2222\\housing.csv", header=False, inferSchema=True)\
    .toDF('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV')
data.printSchema()

feature_columns = data.columns[:-1]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data2 = assembler.transform(data)

train, test = data2.randomSplit([0.7, 0.3])

lr = LinearRegression(featuresCol="features", labelCol="MEDV", maxIter=100, regParam=0.3, elasticNetParam=0.2)
# lr = LinearRegression(featuresCol="features", labelCol="MEDV", maxIter=40)


# Fit the model
lrModel = lr.fit(data2)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("MSE: %f" % trainingSummary.meanSquaredError)
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

spark.stop()

