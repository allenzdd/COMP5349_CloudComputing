'''
edit by lab115_2 May 2018
'''

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from q3_utils import *
from pyspark.sql.functions import *


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("COMP5349 Assignment 2 MLP") \
        .getOrCreate()

    # set init from arugment
    args = argument_set()

    # take the arugment as follow
    train_datafile = args.train_input
    test_datafile = args.test_input
    maxIter = args.maxIter
    layers = np.array(args.layers.split(','), dtype=int)
    blockSize = args.blockSize

    # read data
    train_df = spark.read.csv(train_datafile, header=False, inferSchema="true")
    test_df = spark.read.csv(test_datafile, header=False, inferSchema="true")

    # assembler them
    assembler = VectorAssembler(
        inputCols=train_df.columns[1:], outputCol="features")

    # mlp init
    mlp = MultilayerPerceptronClassifier(
        labelCol='_c0', featuresCol="features", maxIter=maxIter, layers=layers, blockSize=blockSize, seed=1)

    # pipeline set
    pipeline = Pipeline(stages=[assembler, mlp])

    # fit model
    model = pipeline.fit(train_df)

    res = model.transform(test_df)
    ress = res.select(col('_c0').alias('label'), 'prediction')

    # evaluation init
    evall = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    # calculate accuracy
    accuracy = evall.evaluate(ress)

    print("-----------------------------")
    print("the accuracy is %f" % accuracy)
    print("-----------------------------")
