import matplotlib
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import StructField, StructType

# import findspark
# findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
import matplotlib.pyplot as plt
from time import time

from pyspark.sql.functions import *
import argparse
from pyspark import SparkContext

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("COMP5349 Assignment 2 RF") \
        .getOrCreate()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="the input path",
                        default='/share/')
    parser.add_argument("--output", help="the output path",
                        default='/user/zwan5430/a2')

    parser.add_argument("--numOfTrees", help="The tree number",
                        default=8)

    parser.add_argument("--treeDepth", help="The depth of each tree",
                        default=8)

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    tree_num = int(args.numOfTrees)
    tree_depth = int(args.treeDepth)

    train_datafile = input_path + "/MNIST/Train-label-28x28.csv"
    test_datafile = input_path + "/MNIST/Test-label-28x28.csv"

    train_df = spark.read.csv(train_datafile, header=False, inferSchema="true").withColumnRenamed("_c0", "label")
    test_df = spark.read.csv(test_datafile, header=False, inferSchema="true").withColumnRenamed("_c0", "label")

    assembler = VectorAssembler(inputCols=train_df.columns[1:], outputCol="features")

    start_time = time()

    rf = RandomForestClassifier(numTrees=tree_num, maxDepth=tree_depth, labelCol="label",
                                featuresCol="features", seed=42)
    rf_noPCA_pipeline = Pipeline(stages=[assembler, rf])
    model_no_pca = rf_noPCA_pipeline.fit(train_df)
    result = model_no_pca.transform(test_df).select("label", "prediction")

    end_time = time()

    ress = result.select('label', 'prediction')

    # evaluation init
    evall = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    # calculate accuracy
    accuracy = evall.evaluate(ress)

    print("-----------------------------")
    print("the Treenum is %s," % tree_num)
    print("the depth is %s" % tree_depth)
    print("the accuracy is %f" % accuracy)
    print("the time is %s" % (end_time - start_time))
    print("-----------------------------")
