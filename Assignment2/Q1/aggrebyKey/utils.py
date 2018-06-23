

import numpy as np
import argparse
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.sql.functions import *
from pyspark.accumulators import AccumulatorParam
from pyspark.ml import Pipeline
import pyspark


# init the argument set
def argument_set():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", help="the train data path",
                        default='/share/MNIST/Train-label-28x28.csv')
    parser.add_argument("--test_input", help="the test data path",
                        default='/share/MNIST/Test-label-28x28.csv')
    parser.add_argument("--k_PCA", help="the k of PCA", default=50, type=int)
    parser.add_argument("--k_knn", help="the k of knn", default=5, type=int)
    parser.add_argument(
        "--distance", help="the distance method of knn", default="euclidean")
    parser.add_argument(
        "--show_method", help="the method of show result", default="accumulator")
    parser.add_argument(
        "--repartition_test", help="the repartition of test data", default=16, type=int)
    args = parser.parse_args()

    # set the test data repartition part in this file with global variables
    global repartition_test
    repartition_test = args.repartition_test
    return args


# the pipeline function with assembler and PCA
def pipeline_assembler_pca(Spark, trainDat, testDat, k=50):
    # read data
    train_df = Spark.read.csv(trainDat, header=False, inferSchema="true")
    test_df = Spark.read.csv(testDat, header=False, inferSchema="true")

    # assembler them
    assembler = VectorAssembler(
        inputCols=train_df.columns[1:], outputCol="features")
    # PCA init
    pca = PCA(k=k, inputCol="features", outputCol="features_pcas")

    # pipeline set
    pipeline = Pipeline(stages=[assembler, pca])
    # fit model
    model = pipeline.fit(train_df)

    # transform train data
    train_pca_result = model.transform(train_df).select(
        col(train_df.columns[0]).alias("label"), "features_pcas")
    # transform test data
    test_pca_result = model.transform(test_df).select(
        col(test_df.columns[0]).alias("label"), "features_pcas") \
        .repartition(repartition_test)  # add repartition
    return train_pca_result, test_pca_result
