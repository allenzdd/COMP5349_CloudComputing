'''
edit by lab115_2 May 2018
'''

from pyspark.sql import SparkSession
import argparse
from q1_utils import *
from q1_other_function import *
from pyspark.ml.feature import PCA
from pyspark.sql.functions import *
from time import time
from pyspark.accumulators import AccumulatorParam
import pyspark


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("COMP5349 Assignment 2 KNN") \
        .getOrCreate()

    # set init from arugment
    args = argument_set()

    # take the arugment as follow
    train_datafile = args.train_input
    test_datafile = args.test_input
    k_PCA = args.k_PCA
    k_knn = args.k_knn
    dist_knn = args.distance
    result_show_method = args.show_method

    # read data and assembler and PCA with pipeline
    train_vector_pca, test_vector_pca = pipeline_assembler_pca(
        spark, train_datafile, test_datafile, k=k_PCA)

    # broadcast the train data
    # init knn_model with train data
    model = knn_Spark(spark, train_vector_pca, k_knn=k_knn,
                      distance=dist_knn, show_method=result_show_method)

    if result_show_method == "accumulator":
        res, confMatrix = model.predict(test_vector_pca)

        start_time = time()

        # persist the result in order to only store in memory
        print(res.persist(pyspark.StorageLevel.MEMORY_ONLY).collect())

        confMatrix = np.array(confMatrix.value).reshape(10, 10).T

        label = np.array(range(0, 10)).reshape(10, 1)
        stats = make_statistics(confMatrix, label)
        print_table_df(spark, label, stats)
        end_time = time()

        print("--- The knn of spark model used: %2f seconds" %
              (end_time - start_time))

    if result_show_method == "group_agg":
        res = model.predict(test_vector_pca)

        start_time = time()
        # show the evaluate table with dataframe
        result = evaluate_precision_recall_fscore(res.repartition(1).cache())
        result.show()

        end_time = time()

        print("--- The knn of spark model used: %2f seconds" %
              (end_time - start_time))

    spark.stop()
