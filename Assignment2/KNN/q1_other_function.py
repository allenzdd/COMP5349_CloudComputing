import numpy as np
from pyspark.sql import Row
from pyspark.sql.types import *


def get_statistics_for_class(confusion_matrix, label):
    TP = confusion_matrix[label, label]
    FN = np.sum(confusion_matrix[label]) - TP
    FP = np.sum(confusion_matrix[:, label]) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP
    return TP, TN, FP, FN


def precision(TP, TN, FP, FN):
    return TP / (TP + FP)


def recall(TP, TN, FP, FN):
    return TP / (TP + FN)


def F_measure(TP, TN, FP, FN):
    return 2 * TP / (2 * TP + FP + FN)


def make_statistics(confusion_matrix, full_class_list):
    # Determine statistical parameters precision, recall, F-measure from TP and TN
    class_count = len(full_class_list)
    stats = np.zeros((class_count, 3))
    for i in range(class_count):
        TP, TN, FP, FN = get_statistics_for_class(confusion_matrix, i)
        stats[i, 0] = np.round(precision(TP, TN, FP, FN), 4)
        stats[i, 1] = np.round(recall(TP, TN, FP, FN), 4)
        stats[i, 2] = np.round(F_measure(TP, TN, FP, FN), 4)
    return stats


def print_table_df(Spark, label, stats):
    label_stats = np.concatenate((label, stats), axis=1)
    R = Row('label', 'precision', 'recall', 'F_measure')
    df = [R(int(x[0]), float(x[1]), float(x[2]),
            float(x[3])) for x in label_stats]
    schema = StructType([
        StructField("label", IntegerType(), True),
        StructField("precision", FloatType(), True),
        StructField("recall", FloatType(), True),
        StructField("F_measure", FloatType(), True),
    ])
    Spark.createDataFrame(df, schema).show()
