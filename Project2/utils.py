from pyspark.sql.functions import *
from pyspark.sql import SparkSession,DataFrame
from typing import Union,List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def read_csv_table(spark:SparkSession,file_name:str)->DataFrame:
    return spark.read.csv(f'./{file_name}.csv',inferSchema=True, header=True)


def convert_to_pandas(spark:SparkSession)->pd.DataFrame:
    pandasDataFrame = spark.toPandas()
    return pandasDataFrame

def join_dataframe(dataframe_izquierdo,dataframe_derecho,criterio_on,how)->DataFrame:

    join_df = (
    dataframe_izquierdo
        .join(
            dataframe_derecho
            , on = criterio_on
            , how = how
        )
    )
    return join_df
def plot_bar_char(df,categories,values):
    pandf = convert_to_pandas(df)
    pandf_total = pandf.groupby(categories)[values].sum()
    pandf_total.plot(kind="bar", legend=False, color="skyblue")
    plt.xlabel(categories)
    plt.ylabel(values)
    plt.show()

def plot_line_char(df,categories,values):
    pandf = convert_to_pandas(df)
    pandf_total = pandf.groupby(categories)[values].sum()
    pandf_total.plot(kind="line", legend=False, color="skyblue")
    plt.xlabel(categories)
    plt.ylabel(values)
    plt.show()

def plot_histogram(df,values):
    pandf = convert_to_pandas(df)
  
    pandf.plot(kind="hist", legend=False, color="skyblue")
    plt.xlabel('sales_amount')
    plt.ylabel("frecuencia")
    plt.show()

def plot_correlation_matrixt(df):
    pandas = convert_to_pandas(df)
    pandas_num = pandas.select_dtypes(include=['number'])
    correlation_matrix = pandas_num.corr()
    return correlation_matrix


