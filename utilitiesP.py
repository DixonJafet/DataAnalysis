

from pyspark.sql import SparkSession 
from pyspark.sql.functions import * 
from pyspark.sql.types import *
from pyspark.ml import Pipeline 
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.feature import StringIndexer, OneHotEncoder, MinMaxScaler
from pyspark.ml.classification import NaiveBayes 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

def read_data(spark,address,schema):
    df = spark.read.csv(path=address,schema=schema)
    df.printSchema()
    return df

def describe_dataframe(df):
    return df.describe()

def count_nulls(df):
    counted_null_data = ( df
    .select( # se usa la funcion select para seleccionar la columna
        [
            count( # Se usa count en conjunto con when para validar si la columna tiene valores nulos
                when(
                    isnull(c) #Funcion para validar nulos
                    ,c
                )
            ).alias(c)
            for c in df.columns # A esto se le conoce como list comprehension, es hacer un for en una lista
        ]
    )
    )
    return counted_null_data


def null_handling_input(df,col_to_input,value_to_input):
    new_df = df.fillna({col_to_input: value_to_input})
    return new_df


def null_handling_remove(df,col_to_input):
    new_df =  df.dropna(subset=col_to_input)
    return new_df

## E 
def join_dataframe(dataframe_izquierdo:DataFrame, dataframe_derecho:DataFrame, criterio_on:list, how:str)->DataFrame:

    join_df = (
    dataframe_izquierdo
        .join(
            dataframe_derecho
            , on = criterio_on
            , how = how
        )
    )
    return join_df

def encoding_categories(df,col_names):
    indexers = [StringIndexer(inputCol=col_name, outputCol=f"{col_name}Index") for col_name in col_names]
    encoders = [OneHotEncoder(inputCol=f"{col_name}Index", outputCol=f"{col_name}_Encoded")  for col_name in col_names ]

    pipeline = Pipeline(stages=indexers+encoders)
    df_encoded = pipeline.fit(df).transform(df)
    to_drop=  [f"{col_name}Index" for col_name in col_names]
    df_encoded = df_encoded.drop(*to_drop)
    return df_encoded


def normalize_numerical_data(df):
  #  assembler = VectorAssembler(inputCols=col_names, outputCol="features")
   # df_assembled = assembler.transform(df)
    scaler = MinMaxScaler(inputCol="features_vec", outputCol="features")
    df_scaled = scaler.fit(df).transform(df)
 #   df_scaled = df_scaled.drop("features")
    return df_scaled

def drop_string_cols(df,type):
    for column in df.schema.fields:
        if column.dataType.typeName() == type:
            df = df.drop(column.name)
    return df
def split_dataframe(df):
    df.randomSplit([0.7,0.3], 40) 
    splits = df.randomSplit([0.7,0.3], 40) 
    return splits

def vectorize_columns(df:DataFrame, excluded_variable:str)->DataFrame:
    col_names = [column.name for column in df.schema.fields if column.name != excluded_variable]# and  (column.dataType.typeName() == "integer" or column.dataType.typeName() == "float")]
    assembler = VectorAssembler(inputCols=col_names, outputCol="features_vec")
    df_assembled = assembler.transform(df)
    print("Save")
    return df_assembled
#scaled_features   
#WeightPerUnit
def get_string_cols(df):
    str_list = []
    for column in df.schema.fields:
        if column.dataType.typeName() == "string":
            str_list.append(column.name)
    return str_list


def generate_ml_model(train_df,test_df,ml_model):

    lasso_model = ml_model.fit(train_df)
    predictions_lasso = lasso_model.transform(test_df)
    return predictions_lasso

def evaluate_ml_results(evaluator,model):
    lasso_mse= evaluator.evaluate(model)
    print(f"Error cuadrático medio es de: {lasso_mse}")

