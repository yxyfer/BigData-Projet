from pyspark.sql import SparkSession
import pandas as pd

from pyspark.sql.functions import col, isnan, when, count
from pyspark.sql.functions import mean
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    DateType,
    StructType,
    StructField,
)
from pyspark.sql.functions import desc
import pyspark.sql.functions as func


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark_application_name = "WannaFlop_Project"
spark = (SparkSession.builder.appName(spark_application_name).getOrCreate())

class Stock(object):
    def __init__(self, file_path, header=False, delimiter=";", schema=None):
        self.file_path = file_path
        self.header = header
        self.delimiter = delimiter
        self.schema = schema

        self.df = self._load_df()

        self.explore = self.Explore(self)
        self.analysis = self.Analysis(self)

    def _get_num_cols(self):
        num_cols = [
            f.name
            for f in self.df.schema.fields
            if isinstance(f.dataType, DoubleType) or isinstance(f.dataType, IntegerType)
        ]

        return num_cols

    def _get_rounded_df(self):
        rounded_df = self.df
        dbl_cols = self._get_num_cols()
        for col in dbl_cols:
            rounded_df = rounded_df.withColumn(col, func.round("high"))

        return rounded_df

    def _get_periodicity(self):
        self.df["data"][0]

    def _handle_csv(self):
        """
        @description: Read the csv file and return a Spark DataFrame

        @arg csv_file_path: Path to the csv file
        @arg header: boolean whether to load a header or not
        @arg delimiter: which delimiter to use by default
        """
        return (
            spark.read.option("inferSchema", "true")
            .option("nullValue", "null")
            .csv(
                self.file_path,
                sep=self.delimiter,
                schema=self.schema,
                header=self.header,
            )
        )

    def _handle_json(self):
        return spark.read.json(self.file_path)

    def _load_df(self):
        ####### ADD TRY CATCH #####
        extension = self.file_path.split(".")[-1]

        df = None
        if extension == "json":
            df = self._handle_json()
        elif extension == "csv":
            df = self._handle_csv()

        return df

    class Explore:
        """
        Need to add:
            - Correlation
            - period (already done just need to add it)
        """
        def __init__(self, stock):
            self.stock = stock

        def __repr__(self):
            return f"""
            {self._print_schema()}\n
            {self.get_df_abstract()}\n
            {self._nb_rows()}\n
            {self.get_stats()}\n
            {self.get_missing()}"""

        def get_missing(self):
            print("Missing Data per column:")
            self._count_missing().show()

        def get_df_abstract(self):
            rounded_df = self.stock._get_rounded_df()

            # First 40 rows
            print("First 40 rows:")
            rounded_df.show(40)

            # Last 40 rows
            print("Last 40 rows:")
            rounded_df = rounded_df.withColumn("index", monotonically_increasing_id())
            rounded_df.orderBy(desc("index")).drop("index").show(40)

        def get_stats(self):
            rounded_df = self.stock._get_rounded_df()
            summary = rounded_df.summary()
            cols = self.stock._get_num_cols()
            for col in cols:
                summary = summary.withColumn(col, func.round("high"))
            print("Stock Stats:")
            summary.show()

        def _nb_rows(self):
            # Number of total rows
            print("Number of rows: " + str(self.stock.df.count()) + "\n")

        def _print_schema(self):
            self.stock.df.printSchema()

        def _count_missing(self):
            cols = self.stock.df.columns
            cols.remove('Date')
            return self.stock.df.select(
                [
                    count(when(isnan(c) | col(c).isNull(), c)).alias(c)
                    for c in cols
                ]
            )
            #.show()


    class Analysis:
        def __init__(self, stock):
            self.stock = stock
            self.df = stock.df

        def get_oc_avg(self, fun):
            close = self._compute_avg(self.df, "Close", fun)
            opening = self._compute_avg(self.df, "Open", fun)

            return close.join(
                opening, opening.Open_new_time == close.Close_new_time, "inner"
            ).orderBy("Close_new_time").select(
                close.Close_new_time, close.Close_mean, opening.Open_mean
            )

        def get_price_change(self, period=None):
            df = self.df
            if period:
                df= self.get_oc_avg(period)
           
            return  df.withColumn('diff', ( df['Close_mean'] - df['Open_mean'] ))

        def _compute_avg(self, df, col, fun):
            return df.groupBy(fun("Date").alias(col + "_new_time")).agg(
                mean(col).alias(col + "_mean")
            )
