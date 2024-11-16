import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather ETL').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

observation_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.StringType()),
    types.StructField('observation', types.StringType()),
    types.StructField('value', types.IntegerType()),
    types.StructField('mflag', types.StringType()),
    types.StructField('qflag', types.StringType()),
    types.StructField('sflag', types.StringType()),
    types.StructField('obstime', types.StringType()),
])


def main(in_directory, out_directory):
    # 1. Read the input directory of .csv.gz files.
    weather = spark.read.csv(in_directory, schema=observation_schema)
    weather.show()
    # TODO: finish here.
    # 2. Keep only the records we care about:
    #   a. field qflag (quality flag) is null
    #   b. the station starts with 'CA'
    #   c. the observation is 'TMAX'
    weather = weather.filter((weather['qflag'].isNull() & (weather['station'].startswith('CA')) & (weather['observation']=='TMAX')))
    # 3. Divide the temperature by 10 so it's actually in Celsius, and call the resulting column tmax
    # 4. Keep only the columns station, date, and tmax
    cleaned_data = weather.select(
        weather['station'],
        weather['date'],
        (weather['value']/10).alias('tmax')
    )
    cleaned_data.show()
    # 5. Write the result as a directory of JSON files GZIP compressed (in the Spark one-JSON-object-per-line way).
    cleaned_data.write.json(out_directory, compression='gzip', mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
