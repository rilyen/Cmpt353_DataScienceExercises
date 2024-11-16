import sys
from pyspark.sql import SparkSession, functions, types
import re

spark = SparkSession.builder.appName('wikipedia popular').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+


wikipedia_schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('times_requested', types.LongType()),
    types.StructField('total_bytes', types.LongType()),
])

def extract_day_hour(path):
    return re.search('\d{8}-\d{2}', path).group(0)

def main(in_directory, out_directory):
    # create the dataframe using a schema and column containing the filename
    wikipedia = spark.read.csv(in_directory, schema=wikipedia_schema, sep=" ").withColumn('filename', functions.input_file_name())
    # (1) English Wikipedia pages (i.e. language is "en") only
    english_only = wikipedia['language'] == 'en'
    wikipedia = wikipedia.filter(english_only)
    # (2) Exclude the front page 'Main_Page'
    exclude_main_page = wikipedia['title'] != 'Main_Page'
    wikipedia = wikipedia.filter(exclude_main_page)
    # (3) Exclude titles starting with 'Special:'
    exclude_special = ~wikipedia['title'].startswith('Special')
    wikipedia = wikipedia.filter(exclude_special)

    # Find the largest number of page views in each hour
    path_to_hour = functions.udf(extract_day_hour, returnType=types.StringType())
    # Join that back to the collection of all page counts
    wikipedia = wikipedia.withColumn('hour', path_to_hour(wikipedia['filename']))

    # Cache
    wikipedia = wikipedia.cache()

    # Keep only those with the count == max(count) for that hour. If there's a tie we keep both
    wikipedia_max = wikipedia.groupby('hour').agg(functions.max(wikipedia['times_requested']).alias('times_requested'))
    # Sort results by date/hour (and page name if there's a tie)

    wikipedia_max = wikipedia_max.join(wikipedia, on=['times_requested','hour'], how='left')
    wikipedia_max = wikipedia_max.select('hour','title','times_requested')  # relevant columns for csv
    wikipedia_max = wikipedia_max.sort('hour', 'title', ascending=True)

    # Output as a CSV
    wikipedia_max.write.csv(out_directory + '-most_accessed_wikipedia_page', mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
