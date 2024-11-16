import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types, Row
import re
import math


line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred.
    Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        # TODO
        h = m.group(1)
        b = int(m.group(2))
        return Row(hostname = h, bytes = b)
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    # 1. Get the data out of the files into a DataFrame where you have 
    # the hostname and number of bytes for each request. 
    # Do this using an RDD operation
    row = log_lines.map(line_to_row)
    row = row.filter(not_none)
    return row


def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    # TODO: calculate r.
    # 2. Group by hostname; get the number of requests and sum of bytes transferred, to form a data point
    hostnames = logs.groupBy('hostname')
    count_requests = hostnames.count()
    sum_request_bytes = hostnames.sum('bytes')
    data_points = sum_request_bytes.join(count_requests, on='hostname', how='left')
    # print(data.head(6))
    # 3. Produce six values. Add these to get the six sums.
    data_points = data_points.withColumn('xi', data_points['count'])
    data_points = data_points.withColumn('yi', data_points['sum(bytes)'])
    data_points = data_points.withColumn('xiyi', data_points['xi'] * data_points['yi'])
    data_points = data_points.withColumn('xi^2', data_points['xi'] * data_points['xi'])
    data_points = data_points.withColumn('yi^2', data_points['yi'] * data_points['yi'])

    data_points = data_points.select('xi', 'yi', 'xiyi', 'xi^2', 'yi^2')
    n = data_points.count()  # the 6th value
     
    data_points = data_points.groupBy()
    six_sums = data_points.sum()

    xi_sum =  six_sums.first()[0]
    yi_sum =  six_sums.first()[1]
    xiyi_sum =  six_sums.first()[2]
    xi_2_sum =  six_sums.first()[3]
    yi_2_sum =  six_sums.first()[4]

    # 4. Calculate the final value of r.

    numer = n*xiyi_sum - xi_sum*yi_sum
    denom_1 = math.sqrt((n * xi_2_sum) - (xi_sum * xi_sum))
    denom_2 = math.sqrt( (n*yi_2_sum) - (yi_sum*yi_sum) ) 
    denom = denom_1 * denom_2

    r = numer / denom # TODO: it isn't zero.
    print(f"r = {r}\nr^2 = {r*r}")
    # Built-in function should get the same results.
    # print(totals.corr('count', 'bytes'))


if __name__=='__main__':
    in_directory = sys.argv[1]
    spark = SparkSession.builder.appName('correlate logs').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory)
