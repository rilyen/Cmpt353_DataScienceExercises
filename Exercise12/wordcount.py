from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode, lower, count

import string, re, sys

def main(in_directory, out_directory):

    # 1. Read lines from the files with spark.read.text
    lines = spark.read.text(in_directory)  # each line in the text file will be a separate row in the df and it will be stored in a single column named 'value'

    # 2. Split the lines into words with the regular expression below. Use the split
    # and explode functions. Normalize all of the strings to lower-case (so "word" and "Word"
    # are not counted seaprately.)
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)  # regex that matches spaces and/or punctuation
    
    # functions.split() is used to split the values of a column in a df into multiple columns based on a specified delimiter
    #   the result of the split operation is an array of strings in the same column ('values')
    # functions.explode() takes a column of arrays (output of split) and creates a new row for each element in the array
    #   we are taking the line of text in each row and putting each word in its own row
    # lines = split(lines['value'], wordbreak)  # array of words in the 'value' column
    # lines = explode(lines).alias('word')  # each word has its own row
    # words = lines.select('word')
    words = lines.select(explode(split(lines['value'], wordbreak)).alias('word'))

    words = words.select(lower(words['word']).alias('word'))  # normalize words to lowercase
    #words.show()

    # 3. Count the number of times each word occurs
    words = words.groupBy('word').agg(count('*').alias('count'))
    #words.show()

    # 4. Sort by decreasing count (i.e. frequent words first) and alphabetically if there's a tie
    words = words.sort(['count', 'word'], ascending=[False, True])
    #words.show()

    # 5. keep non-empty words (remove empty strings)
    words = words.filter(words['word'] != '')  

    # 6. Write results as CSV files with the word in the first column and count in the second
    #   (uncompressed: they aren't big enough to worry about)
    words.write.mode('overwrite').csv(out_directory)

    return

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    spark = SparkSession.builder.appName('Word Count').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)