import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)
    # comments.show()
    # TODO
    # 1. Calculate the average score for each subreddit, as before.
    subreddit = comments.groupBy('subreddit')
    averages = subreddit.agg(functions.avg(comments['score']).alias('average'))  # outputs a dataframe with the average score for each subreddit
    averages = averages.cache()

    # 2. Exclude any subreddits with average score ≤0.
    averages = averages.filter(averages['average'] > 0)

    # 3. Join the average score to the collection of all comments. Divide to get the relative score.
    averages = averages.join(comments, on='subreddit', how='left')
    averages = averages.withColumn('rel_score', (averages['score'] / averages['average']))  # compute the relative score for each comment
    averages_max_rel_score = averages.groupBy('subreddit').agg(functions.max(averages['rel_score']).alias('rel_score'))  # outputs a dataframe with the highest relative score for each subreddit
    
    # 4. Determine the max relative score for each subreddit.
    max_rel_score = comments.groupBy('subreddit').agg(functions.max(comments['score']).alias('score'))  # outputs a dataframe with the max score for each subreddit
    max_rel_score = max_rel_score.cache()

    # 5. Join again to get the best comment on each subreddit: we need this step to get the author.
    max_rel_score = max_rel_score.join(comments, on=['score', 'subreddit'], how='left')

    # Output should be uncompressed JSON with the fields subreddit, author, and rel_score
    best_author = max_rel_score.join(averages_max_rel_score, on='subreddit', how='left')
    best_author = best_author.select('subreddit', 'author', 'rel_score')
    best_author = best_author.sort('subreddit', 'author', ascending=True, key=lambda x: x.str.lower())
    best_author.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    spark = SparkSession.builder.appName('Reddit Relative Scores').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)
