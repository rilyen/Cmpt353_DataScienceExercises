import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)


def main():
    reddit_counts = sys.argv[1]
    # read the file as a json object per line
    counts = pd.read_json(reddit_counts, lines=True)  
    # only /r/canada subreddit
    canada = pd.DataFrame(columns=['date','subreddit', 'comment_count'])
    canada = counts.loc[counts['subreddit'] == 'canada']  
    # only values from years 2012 and 2013
    canada = canada[(canada['date'] >= '2012-01-01') & (canada['date'] <= "2013-12-31")]  
    canada['date'] = pd.to_datetime(canada['date'])
    # separate the weekends and weekdays
    canada_weekend = pd.DataFrame(columns=['date','subreddot','comment_count'])
    canada_weekday = pd.DataFrame(columns=['date','subreddot','comment_count'])
    canada_weekend = canada.loc[canada['date'].dt.weekday >= 5] #5-6 = sat-sun
    canada_weekday = canada.loc[canada['date'].dt.weekday <= 4] #0-4 = mon-fri

    # Student's T-Test
    # H0: The means are equal between weekdays and weekends
    # Ha: The means are not equal between weekdays and weekends
    # small p-value, so we reject H0 and accept Ha, so the means are not equal
    # we need to check that the data is noramlly-distributed and have equal variances
    initial_ttest = stats.ttest_ind(canada_weekday['comment_count'], canada_weekend['comment_count'])
    # Normal-test
    # Ho: The data is normally distributed
    # Ha: The data is not normally distributed
    # both have small p, so we reject H0 and accept Ha
    # The data is not normally-distributed
    initial_weekday_normality = stats.normaltest(canada_weekday['comment_count'])
    initial_weekend_normality = stats.normaltest(canada_weekend['comment_count'])
    # Variance-test (Levene's test)
    # H0: The two samples have equal variance
    # Ha: The two samples do not have equal variance
    # small p, so we reject the H0 and accept Ha
    # The two samples do not have equal variance
    initial_levene = stats.levene(canada_weekday['comment_count'], canada_weekend['comment_count'])
    # Since the data is not normally distributed, and the two data sets do not have equal variances:
    #   we cannot draw a conclusion from the T-test
    
    # Fix 1: transforming data might save us
    # observe that the histogram is skewed
    plt.clf()
    bins_original = np.linspace(canada['comment_count'].min(), canada['comment_count'].max())
    plt.hist(canada_weekend['comment_count'], bins_original, alpha=0.5, label='weekend comment count')
    plt.hist(canada_weekday['comment_count'], bins_original, alpha=0.5, label='weekday comment count')
    plt.legend(loc='upper right')
    plt.savefig("histogram for reddit comment counts in canada")
    # Transform the count so the data doesn't fail the normality test: 
    #   np.log, np.exp, np.sqrt, counts**2
    # We choose np.sqrt as coming cosest to normal distributions
    # Check the normality test
    # weekday count p-value is smallish, so probably not normally distributed
    # weekend count p-value is large, so is normally distributed
    # levene p-value is large, so probably has equal variance
    sqrt_weekday = np.sqrt(canada_weekday['comment_count'])
    sqrt_weekend = np.sqrt(canada_weekend['comment_count'])
    transformed_weekday_normality = stats.normaltest(sqrt_weekday)
    transformed_weekend_normality = stats.normaltest(sqrt_weekend)
    transformed_levene = stats.levene(sqrt_weekday, sqrt_weekend)
    
    # Fix 2: the Central Limit Theorem might save us
    # The CLT says: If our numbers are large enough, and we look at sample means, then the results should be normal
    # Try: Combine all weekdays and weekend days from each year/week pair and take the mean of their (non-transformed) counts
    # Check these values for normality and equal variance. Apply a T-test if it makes sense to do so.
    weekly_weekday_counts = canada_weekday.copy()
    weekly_weekday_counts['year'] = weekly_weekday_counts['date'].dt.isocalendar()['year']
    weekly_weekday_counts['week'] = weekly_weekday_counts['date'].dt.isocalendar()['week']
    weekly_weekday_counts = weekly_weekday_counts.groupby(['year','week']).agg({'comment_count':'mean'}).reset_index()
    print(weekly_weekday_counts)
    weekly_weekend_counts = canada_weekend.copy()
    weekly_weekend_counts['year'] = weekly_weekend_counts['date'].dt.isocalendar()['year']
    weekly_weekend_counts['week'] = weekly_weekend_counts['date'].dt.isocalendar()['week']
    weekly_weekend_counts = weekly_weekend_counts.groupby(['year','week']).agg({'comment_count':'mean'}).reset_index()
    print(weekly_weekend_counts)
    # Normal-test
    # p-values > 0.05, so we do not reject H0 and have normal distribution
    weekly_weekday_normality = stats.normaltest(weekly_weekday_counts['comment_count'])
    weekly_weekend_normality = stats.normaltest(weekly_weekend_counts['comment_count'])
    # Variance-test (Levene's)
    # p-values > 0.05, so we do not reject H0 and have equal variance
    weekly_levene = stats.levene(weekly_weekday_counts['comment_count'], weekly_weekend_counts['comment_count'])
    # T-test since we have normal distribution and equal variance
    # p-value < 0.05, so we reject H0 and conclude that the weekly comment count means are different
    weekly_ttest = stats.ttest_ind(weekly_weekday_counts['comment_count'], weekly_weekend_counts['comment_count'])
    
    # Fix 3: a non-parametric test might save us (The Mann-Whiteney U-test)
    # H0: There is no difference (in terms of central tendency) between the two groups in the population
    # Ha: There is a difference (in terms of central tendency) between the two groups in the population
    # p-value < 0.05, so we reject H0 and conclude that it is not equally likely that the larger number of comments occur on weekends vs weekdays
    utest = stats.mannwhitneyu(weekly_weekday_counts['comment_count'], weekly_weekend_counts['comment_count'], alternative='two-sided')
    
    
    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_ttest.pvalue,
        initial_weekday_normality_p=initial_weekday_normality.pvalue,
        initial_weekend_normality_p=initial_weekend_normality.pvalue,
        initial_levene_p=initial_levene.pvalue,
        transformed_weekday_normality_p=transformed_weekday_normality.pvalue,
        transformed_weekend_normality_p=transformed_weekend_normality.pvalue,
        transformed_levene_p=transformed_levene.pvalue,
        weekly_weekday_normality_p=weekly_weekday_normality.pvalue,
        weekly_weekend_normality_p=weekly_weekend_normality.pvalue,
        weekly_levene_p=weekly_levene.pvalue,
        weekly_ttest_p=weekly_ttest.pvalue,
        utest_p=utest.pvalue,
    ))

    # check if there are more comment counts on weekdays or weekends by creating plots
    plt.figure(figsize=(10, 6))
    # Plot weekday
    plt.plot(weekly_weekday_counts.index, weekly_weekday_counts['comment_count'], linestyle='-', color='b', label='Weekday Comment Count')
    # Plot weekend
    plt.plot(weekly_weekend_counts.index, weekly_weekend_counts['comment_count'], linestyle='-', color='r', label='Weekend Comment Count')
    plt.xlabel('Time')
    plt.ylabel('Reddit Comment Counts')
    plt.title('Weekday and Weekend Reddit Comment Counts')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Weekday and Weekend Reddit Comment Counts')

if __name__ == '__main__':
    main()
