import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

data_qs1 = data['qs1']
data_qs2 = data['qs2']
data_qs3 = data['qs3']
data_qs4 = data['qs4']
data_qs5 = data['qs5']
data_merge1 = data['merge1']
data_partition_sort = data['partition_sort']

# Use ANOVA (Analysis of variance) to determine if the means of any groups differ.
# Similar to a T-test, but for >2 groups
# Since we have > 40 data points and it's likely normally distributed, by the CLT we proceed...
anova = stats.f_oneway(data_qs1, data_qs2, data_qs3, data_qs4, data_qs5, data_merge1, data_partition_sort)
print(anova.pvalue)
# p-value near 0, which is < 0.05, so there are likely different means among the different sorts

# Post-Hoc Analysis
# Since we had significance in an ANOVA, we will do post hoc analysis (pairwise comparisons between each variable)
# We shall use Tukey's HSD
x_data = pd.DataFrame({'qs1':data_qs1, 'qs2':data_qs2, 'qs3':data_qs3, 'qs4':data_qs4, 'qs5':data_qs5, 'merge1':data_merge1, 'partition_sort':data_partition_sort})
x_melt = pd.melt(x_data)
post_hoc = pairwise_tukeyhsd(
    x_melt['value'], x_melt['variable'],
    alpha=0.05
)
print("Summary:")
print(post_hoc)
print("Reject null hypothesis: ")
print(post_hoc.reject)

fig = post_hoc.plot_simultaneous()
fig.savefig('result_plot.png')
# from the plot we can see visualize where the means are different


