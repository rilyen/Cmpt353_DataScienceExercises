import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    # no header row, using page name as index in dataframe
    f1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1, names = ['lang','page','views','bytes'])
    f2 = pd.read_csv(filename2, sep=' ', header=None, index_col=1, names = ['lang','page','views','bytes'])
    
    plt.figure(figsize=(10,5))
    
    # Plot 1: Distribution of Views
    # only a few pages have many views, then it sharply declines
    f1_sort = f1.sort_values(by=['views'],ascending=False) # higher ranked page will have more views, so sort by descending views
    # print(f1_sort)
    ypoints = f1_sort['views'].values       # get number of views for y-axis
    plt.subplot(1,2,1)
    plt.plot(ypoints)                       # automatically plots against 0 to n-1 (the rank) 
    plt.title('Popularity Distribution')
    plt.xlabel('Rank')
    plt.ylabel('Views')
    

    # Plot 2: Hourly Views
    
    f1_sort['views2'] = f2['views']         # add views from f2 into the same dataframe as f1
    # print(f1_sort)
    plt.subplot(1,2,2)
    plt.scatter(f1_sort['views'].values,f1_sort['views2'].values,color='blue',alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Hourly Correlation')
    plt.xlabel('Hour 1 views')
    plt.ylabel('Hour 2 views')
    
    # Final Output
    plt.savefig('wikipedia.png')
    
    return

if __name__ == '__main__':
    main()