import pandas as pd;

def main():
    totals = pd.read_csv('totals.csv').set_index(keys=['name'])
    counts = pd.read_csv('counts.csv').set_index(keys=['name'])    

    # Find city with the lowest total precipitation over the year
    totals_cities = totals.sum(axis=1)  # sum the rows to get total precipitation for each city
    lowest_total_precipitation = totals_cities.idxmin() # get index of min total
    print("City with lowest total precipitation:")
    print(lowest_total_precipitation)

    # Find the average precipitation in each month
    monthly_counts = counts.sum(axis=0) # total counts for each month
    monthly_precipitation = totals.sum(axis=0) / monthly_counts # average precipitation per month
    print("Average precipitation in each month:")
    print(monthly_precipitation)

    # Find the average precipitation in each city
    counts_cities = counts.sum(axis=1)    # counts per each for each city
    avg_precipitation_city = totals_cities / counts_cities
    print("Average precipitation in each city:")
    print(avg_precipitation_city)    

if __name__ == '__main__':
    main()
