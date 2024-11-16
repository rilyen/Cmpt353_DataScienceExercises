import numpy as np

def main():
    data = np.load('monthdata.npz')
    totals = data['totals']
    counts = data['counts']

    cities_total = np.sum(totals,axis=1) # sum each row of the totals matrix to get total precipitation over the year for each city
    min_index = np.argmin(cities_total) # get the index of the city with lowest total precipitation
    print("Row with lowest total precipitation:")
    print(min_index)

    month_total = np.sum(totals,axis=0) # sum the columns to get total precipitation of all cities over the month
    month_count = np.sum(counts,axis=0) # sum observations over the month
    month_avg = month_total / month_count
    print("Average precipitation in each month:")
    print(month_avg)

    # find the average precipitation for each city
    # daily precipitation averaged over the month
    cities_count = np.sum(counts,axis=1)        # sum each row to get total observations of each city over the yeaer
    cities_avg = cities_total/cities_count      # (precipitation/observations) = avg precipitation
    print("Average precipitation in each city:")
    print(cities_avg)

    print("Quarterly precipitation totals:")
    num_stations = len(totals)      # get the number of stations/cities
    quarterly_totals = totals
    quarterly_totals = np.reshape(quarterly_totals,(4*num_stations,3))
    quarterly_totals = np.sum(quarterly_totals,axis=1)
    quarterly_totals = np.reshape(quarterly_totals,(num_stations,4))
    print(quarterly_totals)

if __name__ == '__main__':
    main()



