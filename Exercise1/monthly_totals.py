import numpy as np
import pandas as pd


def get_precip_data():
    return pd.read_csv('precipitation.csv', parse_dates=['date']) #date column contains datetime objects (instead of strings)


def date_to_month(d):
    # You may need to modify this function, depending on your data types.

    #return '%04i-%02i' % (d.year, d.month)
    return d.to_period('M')


def pivot_months_pandas(data):
    """
    Create monthly precipitation totals for each station in the data set.
    This should use Pandas methods to manipulate the data.
    """
    # 1. Add a column 'month' that contains the results of applying the date_to_month function to the existing 'date' column. 
    # [You may have to modify date_to_month slightly, depending how your data types work out. ]
    data['month'] = date_to_month(data['date'].dt)

    # 2. Use the Pandas groupby method to aggregate over the name and month columns. 
    # Sum each of the aggregated values to get the total. 
    # Hint: grouped_data.aggregate({'precipitation': 'sum'}).reset_index()
    cities_totals = data.groupby(['name','month'])['precipitation'].sum().reset_index()

    # 3. Use the Pandas pivot method to create a row for each station (name) and column for each month.
    monthly = cities_totals.pivot(index='name',columns='month',values='precipitation')

    # 4. Repeat with the 'count' aggregation to get the count of observations.
    cities_counts = data.groupby(['name','month']).size().to_frame('size').reset_index()
    counts = cities_counts.pivot(index='name',columns='month',values='size')

    return monthly, counts


def pivot_months_loops(data):
    """
    Create monthly precipitation totals for each station in the data set.
    
    This does it the hard way: using Pandas as a dumb data store, and iterating in Python.
    """
    # Find all stations and months in the data set.
    stations = set()
    months = set()
    for i,r in data.iterrows():
        stations.add(r['name'])
        m = date_to_month(r['date'])
        months.add(m)

    # Aggregate into dictionaries so we can look up later.
    stations = sorted(list(stations))
    row_to_station = dict(enumerate(stations))
    station_to_row = {s: i for i,s in row_to_station.items()}
    
    months = sorted(list(months))
    col_to_month = dict(enumerate(months))
    month_to_col = {m: i for i,m in col_to_month.items()}

    # Create arrays for the data, and fill them.
    precip_total = np.zeros((len(row_to_station), 12), dtype=np.uint)
    obs_count = np.zeros((len(row_to_station), 12), dtype=np.uint)

    for _, row in data.iterrows():
        m = date_to_month(row['date'])
        r = station_to_row[row['name']]
        c = month_to_col[m]

        precip_total[r, c] += row['precipitation']
        obs_count[r, c] += 1

    # Build the DataFrames we needed all along (tidying up the index names while we're at it).
    totals = pd.DataFrame(
        data=precip_total,
        index=stations,
        columns=months,
    )
    totals.index.name = 'name'
    totals.columns.name = 'month'
    
    counts = pd.DataFrame(
        data=obs_count,
        index=stations,
        columns=months,
    )
    counts.index.name = 'name'
    counts.columns.name = 'month'
    
    return totals, counts


def main():
    data = get_precip_data()
    totals, counts = pivot_months_pandas(data)
    totals.to_csv('totals.csv')
    counts.to_csv('counts.csv')
    np.savez('monthdata.npz',totals=totals.values,counts=counts.values)

if __name__ == '__main__':
    main()
