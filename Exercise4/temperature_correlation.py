import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi


""" _Description_
    Calculates the distance between one city and every station.
    Adapted from the distance function created in Exercise 3
    Returns a cmodified opy of the stations dataframe with an added series 'distance'.
"""
def distance(city, stations):
    r = 6371  # km
    p = pi / 180
    points = city
    df = stations.copy()
    df['distance'] = 0.5 - np.cos((df['latitude']-points['latitude'])*p)/2 + np.cos(points['latitude']*p) * np.cos(df['latitude']*p) * (1-np.cos((df['longitude']-points['longitude'])*p))/2
    df['distance'] = (2 * r * np.arcsin(np.sqrt(df['distance'])))
    return df

""" _Description_
    Returns the best value you can find for avg_tmax for that one city, from the list of all weather staions
    Hint: use distance and numpy.argmin
    Returns the avg_tmax for the station with the smallest distance to the city
"""
def best_tmax(city, stations):
    df = distance(city, stations)
    index = np.argmin(df['distance'])
    station = df.loc[index]
    best_avg_tmax = station['avg_tmax']

    return best_avg_tmax

def main():

    stations_file = sys.argv[1]
    city_data_file = sys.argv[2]
    output_file = sys.argv[3]
    
    stations = pd.read_json(stations_file, lines=True)
    cities = pd.read_csv(city_data_file)[['name', 'population', 'area', 'latitude', 'longitude']]
    stations['avg_tmax'] = stations['avg_tmax']/10  # convert temperature to degrees Celsius
    cities.dropna(subset=['population','area'], inplace=True)  # remove NaN from population and area (can't calculate population density)
    print(cities)
    cities['area'] = cities['area'] / (10**6)  # convert m^2 to km^2 by dividing by 10^6
    print(cities)
    cities = cities[cities['area'] <= 10000]  # exclude cities with area greater than 10000 km^2
    print(cities)
    cities['avg_tmax'] = cities.apply(lambda city: best_tmax(city, stations), axis=1)
    cities['population_density'] = cities['population'] / cities['area']
    
   
    plt.figure(figsize=(10,5))
    plt.scatter(cities['avg_tmax'],cities['population_density'])
    plt.xlabel('Avg Max Temperature (\u00b0C)')
    plt.ylabel('Population Density (people/km\u00b2)')
    plt.title('Average Maximum Temperature vs. Population Density')
    # plt.savefig('test.png')
    plt.savefig(output_file)
    
    return

if __name__ == '__main__':
    main()