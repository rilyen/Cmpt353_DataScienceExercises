import sys
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from math import pi
from pykalman import KalmanFilter


""" Input: gpx file
    Return: dataframe
    Description: Reads and parses input_gpx into a pandas dataframe
"""
def get_data(input_gpx):
    points = pd.DataFrame(columns=['datetime', 'lat', 'lon'])
    tree = ET.parse(input_gpx)  # parse XML into an element tree
    root = tree.getroot()  # <gpx xmlns="http://www.topografix.com/GPX/1/0">
    data = root[0][0]  # <trkseg>
    i = 0
    for child in data:  # <trkpt>
        lat = float(child.attrib['lat'])
        lon = float(child.attrib['lon'])
        datetime = pd.to_datetime(child[0].text, utc=True, format="%Y-%m-%dT%H:%M:%SZ")
        points.loc[i] = [datetime, lat, lon]
        i += 1
    return points

    
def distance(points):
    """
    Input: dataframe with 'lat' and 'lon' columns
    Return: float
    Description: returns the distance (in metres) between the latitude/longitude points using haversine formula
    Calculates the difference of each point, then sums those differences together to get the total distance.
    Reference: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
    """
    df = points.shift(periods=-1)  # get the lat/lon of next point on same row
    r = 6371 # km
    p = pi / 180
    df['distance'] = 0.5 - np.cos((df['lat']-points['lat'])*p)/2 + np.cos(points['lat']*p) * np.cos(df['lat']*p) * (1-np.cos((df['lon']-points['lon'])*p))/2
    df = df[:-1]
    df['distance'] = (2 * r * np.arcsin(np.sqrt(df['distance'])))
    a = df['distance'].sum()
    return a * 1000  # convert to m

def smooth(points):
    """
    Input: dataframe with 'datetime', 'lat', and 'lon' columns
    Output: dataframe with 'lat' and 'lon' columns
    Description: Use Kalman Filtering to smooth the data
    """
    print(points)
    kalman_data = points[['lat', 'lon', 'Bx', 'By']]
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([0.5, 0.5, 0.05, 0.05]) ** 2
    transition_covariance =  np.diag([0.4, 0.4, 0.5, 0.5]) ** 2
    transition = [[1, 0,   5*(1e-7), 34*(1e-7)],
                  [0, 1, -49*(1e-7),  9*(1e-7)], 
                  [0, 0,          1,         0], 
                  [0, 0,          0,         1]]
    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )
    kalman_smoothed, _ = kf.smooth(kalman_data)
    df = pd.DataFrame(data=kalman_smoothed, columns=['lat', 'lon', 'Bx', 'By'])
    return df

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.7f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.7f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def main():
    input_gpx = sys.argv[1]
    input_csv = sys.argv[2]
    
    points = get_data(input_gpx).set_index('datetime')
    sensor_data = pd.read_csv(input_csv, parse_dates=['datetime']).set_index('datetime')
    points['Bx'] = sensor_data['Bx']
    points['By'] = sensor_data['By']

    dist = distance(points)
    print(f'Unfiltered distance: {dist:.2f}')

    smoothed_points = smooth(points)
    smoothed_dist = distance(smoothed_points)
    print(f'Filtered distance: {smoothed_dist:.2f}')

    output_gpx(smoothed_points, 'out.gpx')
    return


if __name__ == '__main__':
    main()
