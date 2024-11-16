import os
import pathlib
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation, parse
    xmlns = 'http://www.topografix.com/GPX/1/0'
    
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.10f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.10f' % (pt['lon']))
        time = doc.createElement('time')
        time.appendChild(doc.createTextNode(pt['datetime'].strftime("%Y-%m-%dT%H:%M:%SZ")))
        trkpt.appendChild(time)
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    doc.documentElement.setAttribute('xmlns', xmlns)

    with open(output_filename, 'w') as fh:
        fh.write(doc.toprettyxml(indent='  '))


def get_data(input_gpx):
    # TODO: you may use your code from exercise 3 here.
    points = pd.DataFrame(columns=['datetime', 'lat', 'lon', 'ele'])
    tree = ET.parse(input_gpx)  # parse XML into an element tree
    root = tree.getroot()  # <gpx xmlns="http://www.topografix.com/GPX/1/0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.topografix.com/GPX/1/0 http://www.topografix.com/GPX/1/0/gpx.xsd" version="1.0" creator="gpx.py -- https://github.com/tkrajina/gpxpy">
    data = root[0][0]  # <trkseg"
    i = 0
    for child in data:  # <trkpt>
        lat = float(child.attrib['lat'])
        lon = float(child.attrib['lon'])
        ele = float(child[0].text)
        datetime = pd.to_datetime(child[1].text, utc=True)  #, format="%Y-%m-%dT%H:%M:%S.%fZ")
        points.loc[i] = [datetime, lat, lon, ele]
        i += 1
    return points


def main():
    input_directory = pathlib.Path(sys.argv[1])
    output_directory = pathlib.Path(sys.argv[2])
    
    # read gopro and phone data
    accl = pd.read_json(input_directory / 'accl.ndjson.gz', lines=True, convert_dates=['timestamp'])[['timestamp', 'x']]
    gps = get_data(input_directory / 'gopro.gpx')
    phone = pd.read_csv(input_directory / 'phone.csv.gz')[['time', 'gFx', 'Bx', 'By']]

    # phone data only has number of seconds from the start time
    # assume that the phone data starts at exactly the same time as the accelerometer data to create timestamp from phone data
    # offset_0 = 0
    # first_time = accl['timestamp'].min()
    # phone['timestamp'] = first_time + pd.to_timedelta(phone['time'] + offset_0, unit='sec')

    # unify the times for each dataset
    # the 3 dataframes will have have the same keys on the rows
    # phone['timestamp'] = phone['timestamp'].dt.round('4s')
    # phone = phone.groupby('timestamp').mean().reset_index()
    
    gps['datetime'] = gps['datetime'].dt.round('4s')
    gps = gps.groupby('datetime').mean().reset_index()
    
    accl['timestamp'] = accl['timestamp'].dt.round('4s')
    accl = accl.groupby('timestamp').mean().reset_index()
    
    # TODO: create "combined" as described in the exercise
    best_offset = -np.inf
    best_correlation = -np.inf
    first_time = accl['timestamp'].min()
    combined = phone
    
    for offset in np.linspace(-5.0, 5.0, 101):
        temp = phone.copy()
        temp['timestamp'] = first_time + pd.to_timedelta(temp['time'] + offset, unit='sec')
        temp['timestamp'] = temp['timestamp'].dt.round('4s')
        temp = temp.groupby('timestamp').mean().reset_index()
        
        correlation = np.sum(temp['gFx'] * accl['x'])
        
        if (correlation > best_correlation):
            best_offset = offset
            best_correlation = correlation
            combined = temp
    
    combined = combined.join(accl.set_index('timestamp'), on='timestamp')
    gps['timestamp'] = gps['datetime']
    gps = gps.reset_index()
    
    combined = combined.join(gps.set_index('timestamp'), on='timestamp')
    
    combined.dropna(inplace=True)
    
    print(f'Best time offset: {best_offset:.1f}')
    # print(f'Best correlation: {best_correlation:.1f}')
    os.makedirs(output_directory, exist_ok=True)
    output_gpx(combined[['datetime', 'lat', 'lon']], output_directory / 'walk.gpx')
    combined[['datetime', 'Bx', 'By']].to_csv(output_directory / 'walk.csv', index=False)


main()
