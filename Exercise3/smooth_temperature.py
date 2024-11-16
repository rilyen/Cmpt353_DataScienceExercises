import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

def to_timestamp(dt):
    return dt.timestamp()

def main():
    
    filename1 = sys.argv[1]
    cpu_data = pd.read_csv(filename1, parse_dates=['timestamp'])
    
    """ LOESS Smoothing """
    plt.figure(figsize=(12, 4))
    plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)
    x_values = cpu_data['timestamp'].apply(to_timestamp)
    y_values = cpu_data['temperature']
    loess_smoothed = lowess(y_values, x_values, frac=0.07)
    plt.plot(cpu_data['timestamp'], loess_smoothed[:,1], 'r-')
    # plt.show()
    # plt.savefig('cpu.png')
    
    """ Kalman Smoothing """
    kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([0.5,   0.5,   0.5,   0.5]) ** 2 # TODO: shouldn't be zero
    transition_covariance =  np.diag([0.05,  0.05,  0.05, 0.05]) ** 2 # TODO: shouldn't be zero
    transition = [[0.94, 0.5,  0.2,  -0.001], 
                  [0.1,  0.4,  2.1,   0], 
                  [0,      0,  0.94,  0], 
                  [0,      0,     0,  1]] # TODO: shouldn't (all) be zero
    
    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )
    kalman_smoothed, _ = kf.smooth(kalman_data)
    
    plt.plot(cpu_data['timestamp'], kalman_smoothed[:,0], 'g-')
    plt.legend(['data points', 'loess-smoothed line', 'kalman-smoothed line'])
    plt.title('CPU Temperature Noise Reduction')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature')
    # plt.show()
    plt.savefig('cpu.png')
    plt.savefig('cpu.svg')
    return 0

if __name__ == '__main__':
    main()