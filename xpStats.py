from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import math as m
import matplotlib.pylab as plt
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
plt.style.use('bmh')

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


def getTrainingData():
    # adjust as range needed, larger ranges result in longer
    # query times and model fit times
    train = pd.read_csv('xpStatsTrainData.csv')
    train = train[['events', 'launch_speed', 'launch_angle']]
    train = train.dropna()
    train = pd.DataFrame(train, columns=['events', 'launch_speed', 'launch_angle'])
    outsList = ['field_out']
    hitsList = ['single', 'double', 'triple', 'home_run']
    desiredOutcome = ['out', 'single', 'double', 'triple', 'home_run']
    train['events'] = train['events'].replace(outsList, 'out')
    train = train.loc[train['events'].isin(desiredOutcome)]
    y_train = train['events']
    X_train = train.drop('events', axis=1)
    unused_variable = os.system("cls")
    print('Beginning to train model, this may take a while.')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40))
    mlp.fit(X_train, y_train)
    unused_variable = os.system("cls")
    return mlp, scaler


def xpredictStats(model, launch_speed, launch_angle, model_scaler):
    test = [[launch_speed, launch_angle]]
    test = pd.DataFrame(test, columns=['launch_speed', 'launch_angle'])
    test = model_scaler.transform(test)
    #prediction = model.predict(test)
    probabilities = model.predict_proba(test)
    # print(probabilities)
    # print(probabilities[0][0])  # probabilty of a double
    # print(probabilities[0][1])  # probabilty of a home run
    # print(probabilities[0][2])  # probabilty of out
    # print(probabilities[0][3])  # probabilty of a single
    # print(probabilities[0][4])  # probabilty of a triple
    xpBA = (probabilities[0][0] + probabilities[0][1] +
            probabilities[0][3] + probabilities[0][4])
    xpSLG = (probabilities[0][3] + 2 * probabilities[0][0] +
             3 * probabilities[0][4] + 4 * probabilities[0][1])
    xpResult = ''
    if(xpSLG > 3.5):
        xpResult == 'Home Run'
    else:
        if(xpSLG >= 1.5 and xpSLG < 3.5):
            xpResult = 'XBH'
        else:
            if(xpSLG >= 0.5 and xpSLG < 1.5):
                xpResult = 'Single'
            else:
                if(xpSLG < 0.5):
                    xpResult = 'Out'
    xpBA = round(xpBA, 3)
    xpSLG = round(xpSLG, 3)
    xpredictedStats = [[launch_speed, launch_angle, xpBA, xpSLG]]
    xpredictedStats = pd.DataFrame(xpredictedStats, columns=['exit_velocity',
                                                             'launch_angle',
                                                             'xpBA', 'xpSLG'])
    return xpredictedStats


def visualize(model, model_scaler, xData):
    print('Confguring Visualization, Please Wait.')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax = plt.subplot(111, projection='polar')
    ax1 = fig.add_axes([.75, 0.1, 0.03, 0.8])
    T_dat = []
    R_dat = []
    Z_dat = []
    for i in range(0, 120):
        for j in range(-180, 180):
            xpStats = xpredictStats(model, i, j, model_scaler)
            xpBA = xpStats.iloc[0]['xpBA']
            T_dat.append(m.radians(j))
            R_dat.append(i)
            Z_dat.append(xpBA)
    T, R, Z, = np.array([]), np.array([]), np.array([])
    for i in range(len(T_dat)):
        T = np.append(T, T_dat[i])
        R = np.append(R, R_dat[i])
        Z = np.append(Z, Z_dat[i])
    ti = np.linspace(T.min(), T.max(), 1000)
    ri = np.linspace(R.min(), R.max(), 1000)
    zi = griddata((T, R), Z, (ti[None, :], ri[:, None]), method='cubic')
    zmin = 0
    zmax = 1
    zi[(zi < zmin) | (zi > zmax)] = None
    ax.contourf(ti, ri, zi, vmax=zmax, vmin=zmin, cmap=cm.viridis,
                extend='both')
    cNorm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)
    cb1 = mpl.colorbar.ColorbarBase(ax1, norm=cNorm, cmap=cm.viridis)
    #ax.set_ylabel('Launch Angle (Deg.)', labelpad=-70)
    ax.set_ylim(0, 120)
    ax.set_xlim(m.pi/2, -m.pi/2)
    ax.set_xticks([0, m.pi/4, m.pi/2])
    ax.set_yticks([0, 60, 120])
    ax1.set_ylabel('xpBA')
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(top=0.9, bottom=0.1, right=.8, left=.0)
    xlab = plot_data(ax, xData)
    ax.set_xlabel(xlab, labelpad=10)
    plt.show()


def plot_data(subplot, data):
    t = m.radians(data.iloc[0]['launch_angle'])
    r = data.iloc[0]['exit_velocity']
    label = 'Batted Ball'
    subplot.scatter(t, r, c='#B22222', label=label)
    ylabel = 'xpResult: ' + ' Exit Velocity: ' + \
        str(r) + 'mph, Launch Angle: ' + str(data.iloc[0]['launch_angle']) + ' deg.'
    return ylabel


def main():
    nnModel, scale = getTrainingData()
    while(1 == 1):
        launchspeed = float(input('Enter Launch Speed (mph): '))
        launchangle = float(input('Enter Launch Angle (deg.): '))
        xpStats = xpredictStats(nnModel, launchspeed, launchangle, scale)
        print(xpStats)
        dec = input('Would you like to Plot?(Y/N): ').upper()
        if(dec == 'Y'):
            # configures visualization and colorbar
            visualize(nnModel, scale, xpStats)
        input('Press Enter to Continue ')
        unused_variable = os.system("cls")


if __name__ == "__main__":
    main()
