import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pylab as plt
from sklearn import metrics


def mk_dataset():
    data = pd.read_csv('xpStatsTrainData.csv')
    data = data[['events', 'launch_speed', 'launch_angle']].dropna()
    # print(data.describe())
    outsList = ['field_out']
    hitsList = ['single', 'double', 'triple', 'home_run']
    desiredOutcome = ['out', 'single', 'double', 'triple', 'home_run']
    #outcomebases = [0, 1, 2, 3, 4]
    data['events'] = data['events'].replace(outsList, 'out')
    data = data.loc[data['events'].isin(desiredOutcome)]
    #data['events'] = data['events'].replace(desiredOutcome, outcomebases)
    data = data.rename(columns={'events': 'bases'})
    # print(data.head())
    X_train, X_test, y_train, y_test = train_test_split(
        data[['launch_speed', 'launch_angle']], data['bases'], test_size=0.2)
    return X_train, X_test, y_train, y_test


def skl_knn(k, X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))


X_train, X_test, y_train, y_test = mk_dataset()

test_model = skl_knn(14, X_train, X_test, y_train, y_test)
print(test_model)
