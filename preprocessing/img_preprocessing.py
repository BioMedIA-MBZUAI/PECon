import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(path):
    data = pd.read_csv(path)
    data = data.drop(['Unnamed: 0','pred'],axis=1)
    data = data.sort_values(by=['study_nums'], ascending=True)
    #print(data.shape)

    X = data.iloc[:,2:].values
    y = data.iloc[:,1].values
    #print(X,y)
    return X, y

def standardize(train_data, val_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    X_train = scaler.transform(train_data)
    X_val = scaler.transform(val_data)
    X_test = scaler.transform(test_data)
    #print(X_train.shape, X_val.shape, X_test.shape)

    return X_train, X_val, X_test


def normalize(train_data, val_data, test_data):
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    X_train = scaler.transform(train_data)
    X_val = scaler.transform(val_data)
    X_test = scaler.transform(test_data)

    return X_train, X_val, X_test

