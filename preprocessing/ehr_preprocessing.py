import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC


def load_data(path):
    data = pd.read_csv(path)
    #print(data.shape)
    X = data.iloc[:,2:].values
    y = data.iloc[:,0].values
    
    #print(X,y)
    
    return X, y

def feature_selection(Xtrain, ytrain, Xval, Xtest):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, random_state=0).fit(Xtrain, ytrain)
    model = SelectFromModel(lsvc, prefit=True)

    X_train_ehr = model.transform(Xtrain)
    X_val_ehr = model.transform(Xval)
    X_test_ehr = model.transform(Xtest)

    #print(X_train_ehr.shape, X_test_ehr.dtype)

    return X_train_ehr, X_val_ehr, X_test_ehr


def standardize(train_data, val_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    X_train = scaler.transform(train_data)
    X_val = scaler.transform(val_data)
    X_test = scaler.transform(test_data)

    return X_train, X_val, X_test


def normalize(train_data, val_data, test_data):
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    X_train = scaler.transform(train_data)
    X_val = scaler.transform(val_data)
    X_test = scaler.transform(test_data)

    return X_train, X_val, X_test


