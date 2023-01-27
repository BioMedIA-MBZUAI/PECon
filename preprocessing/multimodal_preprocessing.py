import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(img_features_path, ehr_features_path):
    img_data = pd.read_csv(img_features_path)
    img_data = img_data.drop(['Unnamed: 0','pred'],axis=1)
    img_data = img_data.sort_values(by=['study_nums'], ascending=True)
    #print(img_data.shape)

    ehr_data = pd.read_csv(ehr_features_path)
    #print(ehr_data.shape)



    img_X = img_data.iloc[:,2:].values
    ehr_X = ehr_data.iloc[:,2:].values
    y = ehr_data.iloc[:,0].values

    #print(img_X, ehr_X, y)
    return img_X, ehr_X, y

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

