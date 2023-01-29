from preprocessing import ehr_preprocessing
import pandas as pd
ROOT = "/home/salwa.khatib/MultiModalFusion/data-dir/ehr.csv"

df = pd.read_csv(ROOT)
df_train = df[df['phase'] == 'train'].iloc[:,:-2]
df_val = df[df['phase'] == 'val'].iloc[:,:-2]
df_test = df[df['phase'] == 'test'].iloc[:,:-2]
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)
df_train = df_train.drop(columns=['pred'], inplace=False, axis = 1)
df_val = df_val.drop(columns=['pred'], inplace=False, axis = 1).iloc[:,1:].values
df_test = df_test.drop(columns=['pred'], inplace=False, axis = 1).iloc[:,1:].values

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)


df1 = df_train.iloc[:,1:].values
df2 = df_train.iloc[:,0].values

x, y, z = ehr_preprocessing.feature_selection(df1, df2, df_val, df_test)
#concatenate the labels with the features
train = pd.DataFrame(x)
val = pd.DataFrame(y)
test = pd.DataFrame(z)
x = pd.concat([val, train,test], axis=0)
x.to_csv('ehr.csv', index=False)
# print(ehr_preprocessing.feature_selection(df1, df2))

