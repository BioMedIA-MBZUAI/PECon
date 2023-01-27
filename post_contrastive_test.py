from models.ehr_classifier import EHR_classifier
from models.img_classifier import IMG_classifier
from models.img_MLP import IMG_MLP
from preprocessing import multimodal_preprocessing, ehr_preprocessing
from dataset.EhrDataset import EhrDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from models import model_checkpoints
from models.ehr_MLP import EHR_MLP
from visualization import plot_metrics
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score



torch.manual_seed(4)

#set data to image or ehr or multimodal
data = 'ehr'

device = "cpu" if torch.cuda.is_available() else "cuda"

#loading data
X_img_train_processed, X_ehr_train_processed, y_train = multimodal_preprocessing.load_data('/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/CT_v2/features_train.csv', '/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/new_data_csv/vision&emr_train_ed.csv')
X_img_val_processed, X_ehr_val_processed, y_val = multimodal_preprocessing.load_data('/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/CT_v2/features_val.csv','/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/new_data_csv/vision&emr_val_ed.csv')
X_img_test_processed, X_ehr_test_processed, y_test = multimodal_preprocessing.load_data('/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/CT_v2/features_test.csv', '/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/new_data_csv/vision&emr_test_ed.csv')

X_train_img, X_val_img, X_test_img = X_img_train_processed, X_img_val_processed, X_img_test_processed
X_train_ehr, X_val_ehr, X_test_ehr = ehr_preprocessing.feature_selection(X_ehr_train_processed, y_train, X_ehr_val_processed, X_ehr_test_processed)

# print('hiii', X_train_ehr.shape, type(X_train_ehr), X_train_ehr.dtype)
# print('hellloo', X_train_img.dtype, X_train_img.shape)
X_train_ehr, X_val_ehr, X_test_ehr  = ehr_preprocessing.normalize(X_train_ehr, X_val_ehr, X_test_ehr)

print(X_train_img.dtype, X_train_img.shape, X_val_img.dtype, X_val_img.shape, X_test_img.dtype, X_test_img.shape, X_train_ehr.dtype, X_train_ehr.shape, X_val_ehr.dtype, X_val_ehr.shape, X_test_ehr.dtype, X_test_ehr.shape)

plot_metrics.tsne_plot(y_train, path = './visualization/tsne_plots/multimodal', name = 'pre_train.png', img_data = X_train_img, ehr_data= X_train_ehr)
plot_metrics.tsne_plot(y_val, path = './visualization/tsne_plots/multimodal', name = 'pre_val_.png', img_data = X_val_img, ehr_data= X_val_ehr)
plot_metrics.tsne_plot(y_test, path = './visualization/tsne_plots/multimodal', name = 'pre_test_.png', img_data = X_test_img, ehr_data= X_test_ehr)


#loading model checkpoints
img_prev_model = IMG_MLP()
img_model = model_checkpoints.load_ckp('./checkpoints/pretrained_img_model.pt', img_prev_model)

ehr_prev_model = EHR_MLP()
ehr_model = model_checkpoints.load_ckp('./checkpoints/pretrained_ehr_model.pt', ehr_prev_model)
#print('ehr_model', ehr_model.ehr_model.fc1.state_dict())
img_model.eval()
ehr_model.eval()

img_train_embeddings = img_model(torch.from_numpy(X_train_img).to(device).float())
img_val_embeddings = img_model(torch.from_numpy(X_val_img).to(device).float())
img_test_embeddings = img_model(torch.from_numpy(X_test_img).to(device).float())


ehr_train_embeddings = ehr_model(torch.from_numpy(X_train_ehr).to(device).float())
ehr_val_embeddings = ehr_model(torch.from_numpy(X_val_ehr).to(device).float())
ehr_test_embeddings = ehr_model(torch.from_numpy(X_test_ehr).to(device).float())



img_train_embeddings = img_train_embeddings.cpu().data.numpy()
img_test_embeddings = img_test_embeddings.cpu().data.numpy()
img_val_embeddings = img_val_embeddings.cpu().data.numpy()


ehr_train_embeddings = ehr_train_embeddings.cpu().data.numpy()
ehr_val_embeddings = ehr_val_embeddings.cpu().data.numpy()
ehr_test_embeddings = ehr_test_embeddings.cpu().data.numpy()


plot_metrics.tsne_plot(y_train, path = './visualization/tsne_plots/multimodal', name = 'contrastive_train_embeddings.png', img_data = img_train_embeddings, ehr_data= ehr_train_embeddings)
plot_metrics.tsne_plot(y_val, path = './visualization/tsne_plots/multimodal', name = 'contrastive_val_embeddings.png', img_data = img_val_embeddings, ehr_data= ehr_val_embeddings)
plot_metrics.tsne_plot(y_test, path = './visualization/tsne_plots/multimodal', name = 'contrastive_test_embeddings.png', img_data = img_test_embeddings, ehr_data= ehr_test_embeddings)
