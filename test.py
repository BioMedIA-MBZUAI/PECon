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
import pandas as pd


torch.manual_seed(0)

#set data to image or ehr or multimodal
data = 'multimodal'

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

#loading model checkpoints
img_prev_model = IMG_MLP()
img_classifier = IMG_classifier(img_prev_model)
img_model = model_checkpoints.load_ckp('./checkpoints/img_classifier.pt', img_classifier)
img_model.eval()

ehr_prev_model = EHR_MLP()
#print('pretrained_model', ehr_prev_model.fc1.state_dict())
ehr_classifier = EHR_classifier(ehr_prev_model)
#print('ehr_classifier', ehr_classifier.ehr_model.fc1.state_dict())
ehr_model = model_checkpoints.load_ckp('./checkpoints/ehr_classifier.pt', ehr_classifier)
#print('ehr_model', ehr_model.ehr_model.fc1.state_dict())
ehr_model.eval()

#inference on validation ehr data

if data == 'ehr':
    y_pred_val = F.softmax(ehr_model(torch.from_numpy(X_val_ehr).to(device).float()))
    y_pred_val = y_pred_val.tolist()
    y_pred_val = [p[1] for p in y_pred_val]
    fpr_ehr1, tpr_ehr1, _ = metrics.roc_curve(y_val, y_pred_val)
    roc_auc_ehr = metrics.auc(fpr_ehr1, tpr_ehr1)
    optimal_proba_cutoff = sorted(list(zip(np.abs(tpr_ehr1 - fpr_ehr1), _)), key=lambda i: i[0], reverse=True)[0][1]

    print(optimal_proba_cutoff)
    ehr_list=[]
    for i in y_pred_val:
        if i>=round(optimal_proba_cutoff,2):
            ehr_list.append(1)
        else:
            ehr_list.append(0)

    print('validation accuracy', accuracy_score(y_val,ehr_list))



    y_pred_test = F.softmax(ehr_model(torch.from_numpy(X_test_ehr).to(device).float()))
    y_pred_test = y_pred_test.tolist()
    y_pred_test = [p[1] for p in y_pred_test]
    
    ehr_test_list=[]
    for i in y_pred_test:
        if i>=round(optimal_proba_cutoff,2):
            ehr_test_list.append(1)
        else:
            ehr_test_list.append(0)

    plot_metrics.confusion_matrix_plot(y_test, ehr_test_list, './visualization/confusion_matrices','ehr_finetuned.png')
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, ehr_test_list).ravel()


    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print('accuracy',accuracy)
    fpr_ehr1, tpr_ehr1, _ = metrics.roc_curve(y_test, y_pred_test)
    roc_auc_ehr = metrics.auc(fpr_ehr1, tpr_ehr1)
    print('aucroc', roc_auc_ehr)
    specificity = tn / (tn+fp)
    print('specificity',specificity)
    sensitivity = tp/(tp+fn)
    print('sensitivity',sensitivity)
    ppv = tp/(tp+fp)
    print('ppv',ppv)
    npv = tn/(tn+fn)
    print('npv',npv)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*(precision*recall)/(precision+recall)
    print('f1_score', f1_score)
    print('y_test',y_test)
    print('y_pred', y_pred_test)


    df = pd.read_csv('./results/original_labels.csv')

    df['ehr_predict_proba'] = y_pred_test
    df['pred'] = ehr_test_list

    df.to_csv('./results/finetuned_ehr_pred.csv')
    
    lines =['optimal_threshold' + '= ' + str(optimal_proba_cutoff),'accuracy' + '= ' + str(accuracy), 'aucroc' + '= ' + str(roc_auc_ehr) , 'specificity' + '= ' + str(specificity), 'sensitivity' + '= '   + str(sensitivity), 'ppv' + '= ' + str(ppv), 'npv' + '= '  + str(npv), 'f1_score' + '= ' + str(f1_score) ]
    with open('./results/finetuned_ehr.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')




elif data == 'image':
    y_pred_val = F.softmax(img_model(torch.from_numpy(X_val_img).to(device).float()))
    y_pred_val = y_pred_val.tolist()
    y_pred_val = [p[1] for p in y_pred_val]
    fpr_img1, tpr_img1, _ = metrics.roc_curve(y_val, y_pred_val)
    roc_auc_img = metrics.auc(fpr_img1, tpr_img1)
    optimal_proba_cutoff = sorted(list(zip(np.abs(tpr_img1 - fpr_img1), _)), key=lambda i: i[0], reverse=True)[0][1]

    print(optimal_proba_cutoff)
    img_list=[]
    for i in y_pred_val:
        if i>=round(optimal_proba_cutoff,2):
            img_list.append(1)
        else:
            img_list.append(0)

    print('validation accuracy', accuracy_score(y_val,img_list))



    y_pred_test = F.softmax(img_model(torch.from_numpy(X_test_img).to(device).float()))
    y_pred_test = y_pred_test.tolist()
    y_pred_test = [p[1] for p in y_pred_test]
    
    img_test_list=[]
    for i in y_pred_test:
        if i>=round(optimal_proba_cutoff,2):
            img_test_list.append(1)
        else:
            img_test_list.append(0)

    plot_metrics.confusion_matrix_plot(y_test, img_test_list, './visualization/confusion_matrices','img_finetuned.png')
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, img_test_list).ravel()


    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print('accuracy',accuracy)
    fpr_img1, tpr_img1, _ = metrics.roc_curve(y_test, y_pred_test)
    roc_auc_img = metrics.auc(fpr_img1, tpr_img1)
    print('aucroc', roc_auc_img)
    specificity = tn / (tn+fp)
    print('specificity',specificity)
    sensitivity = tp/(tp+fn)
    print('sensitivity',sensitivity)
    ppv = tp/(tp+fp)
    print('ppv',ppv)
    npv = tn/(tn+fn)
    print('npv',npv)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*(precision*recall)/(precision+recall)
    print('f1_score', f1_score)

    print('y_test',y_test)
    print('y_pred', y_pred_test)


    df = pd.read_csv('./results/original_labels.csv')

    df['ehr_predict_proba'] = y_pred_test
    df['pred'] = img_test_list

    df.to_csv('./results/finetuned_img_pred.csv')

    lines =['optimal_threshold' + '= ' + str(optimal_proba_cutoff),'accuracy' + '= ' + str(accuracy), 'aucroc' + '= ' + str(roc_auc_img) , 'specificity' + '= ' + str(specificity), 'sensitivity' + '= '   + str(sensitivity), 'ppv' + '= ' + str(ppv), 'npv' + '= '  + str(npv), 'f1_score' + '= ' + str(f1_score) ]
    with open('./results/finetuned_img.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


elif data == 'multimodal':
    multimodal_img_prev_model = IMG_MLP()
    multimodal_img_classifier = IMG_classifier(multimodal_img_prev_model)
    multimodal_img_model = model_checkpoints.load_ckp('./checkpoints/multimodal_img_classifier.pt', multimodal_img_classifier)
    multimodal_img_model.eval()

    multimodal_ehr_prev_model = EHR_MLP()
    multimodal_ehr_classifier = EHR_classifier(multimodal_ehr_prev_model)
    multimodal_ehr_model = model_checkpoints.load_ckp('./checkpoints/multimodal_ehr_classifier.pt', multimodal_ehr_classifier)
    multimodal_ehr_model.eval()

    y_ehr_pred_val = F.softmax(multimodal_ehr_model(torch.from_numpy(X_val_ehr).to(device).float()))
    y_ehr_pred_val = y_ehr_pred_val.tolist()
    y_ehr_pred_val = [p[1] for p in y_ehr_pred_val]
    
    y_img_pred_val = F.softmax(multimodal_img_model(torch.from_numpy(X_val_img).to(device).float()))
    y_img_pred_val = y_img_pred_val.tolist()
    y_img_pred_val = [p[1] for p in y_img_pred_val]
    
    # print('y_ehr_pred_val', y_ehr_pred_val)
    # print('y_img_pred_val', y_img_pred_val)

    y_val_pred = []
    for i in range(len(y_ehr_pred_val)):
        y_val_pred.append((y_ehr_pred_val[i]+y_img_pred_val[i])/2)

    fpr_1, tpr_1, _ = metrics.roc_curve(y_val, y_val_pred)
    roc_auc = metrics.auc(fpr_1, tpr_1)
    optimal_proba_cutoff = sorted(list(zip(np.abs(tpr_1 - fpr_1), _)), key=lambda i: i[0], reverse=True)[0][1]

    print('optimal cutoff',optimal_proba_cutoff)
    img_list=[]
    for i in y_val_pred:
        if i>=round(optimal_proba_cutoff,2):
            img_list.append(1)
        else:
            img_list.append(0)

    print('validation accuracy', accuracy_score(y_val,img_list))

    

    y_ehr_pred_test = F.softmax(multimodal_ehr_model(torch.from_numpy(X_test_ehr).to(device).float()))
    y_ehr_pred_test = y_ehr_pred_test.tolist()
    y_ehr_pred_test = [p[1] for p in y_ehr_pred_test]
    
    y_img_pred_test = F.softmax(multimodal_img_model(torch.from_numpy(X_test_img).to(device).float()))
    y_img_pred_test = y_img_pred_test.tolist()
    y_img_pred_test = [p[1] for p in y_img_pred_test]
    
    # print('y_ehr_pred_test', y_ehr_pred_test)
    # print('y_img_pred_test', y_img_pred_test)

    y_test_pred = []
    for i in range(len(y_ehr_pred_test)):
        y_test_pred.append((y_ehr_pred_test[i]+y_img_pred_test[i])/2)

    test_list=[]
    for i in y_test_pred:
        if i>=round(optimal_proba_cutoff,2):
            test_list.append(1)
        else:
            test_list.append(0)

    plot_metrics.confusion_matrix_plot(y_test, test_list, './visualization/confusion_matrices','multimodal_finetuned.png')
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, test_list).ravel()


    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print('accuracy',accuracy)
    fpr_img1, tpr_img1, _ = metrics.roc_curve(y_test, y_test_pred)
    roc_auc_multimodal = metrics.auc(fpr_img1, tpr_img1)
    print('aucroc', roc_auc_multimodal)
    specificity = tn / (tn+fp)
    print('specificity',specificity)
    sensitivity = tp/(tp+fn)
    print('sensitivity',sensitivity)
    ppv = tp/(tp+fp)
    print('ppv',ppv)
    npv = tn/(tn+fn)
    print('npv',npv)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*(precision*recall)/(precision+recall)
    print('f1_score', f1_score)

    print('y_test',y_test)
    print('y_pred', y_test_pred)

    df = pd.read_csv('./results/original_labels.csv')

    df['img_predict_proba'] = y_img_pred_test
    df['ehr_predict_proba'] = y_ehr_pred_test
    df['avg_pred'] = y_test_pred
    df['pred'] = test_list

    df.to_csv('./results/multimodal_pred.csv')

    lines =['optimal_threshold' + '= ' + str(optimal_proba_cutoff),'accuracy' + '= ' + str(accuracy), 'aucroc' + '= ' + str(roc_auc_multimodal) , 'specificity' + '= ' + str(specificity), 'sensitivity' + '= '   + str(sensitivity), 'ppv' + '= ' + str(ppv), 'npv' + '= '  + str(npv), 'f1_score' + '= ' + str(f1_score) ]
    with open('./results/multimodal.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')



