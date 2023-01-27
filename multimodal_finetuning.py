from dataset.MultimodalDataset import MultimodalDataset
from preprocessing import img_preprocessing, ehr_preprocessing, multimodal_preprocessing
import torch
from models.CLIP import CLIP
from models.img_MLP import IMG_MLP
from models.ehr_MLP import EHR_MLP
from models.img_classifier import IMG_classifier
from models.ehr_classifier import EHR_classifier
from loss.CLIPloss import ClipLoss
from tqdm import tqdm
from visualization import plot_metrics
from models import model_checkpoints
import torch.nn as nn
torch.manual_seed(4)


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



#datasets
train_set = MultimodalDataset(X_train_img, X_train_ehr, y_train)
val_set = MultimodalDataset(X_val_img, X_val_ehr, y_val)
test_set = MultimodalDataset(X_test_img, X_test_ehr, y_test)

#dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

#comment - load pretrained
#intialising models
img_pretrained_model = IMG_MLP()
img_pre_model = model_checkpoints.load_ckp('./checkpoints/pretrained_img_model.pt', img_pretrained_model)
img_model = IMG_classifier(img_pre_model)
#img_model = model_checkpoints.load_ckp('./checkpoints/img_classifier.pt', img_classifier_model)

ehr_pretrained_model = EHR_MLP()
ehr_pre_model = model_checkpoints.load_ckp('./checkpoints/pretrained_ehr_model.pt', ehr_pretrained_model)
ehr_model = EHR_classifier(ehr_pre_model)
#ehr_model = model_checkpoints.load_ckp('./checkpoints/ehr_classifier.pt', ehr_classifier_model)


img_model = img_model.to(device)
ehr_model = ehr_model.to(device)


#specifying loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam([{"params": img_model.parameters()}, {"params": ehr_model.parameters()}], lr=0.01)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.1)

#train
epochs= 100
valid_acc_max = 0
epochs_list=[]
training_loss=[]
validation_loss=[]
training_acc = []
validation_acc = []

for epoch in range(0, epochs):
    train_loss=0
    test_loss=0
    train_total=0
    train_correct=0
    
    epochs_list.append(epoch)
    print("epoch number: {0}".format(epoch))
    img_model.train()
    ehr_model.train()
    with tqdm(train_loader, unit = 'batch') as tepoch:
        for batch_idx, (img_train_data, ehr_train_data, train_labels) in enumerate(tepoch):
            img_train_data = img_train_data.to(device)
            ehr_train_data = ehr_train_data.to(device)
            train_labels = train_labels.to(device)

            img_logits = img_model(img_train_data.float())
            ehr_logits = ehr_model(ehr_train_data.float())

            output = 0.5 * img_logits + 0.5 * ehr_logits

            optimizer1.zero_grad()

            loss = criterion(output,train_labels)

            loss.backward()
            optimizer1.step()

            train_loss += loss.item()

            _, predicted = output.max(1)
            train_total += train_labels.size(0)
            train_correct += predicted.eq(train_labels).sum().item()
            
    with torch.no_grad():
        test_total=0
        test_correct= 0
        
        img_model.eval()
        ehr_model.eval()
        with tqdm(val_loader, unit ="batch") as tepoch:
            for batch_idx ,(img_val_data, ehr_val_data, val_labels) in enumerate(tepoch):
                img_val_data = img_val_data.to(device)
                ehr_val_data = ehr_val_data.to(device)
                val_labels = val_labels.to(device)

                img_pred_logits = img_model(img_val_data.float())
                ehr_pred_logits = ehr_model(ehr_val_data.float())

                #print('img_logits', img_pred_logits)
                #print('ehr_logits', ehr_pred_logits)
                y_pred = 0.5 * img_pred_logits + 0.5 * ehr_pred_logits
                #print('pred', y_pred)

                loss = criterion(y_pred, val_labels)
                
                test_loss+=loss.item()
                _, predicted = y_pred.max(1)

                #print('predicted', predicted)

                test_total += val_labels.size(0)
                #print('val_labels', val_labels)
                test_correct += predicted.eq(val_labels).sum().item()
                #print('test_crct', test_correct)

    img_checkpoint = {
        'epoch': epoch + 1,
        'valid_acc_max': 100.*test_correct/test_total,
        'state_dict': img_model.state_dict(),
        'optimizer': optimizer1.state_dict(),
    }

    ehr_checkpoint = {
        'epoch': epoch + 1,
        'valid_loss_min': 100.*test_correct/test_total,
        'state_dict': ehr_model.state_dict(),
    }


    if  100.*test_correct/test_total > valid_acc_max:
        # save checkpoint as best model
        model_checkpoints.save_ckp(img_checkpoint, True, './checkpoints/multimodal_img_classifier.pt')
        model_checkpoints.save_ckp(ehr_checkpoint, True, './checkpoints/multimodal_ehr_classifier.pt')
        
        valid_acc_max = 100.*test_correct/test_total

    training_loss.append(train_loss/(batch_idx+1))
    validation_loss.append(test_loss/(batch_idx+1))
    training_acc.append(100.*train_correct/train_total)
    validation_acc.append(100.*test_correct/test_total)


    print('test loss: {:.4f} train_accuracy: {:.4f} test_accuracy: {:.4f} valid_acc{:.4f}'.format(test_loss/(batch_idx+1), 100.*train_correct/train_total, 100.*test_correct/test_total, valid_acc_max))
    scheduler1.step()
    

# plot_metrics.plot_loss(epochs_list, training_loss,'Training Loss','./visualization/classifier/ehr_classifier/loss','ehr_classifier_training_loss.png')
# plot_metrics.plot_loss(epochs_list, validation_loss,'Validation Loss' ,'./visualization/classifier/ehr_classifier/loss','ehr_classifier_validation_loss.png')

# plot_metrics.plot_acc(epochs_list, training_acc,'Training Loss','./visualization/classifier/ehr_classifier/acc','ehr_classifier_training_acc.png')
# plot_metrics.plot_acc(epochs_list, validation_acc,'Validation Loss' ,'./visualization/classifier/ehr_classifier/acc','ehr_classifier_validation_acc.png')



