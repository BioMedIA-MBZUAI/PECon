from models.ehr_classifier import EHR_classifier
from preprocessing import ehr_preprocessing
from dataset.EhrDataset import EhrDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from models import model_checkpoints
from models.ehr_MLP import EHR_MLP
from visualization import plot_metrics
torch.manual_seed(0)


device = "cpu" if torch.cuda.is_available() else "cuda"

#loading data
X_ehr_train_processed, y_train = ehr_preprocessing.load_data('/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/new_data_csv/vision&emr_train_ed.csv')
X_ehr_val_processed, y_val = ehr_preprocessing.load_data('/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/new_data_csv/vision&emr_val_ed.csv')
X_ehr_test_processed, y_test = ehr_preprocessing.load_data('/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/new_data_csv/vision&emr_test_ed.csv')

X_train_ehr, X_val_ehr, X_test_ehr = ehr_preprocessing.feature_selection(X_ehr_train_processed, y_train, X_ehr_val_processed, X_ehr_test_processed)

# print('hiii', X_train_ehr.shape, type(X_train_ehr), X_train_ehr.dtype)
# print('hellloo', X_train_img.dtype, X_train_img.shape)
X_train_ehr, X_val_ehr, X_test_ehr  = ehr_preprocessing.normalize(X_train_ehr, X_val_ehr, X_test_ehr)

print(X_train_ehr.dtype, X_train_ehr.shape, X_val_ehr.dtype, X_val_ehr.shape, X_test_ehr.dtype, X_test_ehr.shape)

#tsne-plots
plot_metrics.tsne_plot(y_train, path = './visualization/tsne_plots/ehr', name = 'ehr_train_data.png', img_data = X_train_ehr)
plot_metrics.tsne_plot(y_val, path = './visualization/tsne_plots/ehr', name = 'ehr_val_data.png', img_data = X_val_ehr)
plot_metrics.tsne_plot(y_test, path = './visualization/tsne_plots/ehr', name = 'ehr_test_data.png', img_data = X_test_ehr)


train_set = EhrDataset(X_train_ehr, y_train)
val_set = EhrDataset(X_val_ehr, y_val)
test_set = EhrDataset(X_test_ehr, y_test)

#dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

pretrained_model = EHR_MLP()
ehr_model = model_checkpoints.load_ckp('./checkpoints/pretrained_ehr_model.pt', pretrained_model)


#intialising models
ehr_model = ehr_model.to(device)
ehr_classifier = EHR_classifier(ehr_model)
ehr_classifier = ehr_classifier.to(device)

# specify loss function
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = torch.optim.Adam(ehr_classifier.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)



epochs= 80

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
    ehr_classifier.train()
    with tqdm(train_loader, unit = 'batch') as tepoch:
        for batch_idx, (ehr_train_data, train_labels) in enumerate(tepoch):
            ehr_train_data = ehr_train_data.to(device)
            train_labels = train_labels.to(device)


            output = ehr_classifier(ehr_train_data.float())

            optimizer.zero_grad()
            loss = criterion(output,train_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            _, predicted = output.max(1)
            train_total += train_labels.size(0)
            train_correct += predicted.eq(train_labels).sum().item()
            
    with torch.no_grad():
        test_total=0
        test_correct= 0
        
        ehr_classifier.eval()
        with tqdm(val_loader, unit ="batch") as tepoch:
            for batch_idx ,(ehr_val_data, val_labels) in enumerate(tepoch):
                ehr_val_data = ehr_val_data.to(device)
                val_labels = val_labels.to(device)
            
                y_pred_test = ehr_classifier(ehr_val_data.float())

                loss = criterion(y_pred_test, val_labels)
                
                test_loss+=loss.item()

                _, predicted = y_pred_test.max(1)
                test_total += val_labels.size(0)
                test_correct += predicted.eq(val_labels).sum().item()

    ehr_checkpoint = {
        'epoch': epoch + 1,
        'valid_acc_max': 100.*test_correct/test_total,
        'state_dict': ehr_classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    if  100.*test_correct/test_total > valid_acc_max:
        # save checkpoint as best model
        model_checkpoints.save_ckp(ehr_checkpoint, True, './checkpoints/ehr_classifier.pt')
        valid_acc_max = 100.*test_correct/test_total

    training_loss.append(train_loss/(batch_idx+1))
    validation_loss.append(test_loss/(batch_idx+1))
    training_acc.append(100.*train_correct/train_total)
    validation_acc.append(100.*test_correct/test_total)


    print('test loss: {:.4f} train_accuracy: {:.4f} test_accuracy: {:.4f} valid_acc{:.4f}'.format(test_loss/(batch_idx+1), 100.*train_correct/train_total, 100.*test_correct/test_total, valid_acc_max))
    scheduler.step()


plot_metrics.plot_loss(epochs_list, training_loss,'Training Loss','./visualization/classifier/ehr_classifier/loss','ehr_classifier_training_loss.png')
plot_metrics.plot_loss(epochs_list, validation_loss,'Validation Loss' ,'./visualization/classifier/ehr_classifier/loss','ehr_classifier_validation_loss.png')

plot_metrics.plot_acc(epochs_list, training_acc,'Training Loss','./visualization/classifier/ehr_classifier/acc','ehr_classifier_training_acc.png')
plot_metrics.plot_acc(epochs_list, validation_acc,'Validation Loss' ,'./visualization/classifier/ehr_classifier/acc','ehr_classifier_validation_acc.png')



