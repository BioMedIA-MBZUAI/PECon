from models.img_classifier import IMG_classifier
from preprocessing import img_preprocessing
from dataset.ImgDataset import ImgDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from models import model_checkpoints
from models.img_MLP import IMG_MLP
from visualization import plot_metrics
torch.manual_seed(4)


device = "cpu" if torch.cuda.is_available() else "cuda"

#loading data
X_img_train_processed, y_train = img_preprocessing.load_data('/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/CT_v2/features_train.csv')
X_img_val_processed, y_val = img_preprocessing.load_data('/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/CT_v2/features_val.csv')
X_img_test_processed, y_test = img_preprocessing.load_data('/home/santosh.sanjeev/PE-Research/RadFusion-Dataset/dataset/multimodalpulmonaryembolismdataset/CT_v2/features_test.csv')

X_train_img, X_val_img, X_test_img = X_img_train_processed, X_img_val_processed, X_img_test_processed
# print('hiii', X_train_ehr.shape, type(X_train_ehr), X_train_ehr.dtype)
# print('hellloo', X_train_img.dtype, X_train_img.shape)

print(X_train_img.dtype, X_train_img.shape, X_val_img.dtype, X_val_img.shape, X_test_img.dtype, X_test_img.shape)

# plot_metrics.tsne_plot(y_train, path = './visualization/tsne_plots/imaging', name = 'image_train_data.png', img_data = X_img_train_processed)
# plot_metrics.tsne_plot(y_val, path = './visualization/tsne_plots/imaging', name = 'image_val_data.png', img_data = X_img_val_processed)
# plot_metrics.tsne_plot(y_test, path = './visualization/tsne_plots/imaging', name = 'image_test_data.png', img_data = X_img_test_processed)


train_set = ImgDataset(X_train_img, y_train)
val_set = ImgDataset(X_val_img, y_val)
test_set = ImgDataset(X_test_img, y_test)

#dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

pretrained_model = IMG_MLP()
img_model = model_checkpoints.load_ckp('./checkpoints/pretrained_img_model.pt', pretrained_model)
img_model.eval()

#intialising models
img_model = img_model.to(device)
img_classifier = IMG_classifier(img_model)
img_classifier = img_classifier.to(device)

# specify loss function
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = torch.optim.Adam(img_classifier.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)



epochs= 60

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
    img_classifier.train()
    with tqdm(train_loader, unit = 'batch') as tepoch:
        for batch_idx, (img_train_data, train_labels) in enumerate(tepoch):
            img_train_data = img_train_data.to(device)
            train_labels = train_labels.to(device)


            output = img_classifier(img_train_data.float())

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
        
        img_classifier.eval()
        with tqdm(val_loader, unit ="batch") as tepoch:
            for batch_idx ,(img_val_data, val_labels) in enumerate(tepoch):
                img_val_data = img_val_data.to(device)
                val_labels = val_labels.to(device)
            
                y_pred_test = img_classifier(img_val_data.float())

                loss = criterion(y_pred_test, val_labels)
                
                test_loss+=loss.item()

                _, predicted = y_pred_test.max(1)
                test_total += val_labels.size(0)
                test_correct += predicted.eq(val_labels).sum().item()

    img_checkpoint = {
        'epoch': epoch + 1,
        'valid_acc_max': 100.*test_correct/test_total,
        'state_dict': img_classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    if  100.*test_correct/test_total > valid_acc_max:
        # save checkpoint as best model
        model_checkpoints.save_ckp(img_checkpoint, True, './checkpoints/img_classifier.pt')
        valid_acc_max = 100.*test_correct/test_total

    training_loss.append(train_loss/(batch_idx+1))
    validation_loss.append(test_loss/(batch_idx+1))
    training_acc.append(100.*train_correct/train_total)
    validation_acc.append(100.*test_correct/test_total)


    print('test loss: {:.4f} train_accuracy: {:.4f} test_accuracy: {:.4f} valid_acc{:.4f}'.format(test_loss/(batch_idx+1), 100.*train_correct/train_total, 100.*test_correct/test_total, valid_acc_max))
    scheduler.step()


plot_metrics.plot_loss(epochs_list, training_loss,'Training Loss','./visualization/classifier/img_classifier/loss','img_classifier_training_loss.png')
plot_metrics.plot_loss(epochs_list, validation_loss,'Validation Loss' ,'./visualization/classifier/img_classifier/loss','img_classifier_validation_loss.png')
plot_metrics.plot_acc(epochs_list, training_loss,'Training Loss','./visualization/classifier/img_classifier/acc','img_classifier_training_acc.png')
plot_metrics.plot_acc(epochs_list, validation_loss,'Validation Loss' ,'./visualization/classifier/img_classifier/acc','img_classifier_validation_acc.png')