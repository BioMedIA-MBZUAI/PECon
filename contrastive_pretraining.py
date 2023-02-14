from dataset.MultimodalDataset import MultimodalDataset
from preprocessing import img_preprocessing, ehr_preprocessing, multimodal_preprocessing
import torch
import os
import models
from datasets import CTPEDataset3d
from models.CLIP import CLIP
from models.img_MLP import IMG_MLP
from models.ehr_MLP import EHR_MLP
from loss.CLIPloss import ClipLoss
from tqdm import tqdm
import utils
from visualization import plot_metrics
from models import model_checkpoints
from args import TrainArgParser
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

#intialising models
img_model = IMG_MLP()
ehr_model = EHR_MLP()

# Intializing PeNet backbone
parser = TrainArgParser()
args_ = parser.parse_args()
EXP_NAME = args_.name
start_epoch = 0
print("Experiment Name: ", EXP_NAME)


model_fn = models.__dict__[args_.model]
backbone_model = model_fn(**vars(args_))
backbone_model = backbone_model

# Combined models
combined_model = CLIP(img_model, ehr_model, backbone_model, unfreeze_penet = args_.unfreeze_penet)
# ehr_model = ehr_model.to(device)
# img_model = img_model.to(device)
# combined_model = combined_model.to(device)
print(utils.get_available_devices())
combined_model = torch.nn.DataParallel(combined_model, utils.get_available_devices()[1])
combined_model = combined_model.to(utils.get_available_devices()[1][0])

#specifying loss function and optimizer
criterion = ClipLoss()
optimizer = torch.optim.SGD(combined_model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

# Loading pretrained model
if(args_.use_pretrained):
    print("[INFO] Loading PeNet pretrained model from {}...".format(args_.ckpt_path))
    backbone_model.load_pretrained(args_.ckpt_path, args_.gpu_ids)

if(args_.resume_training):
    print("[INFO] Resuming training from checkpoint...")
    assert args_.ehr_resume_path != None and args_.img_resume_path != None and args_.penet_resume_path != None, "[ERROR] Please specify all paths for resuming training"
    print("[INFO] Loading pretrained EHR model from {}...".format(args_.ehr_resume_path))
    ehr_model = utils.resume_checkpoint(ehr_model, args_.ehr_resume_path, device=device)
    print("[INFO] Loading pretrained IMG model from {}...".format(args_.img_resume_path))
    img_model = utils.resume_checkpoint(img_model, args_.img_resume_path, device=device)
    if(args_.unfreeze_penet):
        print("[INFO] Loading pretrained PeNet model from {}...".format(args_.penet_resume_path))
        backbone_model.load_pretrained(args_.penet_resume_path, args_.gpu_ids)
    print("[INFO] Loading optimizer...")
    optimizer = utils.load_optimizer(optimizer, args_.img_resume_path, device=device)
    print("[INFO] Loading scheduler...")
    scheduler = utils.load_scheduler(scheduler, args_.img_resume_path, device=device)
    start_epoch = utils.start_epoch(args_.img_resume_path, device=device)
    try:
        EXP_NAME = utils.load_exp_name(args_.img_resume_path, device=device)
        print("[INFO] Experiment name loaded from checkpoint: {}".format(EXP_NAME))
    except:
        print("[WARNING] Could not load experiment name from checkpoint. Using default experiment name: {}".format(EXP_NAME))
    


# Data Loaders
train_set = CTPEDataset3d(args_, phase='train')
val_set = CTPEDataset3d(args_, phase='val', is_training_set=False)
# test_set = CTPEDataset3d(args_, phase='test', is_training_set=False)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args_.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args_.batch_size, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=args_.batch_size, shuffle=False)

print('train_set: ', len(train_set))
print('val_set: ', len(val_set))
# print('test_set: ', len(test_set))

#train
epochs= args_.num_epochs
print('[INFO] Training for {} epochs...'.format(epochs))
print("[INFO] Epochs per evaluation: {}".format(args_.epochs_per_eval))
print('[INFO] Clip batch size: {}'.format(args_.clip_bs))


def train():
    valid_loss_min = 20
    epochs_list=[]
    training_loss=[]
    validation_loss=[]
    loss_computations = (len(train_set) / args_.clip_bs)
    print("[INFO] Number of loss computations per epoch: {}".format(loss_computations))
    #print('epochs', epochs)
    for epoch in range(start_epoch, epochs):
        #print('[INFO] helllooo: {}'.format(args_.clip_bs))
        train_loss=0
        test_loss=0
        epochs_list.append(epoch)
        print("epoch number: {0}".format(epoch))

        combined_model.train()
        bs_counter = 0
        f1s = []
        f2s = []

        f1ten = torch.zeros((128,args_.clip_bs), dtype=torch.float64)
        f2ten = torch.zeros((128,args_.clip_bs), dtype=torch.float64)

        with tqdm(train_loader, unit = 'batch') as tepoch:
            for batch_idx, (img_train_data, ehr_train_data, train_labels) in enumerate(tepoch):

                img_train_data = img_train_data.to(utils.get_available_devices()[1][0])
                #print('hiiii',img_train_data.shape)
                ehr_train_data = ehr_train_data.to(utils.get_available_devices()[1][0])
                #print('hiiii',ehr_train_data.shape)
                # train_labels = train_labels.to(device)
                # for key, value in train_labels.items():
                #     train_labels[key] = train_labels[key].to(device)

                # print(img_train_data.device, ehr_train_data.device)
                
                f1,f2, logits_scale = combined_model.forward(img_train_data.float(), ehr_train_data.float())
                
                optimizer.zero_grad()

                # CLIP Loss
                f1ten[:,bs_counter] = f1.cpu()
                f2ten[:,bs_counter] = f2.cpu()
                logits_scale = logits_scale.cpu()

                bs_counter += 1

                if bs_counter == args_.clip_bs or batch_idx == len(train_loader) - 1:

                    print("\n[INFO] Time to compute clip loss...")
                    bs_counter = 0
                    loss = criterion(f1ten, f2ten, logits_scale)
                    f1ten = torch.zeros((128,args_.clip_bs), dtype=torch.float64)
                    f2ten = torch.zeros((128,args_.clip_bs), dtype=torch.float64)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    print("\n[INFO] train loss: {0}".format(train_loss))
        
        if(epoch % args_.epochs_per_eval==0):
            print("\n[INFO] Starting validation...")
            with torch.no_grad():
                combined_model.eval()
                f1ten = torch.zeros((128,args_.clip_bs), dtype=torch.float64)
                f2ten = torch.zeros((128,args_.clip_bs), dtype=torch.float64)

                
                with tqdm(val_loader, unit ="batch") as tepoch:
                    for val_batch_idx ,(img_val_data, ehr_val_data, val_labels) in enumerate(tepoch):
                        img_val_data = img_val_data.to(utils.get_available_devices()[1][0])
                        ehr_val_data = ehr_val_data.to(utils.get_available_devices()[1][0])
                        # val_labels = val_labels.to(device)
                        
                        f1,f2, logits_scale = combined_model.forward(img_val_data.float(), ehr_val_data.float())


                        f1ten[:,bs_counter] = f1.cpu()
                        f2ten[:,bs_counter] = f2.cpu()
                        logits_scale = logits_scale.cpu()
                        bs_counter += 1

                        if(bs_counter == args_.clip_bs or val_batch_idx == len(val_loader) - 1):
                            print("\n[INFO] Time to compute clip loss...")
                            bs_counter = 0
                            f1ten = torch.zeros((128,args_.clip_bs), dtype=torch.float64)
                            f2ten = torch.zeros((128,args_.clip_bs), dtype=torch.float64)

                            # print(f1.shape, f2.shape, logits_scale.shape)
                            loss = criterion(f1ten, f2ten, logits_scale)
                            
                            test_loss+=loss.item()
                            print("\n[INFO] val loss: {0}".format(test_loss))

  
        img_checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': test_loss/(batch_idx+1),
            'state_dict': combined_model.module.visual.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_name': EXP_NAME,
            'scheduler': scheduler.state_dict()
        }

        ehr_checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': test_loss/(batch_idx+1),
            'state_dict': combined_model.module.text.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_name': EXP_NAME,
            'scheduler': scheduler.state_dict()
        }

        penet_checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': test_loss/(batch_idx+1),
            'state_dict': combined_model.module.penetbackbone.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_name': EXP_NAME,
            'scheduler': scheduler.state_dict()
        }

        # save checkpoint
        
        MYDIR = os.path.join("./checkpoints/", EXP_NAME)
        CHECK_FOLDER = os.path.isdir(MYDIR)

        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(MYDIR)

        # # ## TODO: save the model if validation loss has decreased
        if  test_loss/(val_batch_idx+1) <= valid_loss_min:
            # save checkpoint as best model
            model_checkpoints.save_ckp(img_checkpoint, os.path.join(MYDIR,f"epoch{str(epoch)}-pretrained_img_model.pt"))
            model_checkpoints.save_ckp(ehr_checkpoint, os.path.join(MYDIR,f"epoch{str(epoch)}-pretrained_ehr_model.pt"))
            if(args_.unfreeze_penet):
                model_checkpoints.save_ckp(penet_checkpoint, os.path.join(MYDIR,f"epoch{str(epoch)}-pretrained_penet_model.pt"))
            valid_loss_min = test_loss/(val_batch_idx+1)

        training_loss.append(train_loss/(batch_idx+1))
        validation_loss.append(test_loss/(val_batch_idx+1))
        print('\naverage train loss: {:.4f} average validation loss: {:.4f}'.format(train_loss/(batch_idx+1),test_loss/(val_batch_idx+1)))
        scheduler.step()


    MYDIR = os.path.join("./visualization/pretraining/", EXP_NAME)
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)

    plot_metrics.plot_loss(epochs_list, training_loss,'Training Loss',MYDIR,'training_loss.png')
    plot_metrics.plot_loss(epochs_list, validation_loss,'Validation Loss' ,MYDIR,'validation_loss.png')

if __name__ == '__main__':
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    train()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()

