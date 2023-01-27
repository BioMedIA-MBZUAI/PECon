import shutil
import torch.nn as nn
import torch
torch.manual_seed(0)

def save_ckp(state, is_best, checkpoint_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model, optimizer = None):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    
    # initialize optimizer from checkpoint to optimizer
    
    #optimizer.load_state_dict(checkpoint['optimizer'])
    
    # initialize valid_loss_min from checkpoint to valid_loss_min
    
    # valid_acc_min = checkpoint['valid_acc_max']
    # return model, optimizer, epoch value, min validation loss 
    #return model, optimizer, checkpoint['epoch'], valid_acc_min

    return model