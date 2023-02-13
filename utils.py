import torch

def resume_checkpoint(model, ckpt_path, device):
    """Resume training from a checkpoint.
    Args:
        ckpt_path: Path to checkpoint to resume from.
        device: Device to load checkpoint to.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_optimizer(optimizer, ckpt_path, device):
    """Load optimizer from checkpoint.
    Args:
        ckpt_path: Path to checkpoint to load optimizer from.
        device: Device to load checkpoint to.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer

def load_scheduler(scheduler, ckpt_path, device):
    """Load scheduler from checkpoint.
    Args:
        ckpt_path: Path to checkpoint to load scheduler from.
        device: Device to load checkpoint to.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    scheduler.load_state_dict(checkpoint['scheduler'])
    return scheduler

def start_epoch(ckpt_path, device):
    """Get epoch to start training from.
    Args:
        ckpt_path: Path to checkpoint to load epoch from.
        device: Device to load checkpoint to.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    return checkpoint['epoch']

def load_exp_name(ckpt_path, device):
    """Get experiment name from checkpoint.
    Args:
        ckpt_path: Path to checkpoint to load experiment name from.
        device: Device to load checkpoint to.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    return checkpoint['exp_name']
