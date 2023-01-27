import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F




class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        #self.logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logits_scale):
        
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = image_features, text_features
            if self.local_loss:
                logits_per_image = logits_scale * image_features @ all_text_features.T
                logits_per_text = logits_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logits_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logits_scale * image_features @ text_features.T
            logits_per_text = logits_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss