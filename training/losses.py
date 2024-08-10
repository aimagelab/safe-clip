import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss_Positive(nn.Module):
    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.labels = None
        self.last_local_batch_size = None
        self.logit_scale = (torch.ones([]) * 4.6052).exp()

    def forward(self, image_feats, text_feats, current_batch=None, validation=False):
        image_embed = image_feats
        text_embed = text_feats
        local_batch_size = text_embed.shape[0] if current_batch is None else current_batch
        global_batch_size = local_batch_size

        if local_batch_size != self.last_local_batch_size:
            self.labels = torch.arange(local_batch_size, device=image_embed.device)   
            self.last_local_batch_size = local_batch_size

        # Normalize features
        image_embed = image_embed / image_embed.norm(dim=1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)

        image_embed_all = image_embed
        text_embed_all = text_embed

        # Cosine similarity as logits
        logits_per_image = self.logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = self.logit_scale * text_embed @ image_embed_all.t()

        # Classic CLIP loss
        loss_clip = 0.5 * (
            F.cross_entropy(logits_per_image[:local_batch_size, :global_batch_size] / self.temperature, self.labels)
            + F.cross_entropy(logits_per_text[:local_batch_size, :global_batch_size] / self.temperature, self.labels)
        )
        
        return {
            'loss': loss_clip, 
            'values': {
                'image': logits_per_image[:local_batch_size, :local_batch_size], 
                'text': logits_per_text[:local_batch_size, :local_batch_size]
            },
            'references': self.labels
        }


class CosineDistance(torch.nn.CosineSimilarity):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        return 1 - super().__call__(x, y)