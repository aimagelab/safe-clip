import os
import torch
from tqdm import tqdm
from training.losses import CLIPLoss_Positive, CosineDistance
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader

from training.dataset.visu import ViSU
from training.recall_computation import compute_recall
from training.utils.logger import WandbLogger

@torch.inference_mode()
def validate(
    text_encoder_ft: PeftModel,
    text_encoder_original: CLIPTextModelWithProjection,
    vision_encoder_ft: PeftModel,
    vision_encoder_original: CLIPVisionModelWithProjection,
    tokenizer: CLIPTokenizer,
    validation_dataset: ViSU,
    contrastive_loss_function: CLIPLoss_Positive,
    distance_loss_function: CosineDistance,
    lambdas=(1,1,1,1,1,1,1,1),
    batch_size=32,
    clip_backbone='openai/clip-vit-large-patch14',
    wandb_activated=False,
    run=None,
    device='cuda',
    debug=False,
    wandb_logger: WandbLogger | None = None
):
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=len(os.sched_getaffinity(0)))

    text_encoder_ft.eval().to(device)
    text_encoder_original.eval().to(device)
    vision_encoder_ft.eval().to(device)
    vision_encoder_original.eval().to(device)
    lambdas = lambdas.to(device).float()

    text_safe_loss_cumulative = 0
    text_nsfw_loss_cumulative = 0
    vision_safe_loss_cumulative = 0
    vision_nsfw_loss_cumulative = 0
    S_Vref_contrastive_loss_cumulative = 0
    U_Vref_contrastive_loss_cumulative = 0
    V_Sref_contrastive_loss_cumulative = 0
    G_Sref_contrastive_loss_cumulative = 0
    validation_loss = 0

    for (safe_image, nsfw_image, safe_caption, nsfw_caption) in tqdm(validation_dataloader):
        this_batch_size = len(safe_caption)
        
        text_safe_ids = tokenizer(safe_caption, return_tensors='pt', padding='max_length', truncation=True)
        text_safe_ids['input_ids'] = text_safe_ids['input_ids'].to(device)
        text_safe_ids['attention_mask'] = text_safe_ids['attention_mask'].to(device)
        text_nsfw_ids = tokenizer(nsfw_caption, return_tensors='pt', padding='max_length', truncation=True)
        text_nsfw_ids['input_ids']  = text_nsfw_ids['input_ids'].to(device)
        text_nsfw_ids['attention_mask'] = text_nsfw_ids['attention_mask'].to(device)
        
        safe_image = safe_image.to(device)
        nsfw_image = nsfw_image.to(device)

        model_text_safe_embeddings = text_encoder_ft(**text_safe_ids).text_embeds
        model_text_nsfw_embeddings = text_encoder_ft(**text_nsfw_ids).text_embeds

        reference_text_embeddings = text_encoder_original(**text_safe_ids).text_embeds
        reference_vision_embeddings = vision_encoder_original(**{'pixel_values': safe_image}).image_embeds
        
        model_vision_safe_embeddings = vision_encoder_ft(**{'pixel_values': safe_image}).image_embeds
        model_vision_nsfw_embeddings = vision_encoder_ft(**{'pixel_values': nsfw_image}).image_embeds

        # defined on paper as "loss redirection 1"
        U_Vref_contrastive_loss = contrastive_loss_function(model_text_nsfw_embeddings, reference_vision_embeddings)
        G_Sref_contrastive_loss = contrastive_loss_function(reference_text_embeddings, model_vision_nsfw_embeddings) 

        # defined on paper as "loss redirection 2"
        text_nsfw_loss = distance_loss_function(model_text_nsfw_embeddings, reference_text_embeddings).mean(dim=0)
        vision_nsfw_loss = distance_loss_function(model_vision_nsfw_embeddings, reference_vision_embeddings).mean(dim=0)

        # defined on paper as "loss preservation 1"
        text_safe_loss = distance_loss_function(model_text_safe_embeddings, reference_text_embeddings).mean(dim=0)
        vision_safe_loss = distance_loss_function(model_vision_safe_embeddings, reference_vision_embeddings).mean(dim=0)

        # defined on paper as "loss preservation 2"
        S_Vref_contrastive_loss = contrastive_loss_function(model_text_safe_embeddings, reference_vision_embeddings)
        V_Sref_contrastive_loss = contrastive_loss_function(reference_text_embeddings, model_vision_safe_embeddings)

        # Keep cumulative losses during validation cycle
        text_safe_loss_cumulative += (text_safe_loss * this_batch_size).cpu()
        text_nsfw_loss_cumulative += (text_nsfw_loss * this_batch_size).cpu()
        vision_safe_loss_cumulative += (vision_safe_loss * this_batch_size).cpu()
        vision_nsfw_loss_cumulative += (vision_nsfw_loss * this_batch_size).cpu()

        S_Vref_contrastive_loss_cumulative += (S_Vref_contrastive_loss['loss'] * this_batch_size).cpu()
        U_Vref_contrastive_loss_cumulative += (U_Vref_contrastive_loss['loss'] * this_batch_size).cpu()
        V_Sref_contrastive_loss_cumulative += (V_Sref_contrastive_loss['loss'] * this_batch_size).cpu()
        G_Sref_contrastive_loss_cumulative += (G_Sref_contrastive_loss['loss'] * this_batch_size).cpu()

        losses = torch.cat(
            [
                x[(None,)+(...,)] \
                    for x in (text_safe_loss, text_nsfw_loss, vision_safe_loss, vision_nsfw_loss, S_Vref_contrastive_loss['loss'], U_Vref_contrastive_loss['loss'], V_Sref_contrastive_loss['loss'], G_Sref_contrastive_loss['loss'])
            ]
        )

        validation_loss += (lambdas @ losses[(None,)+(...,)].T) / lambdas.numel()
        if debug:
            break 

    if wandb_activated and run is not None:
        wandb_logger.log_validation(len(validation_dataset), text_safe_loss_cumulative, text_nsfw_loss_cumulative, vision_safe_loss_cumulative, vision_nsfw_loss_cumulative, S_Vref_contrastive_loss_cumulative, U_Vref_contrastive_loss_cumulative, V_Sref_contrastive_loss_cumulative, G_Sref_contrastive_loss_cumulative, validation_loss, this_batch_size)

    print('Computing Recall...')
    S_V_recall, S_G_recall, U_V_recall, U_G_recall = compute_recall(text_encoder_ft, vision_encoder_ft, clip_backbone=clip_backbone, batch_size=batch_size, dataset=validation_dataset, debug=debug)
    recall_sum = S_V_recall[0][0] + S_V_recall[1][0] + S_G_recall[0][0] + S_G_recall[1][0] + U_V_recall[0][0] + U_V_recall[1][0] # we do not care to evaluate on U_G_recall

    if wandb_activated and run is not None:
        wandb_logger.log_recall([S_V_recall, S_G_recall, U_V_recall, U_G_recall], recall_sum)

    text_safe_ids['input_ids'] = text_safe_ids['input_ids'].to('cpu')
    text_safe_ids['attention_mask'] = text_safe_ids['attention_mask'].to('cpu')
    text_nsfw_ids['input_ids'] = text_nsfw_ids['input_ids'].to('cpu')
    text_nsfw_ids['attention_mask'] = text_nsfw_ids['attention_mask'].to('cpu')
    
    safe_image = safe_image.to('cpu')
    nsfw_image = nsfw_image.to('cpu')
    
    return (recall_sum, [S_V_recall, S_G_recall, U_V_recall, U_G_recall], validation_loss / len(validation_dataset))