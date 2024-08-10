import os
import itertools
import torch
from torch import cuda
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
from peft import PeftModel

from training.dataset.visu import ViSU
from training.losses import CLIPLoss_Positive, CosineDistance
from training.utils.logger import WandbLogger, summarize
from training.validation import validate
from training.utils.checkpointing import CheckpointManager


@torch.enable_grad()
def training(
    text_encoder_ft: PeftModel,
    text_encoder_original: CLIPTextModelWithProjection,
    vision_encoder_ft: PeftModel,
    vision_encoder_original: CLIPVisionModelWithProjection,
    tokenizer: CLIPTokenizer,
    train_dataset: ViSU,
    validation_dataset: ViSU,
    contrastive_loss_function: CLIPLoss_Positive,
    distance_loss_function: CosineDistance,
    lambdas=(1,1,1,1,1,1,1,1),
    batch_size=32,
    lr=1e-5,
    epoches=10,
    gradient_accumulation_steps=1,
    initial_patience=5,
    wandb_activated=False,
    run=None,
    device='cuda',
    checkpoint_saving_path='',
    resume: bool = None,
    clip_backbone='openai/clip-vit-large-patch14',
    debug=False,
    wandb_logger: WandbLogger | None = None
):
    optimizer = torch.optim.Adam((p for n,p in itertools.chain(text_encoder_ft.named_parameters(), vision_encoder_ft.named_parameters()) if 'lora' in n), lr=lr)

    for n,m in itertools.chain(text_encoder_ft.named_parameters(), vision_encoder_ft.named_parameters()): 
        if 'lora' in n: 
            m.requires_grad_(True)
        else: 
            m.requires_grad_(False)
    
    checkpoint_manager = CheckpointManager(
        checkpoint_saving_path,
        text_encoder_ft=text_encoder_ft,
        vision_encoder_ft=vision_encoder_ft,
        optimizer=optimizer
    )

    initial_epoch, best_validation_loss, best_recall_sum, patience = checkpoint_manager.resume(mode='last') if resume else (-1, 9999, 0, initial_patience) 

    if resume:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    for epoch in range(initial_epoch + 1, epoches):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=len(os.sched_getaffinity(0)))

        text_encoder_ft.train().to(device)
        text_encoder_original.train().to(device)
        vision_encoder_ft.train().to(device)
        vision_encoder_original.train().to(device)

        lambdas = lambdas.to(device).float()

        training_start = cuda.Event(enable_timing=True)
        validation_start = cuda.Event(enable_timing=True)
        training_end = cuda.Event(enable_timing=True)
        validation_end = cuda.Event(enable_timing=True)

        training_start.record()

        for idx, (safe_image, nsfw_image, safe_caption, nsfw_caption) in enumerate(tqdm(train_dataloader)):

            # * Encoding inputs
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

            with torch.no_grad():
                reference_text_embeddings = text_encoder_original(**text_safe_ids).text_embeds
                reference_vision_embeddings = vision_encoder_original(**{'pixel_values': safe_image}).image_embeds

            model_vision_safe_embeddings = vision_encoder_ft(**{'pixel_values': safe_image}).image_embeds
            model_vision_nsfw_embeddings = vision_encoder_ft(**{'pixel_values': nsfw_image}).image_embeds
            
            # * Losses
            # S --> text_safe,
            # U --> text_nsfw,
            # V --> vision_safe,
            # G --> vision_nsfw

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

            losses = torch.cat(
                [
                    x[(None,)+(...,)] \
                        for x in (text_safe_loss, text_nsfw_loss, vision_safe_loss, vision_nsfw_loss, S_Vref_contrastive_loss['loss'], U_Vref_contrastive_loss['loss'], V_Sref_contrastive_loss['loss'], G_Sref_contrastive_loss['loss'])
                ]
            )
            training_loss = (lambdas @ losses[(None,)+(...,)].T) / lambdas.numel()


            if (idx + 1) % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            training_loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()

            if wandb_activated and run is not None:
                wandb_logger.log_training_iteration(
                    idx, text_safe_loss, text_nsfw_loss, vision_safe_loss, vision_nsfw_loss, S_Vref_contrastive_loss, U_Vref_contrastive_loss, V_Sref_contrastive_loss, G_Sref_contrastive_loss, training_loss
                )

            if debug:
                break

        training_end.record()
        cuda.synchronize()
        training_time = training_start.elapsed_time(training_end)/1000
        validation_start.record()
        print('Validating...')
        this_recall_sum, this_recalls, this_validation_loss = validate(
            text_encoder_ft=text_encoder_ft,
            text_encoder_original=text_encoder_original,
            vision_encoder_ft=vision_encoder_ft,
            vision_encoder_original=vision_encoder_original,
            tokenizer=tokenizer,
            validation_dataset=validation_dataset,
            contrastive_loss_function=contrastive_loss_function,
            distance_loss_function=distance_loss_function,
            lambdas=lambdas,
            batch_size=batch_size,
            clip_backbone=clip_backbone,
            wandb_activated=wandb_activated,
            run=run,
            device=device,
            debug=debug,
            wandb_logger=wandb_logger
        )
        validation_end.record()
        cuda.synchronize()
        validation_time = validation_start.elapsed_time(validation_end)/1000
        
        # * Evaluating exit criterion
        is_better_recall = this_recall_sum > best_recall_sum
        is_better_loss = this_validation_loss < best_validation_loss

        if not is_better_loss and not is_better_recall:
            patience -= 1
            if patience == 0:
                break

        else:
            patience = initial_patience
            
            if is_better_loss:
                best_validation_loss = this_validation_loss
                if wandb_activated and run is not None:
                    wandb_logger.log(best_validation_loss=best_validation_loss)

                checkpoint_manager.best('validation-loss', epoch, best_validation_loss, best_recall_sum, patience)

            if is_better_recall:
                best_recall_sum = this_recall_sum
                if wandb_activated and run is not None:
                    wandb_logger.log(best_recall_sum=best_recall_sum)

                checkpoint_manager.best('recall', epoch, best_validation_loss, best_recall_sum, patience)

        if wandb_activated and run is not None:
            wandb_logger.log_patience(patience)
        
        checkpoint_manager.step(epoch, best_validation_loss, best_recall_sum, patience)

        summarize(
            epoch, patience, training_loss, this_validation_loss, this_recalls, best_recall_sum, best_validation_loss, training_time, validation_time, str(checkpoint_saving_path)
        )