import os
import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from transformers import CLIPTokenizer


device = 'cuda'

def recall(temb, vemb, K=(1,5,10,20)):
    num_text = temb.shape[0]
    num_im = vemb.shape[0]
    text_to_image_map = image_to_text_map = torch.LongTensor(tuple(i for i in range(num_text)))#.unsqueeze(-1)

    # text-to-image recall
    print("Text-to-image recall...")
    
    dist_matrix = temb.cpu() @ vemb.cpu().T  # dist_matrix[i] gives logits for ith text

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(text_to_image_map.device)

    text_to_image_recall = []

    for k in K:
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    # image-to-text recall
    print("Image-to-text recall...")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(text_to_image_map.device)

    image_to_text_recall = []

    for k in K:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.eq(topk, image_to_text_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)#

    print("Done.")
    return text_to_image_recall, image_to_text_recall

def encode_dataset(text_encoder, visual_encoder, dataset, clip_backbone='openai/clip-vit-large-patch14', batch_size=32, debug=False):
    tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)
    if debug:
        dataset = Subset(dataset, range(20))

    if debug:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=len(os.sched_getaffinity(0)), pin_memory=True)

    all_text_safe_embeddings = []
    all_text_nsfw_embeddings = []
    all_visual_safe_embeddings = []
    all_visual_nsfw_embeddings = []

    with torch.inference_mode():
        for (safe_image, nsfw_image, safe_caption, nsfw_caption) in tqdm.tqdm(dataloader):
            safe_ids = tokenizer(safe_caption, return_tensors='pt', padding='max_length', truncation=True)
            safe_ids['input_ids'] = safe_ids['input_ids'].to(device)
            safe_ids['attention_mask'] = safe_ids['attention_mask'].to(device)
            nsfw_ids = tokenizer(nsfw_caption, return_tensors='pt', padding='max_length', truncation=True)
            nsfw_ids['input_ids']  = nsfw_ids['input_ids'].to(device)
            nsfw_ids['attention_mask'] = nsfw_ids['attention_mask'].to(device)
            
            text_safe_embeddings = text_encoder(**safe_ids)
            safe_ids = safe_ids.to('cpu')
            text_safe_embeddings.text_embeds = text_safe_embeddings.text_embeds.to('cpu')
            text_safe_embeddings.last_hidden_state = text_safe_embeddings.last_hidden_state.to('cpu')

            text_nsfw_embeddings = text_encoder(**nsfw_ids)
            nsfw_ids = nsfw_ids.to('cpu')
            text_nsfw_embeddings.text_embeds = text_nsfw_embeddings.text_embeds.to('cpu')
            text_nsfw_embeddings.last_hidden_state = text_nsfw_embeddings.last_hidden_state.to('cpu')
            safe_image = safe_image.to(device)
            nsfw_image = nsfw_image.to(device)
            
            visual_safe_embeddings = visual_encoder(**{'pixel_values': safe_image})
            safe_image = safe_image.to('cpu')
            visual_safe_embeddings.image_embeds = visual_safe_embeddings.image_embeds.to('cpu')
            visual_safe_embeddings.last_hidden_state = visual_safe_embeddings.last_hidden_state.to('cpu')

            visual_nsfw_embeddings = visual_encoder(**{'pixel_values': nsfw_image})
            nsfw_image = nsfw_image.to('cpu')
            visual_nsfw_embeddings.image_embeds = visual_nsfw_embeddings.image_embeds.to('cpu')
            visual_nsfw_embeddings.last_hidden_state = visual_nsfw_embeddings.last_hidden_state.to('cpu')

            all_text_safe_embeddings.append(text_safe_embeddings.text_embeds)
            all_text_nsfw_embeddings.append(text_nsfw_embeddings.text_embeds)
            all_visual_safe_embeddings.append(visual_safe_embeddings.image_embeds)
            all_visual_nsfw_embeddings.append(visual_nsfw_embeddings.image_embeds)

        all_text_safe_embeddings = torch.cat(all_text_safe_embeddings, 0)
        all_text_nsfw_embeddings = torch.cat(all_text_nsfw_embeddings, 0)
        all_visual_safe_embeddings = torch.cat(all_visual_safe_embeddings, 0)
        all_visual_nsfw_embeddings = torch.cat(all_visual_nsfw_embeddings, 0)
    
    
    return all_text_safe_embeddings, all_text_nsfw_embeddings, all_visual_safe_embeddings, all_visual_nsfw_embeddings

def compute_recall(text_encoder, visual_encoder, dataset, clip_backbone='openai/clip-vit-large-patch14', batch_size=32, debug=False):
    
    all_feats = encode_dataset(text_encoder, visual_encoder, dataset, clip_backbone=clip_backbone, batch_size=batch_size, debug=debug)
    all_text_safe_embeddings, all_text_nsfw_embeddings, all_visual_safe_embeddings, all_visual_nsfw_embeddings = all_feats

    K=(1,5,10,20)
    
    all_text_safe_embeddings = all_text_safe_embeddings / all_text_safe_embeddings.norm(dim=-1, keepdim=True)
    all_text_nsfw_embeddings = all_text_nsfw_embeddings / all_text_nsfw_embeddings.norm(dim=-1, keepdim=True)
    all_visual_safe_embeddings = all_visual_safe_embeddings / all_visual_safe_embeddings.norm(dim=-1,keepdim=True)
    all_visual_nsfw_embeddings = all_visual_nsfw_embeddings / all_visual_nsfw_embeddings.norm(dim=-1,keepdim=True)

    S_V_ranks = recall(all_text_safe_embeddings, all_visual_safe_embeddings, K=K)
    S_G_ranks = recall(all_text_safe_embeddings, all_visual_nsfw_embeddings, K=K)
    U_V_ranks = recall(all_text_nsfw_embeddings, all_visual_safe_embeddings, K=K)
    U_G_ranks = recall(all_text_nsfw_embeddings, all_visual_nsfw_embeddings, K=K)

    return S_V_ranks, S_G_ranks, U_V_ranks, U_G_ranks