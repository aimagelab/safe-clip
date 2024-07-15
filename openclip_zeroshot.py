# load safe clip model with openclip

from PIL import Image
import requests
import open_clip
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import hf_hub_download

class QuickGELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.tensor(1.702, dtype=torch.float32)
        
    def forward(self, x):
        return x * torch.sigmoid(self.value * x)

def replace_activation(model):
    for pt_layer in model.transformer.resblocks:
        pt_layer.mlp.gelu = QuickGELU()
    
    for pt_layer in model.visual.transformer.resblocks:
        pt_layer.mlp.gelu = QuickGELU()

    return model

processor_hf = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
file_path = hf_hub_download(repo_id='aimagelab/safeclip_vit-l_14', filename='open_clip_pytorch_model.bin')
model_oc, train_preprocess_oc, preprocess_oc = open_clip.create_model_and_transforms('ViT-L/14', pretrained=file_path)
model_oc= replace_activation(model_oc)

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor_hf(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
expanded_tensor = []
for el in inputs.input_ids:
    expanded_tensor.append(F.pad(el, (0, 77 - el.size(0))))
new_input_ids = torch.stack(expanded_tensor)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model_oc.encode_image(inputs.pixel_values)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    text_features = model_oc.encode_text(new_input_ids)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
