import json
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from transformers import CLIPProcessor


filenames = {
    'train': {'ailb': 'visu_dpo_train_nsfw_unfiltered_images_ailb.json', 'leonardo': 'visu_dpo_train_nsfw_unfiltered_images.json'},
    'validation': {'ailb': 'visu_validation_dpo_nsfw_unfiltered_images_ailb.json', 'leonardo': 'visu_validation_dpo_corrected_nsfw_images.json'},
    'test': {'ailb': 'visu_test_dpo_nsfw_unfiltered_images_ailb.json', 'leonardo': 'visu_test_dpo_corrected_nsfw_images.json'}
}

def load_cap_json(train_file):
    with open(train_file, 'r') as f:
        texts_json = json.load(f)
    return texts_json

class ViSU(Dataset):
    def __init__(
        self, root, coco_root, split='train', clip_backbone='openai/clip-vit-large-patch14',
    ):
        self.root = root
        self.split = split
        self.coco_root = coco_root
        self.clip_backbone = clip_backbone
        self.imageprocessor = CLIPProcessor.from_pretrained(self.clip_backbone)
        self.cluster = 'leonardo' if 'leonardo' in str(Path(__file__)) else 'ailb'
        self.filename = filenames[self.split][self.cluster]
        self.path = Path(self.root) / self.split / self.filename
        self.data_json = load_cap_json(self.path)
        
        if isinstance(self.data_json, dict) and 'data' in self.data_json.keys():
            self.dataset_info = self.data_json['info']
            self.data_json = self.data_json['data']

        self.coco_train_root = Path(self.coco_root) / 'train2017'
        self.coco_val_root = Path(self.coco_root) / 'val2017'
        self.coco_test_root = Path(self.coco_root) / 'test2017'
        self.coco_imgs_filename_pattern = "*" * 12 + ".jpg"

        self.data_json =  {int(k):v for k,v in self.data_json.items()}

    def img_id_to_path(self, idx):
        f = None
        for coco_images_root in (self.coco_train_root, self.coco_val_root, self.coco_test_root):

            f = Path(coco_images_root) / (
                self.coco_imgs_filename_pattern.replace(
                    '*' * self.coco_imgs_filename_pattern.count('*'), 
                    str(idx).zfill(self.coco_imgs_filename_pattern.count('*'))
                )
            )
            if Path(f).is_file():
                return f

        raise ValueError(f'Image {idx} not found in any of the splits.')
        
    def __getitem__(self, index):
        element = self.data_json[index]

        if 'coco_id' in element.keys():
            safe_image_path, nsfw_image_path, safe_caption, nsfw_caption = (
                self.img_id_to_path(element['coco_id']),
                element['nsfw_im_path'],
                element['safe'],
                element['nsfw']
            )
            
        safe_image = Image.open(safe_image_path)
        safe_image = self.imageprocessor(images=safe_image, return_tensors="pt")['pixel_values'].squeeze()
        nsfw_image = Image.open(nsfw_image_path)
        nsfw_image = self.imageprocessor(images=nsfw_image, return_tensors="pt")['pixel_values'].squeeze()

        return (safe_image, nsfw_image, safe_caption, nsfw_caption)
    
    def __len__(self):
        return len(self.data_json)

class ViSUPrompts(ViSU):
    def __getitem__(self, index):
        element = self.data_json[index]
        safe, nsfw, tag = element['safe'], element['nsfw'], element['tag']

        return (safe, nsfw, tag)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]