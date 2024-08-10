import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-backbone", type=str, help="Path to the CLIP backbone", default='openai/clip-vit-large-patch14')
    parser.add_argument("--lora-r", type=int, help="LoRa bottleneck dimension", default=16)
    parser.add_argument("--epoches", type=int, help="Epoches number", default=50)
    parser.add_argument("--lr", type=float, help="Learning rate value", default=1e-3)
    parser.add_argument("--wandb-activated", action="store_true", help="WandB enabled", default=False)
    parser.add_argument("--wandb-config", type=str, help="WandB configs [str but in dict shape!]", default="{'project': 'safe-clip', 'name': 'clip_ft_v00', 'entity': 'unl4xai'}")
    parser.add_argument("--wandb-run-id", type=str, help="WandB run ID to resume it", default='None')
    parser.add_argument("--visu-dataset-root", type=str, help="Root of the ViSU dataset", required=True)
    parser.add_argument("--coco-dataset-root", type=str, help="Root of the COCO dataset", required=True)
    parser.add_argument("-l", "--lambdas", type=str, help="Lambda weights values in a tuple", default='(0.1,0.1,0.1,0.2,0.25,0.25,0.25,0.5)')
    parser.add_argument("--bs", type=int, help="Batch size", default=128)
    parser.add_argument("--device", type=str, help="Chosen device", choices=('cuda','cpu'), default='cuda')
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Number of gradient accumulation steps", default=1)
    parser.add_argument("--initial-patience", type=int, help="Initial patience value", default=5)
    parser.add_argument("--checkpoint-saving-root", type=str, help="Root where to save the best checkpoint", default='checkpoints')
    parser.add_argument("--resume", action="store_true", help="Resume training", default=False)
    parser.add_argument("--resume-checkpoints-path", type=str, help="Resume checkpoints path", default='None')
    parser.add_argument("--debug", action="store_true", help="Debug mode", default=False)
    args = parser.parse_args()

    return args