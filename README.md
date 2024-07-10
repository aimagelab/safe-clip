<div align="center">
  <h1>Safe-CLIP: Removing NSFW Concepts from</br>Vision-and-Language Models</h1>
  
</div>

This repository contains the reference code for the paper [**Safe-CLIP: Removing NSFW Concepts from Vision-and-Language Models**](https://arxiv.org/abs/2311.16254).
<br></br>
<p align="center">
  <img src="imgs/safe-clip-figure.png" alt="Safe-CLIP" width="820" />
</p> 

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Text-to-Image](#text-to-image)
4. [Image-to-Text](#image-to-text)
5. [Citation](#citation)


## Overview
Source code and trained model will be released soon.

## Installation
To create the conda environment named safe-clip use the following instructions.
With this environment you have all the packages to run the code inside this repo. 
```
conda create -n safe-clip python==3.9
conda activate safe-clip
pip install -r requirements.txt
```

## Text-to-Image
Section to perform Text-to-Image generation, starting from a user text input.

We provide a code snippet to generate images without NSFW contents.


```python
conda activate safe-clip

out_dir="your_path"
cd "$out_dir/safe-clip"

python -u safeclip_t2i_generation_pipeline.py
```
If you prefer we release also a [google colab notebook](https://colab.research.google.com/drive/1Gz2333WX6U7veCUKYwhXF8dXp4UCeLoU?usp=sharing), with the same code.
In this way also if you don't have GPUs you can try our Safe-CLIP model. 

## Image-to-Text
Here we introduce the Image-to-Text generation code.

The code snippet use the LLaVA architecture to answer questions to an image.
The user can specificy an image `image-file` and a question `query` through the parameter of the script. Moreover, mutually exclusive you can use two different architecture.
`paper_model` use one of the [first architecture of LLaVA](https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview), with Lllama 2 as LLM and the image is processed at the resolution of 224px.
`llava_15` use [LLaVA 1.5](https://huggingface.co/liuhaotian/llava-v1.5-13b) with Vicuna as LLM and the image is considered at the resolution of 336px.

```python
conda activate safe-clip

out_dir="your_path"
cd "$out_dir/safe-clip"

python -u LLaVA_generation/main.py --paper_model
```

# Citation

Please cite with the following BibTeX:
```
@article{poppi2024removing,
  title={{Safe-CLIP: Removing NSFW Concepts from Vision-and-Language Models}},
  author={Poppi, Samuele and Poppi, Tobia and Cocchi, Federico and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  journal={arXiv preprint arXiv:2311.16254},
  year={2024}
}
```
