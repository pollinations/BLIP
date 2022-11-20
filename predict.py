"""
Download the weights in ./checkpoints beforehand for fast inference
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth
"""

import os
from pathlib import Path

import torch
from cog import BasePredictor, Input, Path
from models.blip import blip_decoder
from models.blip_itm import blip_itm
from models.blip_vqa import blip_vqa
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda:0"
        os.system("ls -l /checkpoints")
        self.models = {
            # 'image_captioning': blip_decoder(pretrained='/checkpoints/model_base_caption_capfilt_large.pth',
            #                                  image_size=384, vit='base'),
            'visual_question_answering': blip_vqa(pretrained='/checkpoints/model_vqa.pth',
                                                  image_size=480, vit='base'),
            # 'image_text_matching': blip_itm(pretrained='/checkpoints/model_base_retrieval_coco.pth',
            #                                 image_size=384, vit='base')
        }

    # @cog.input(
    #     "image",
    #     type=Path,
    #     help="input image",
    # )
    # @cog.input(
    #     "task",
    #     type=str,
    #     default='image_captioning',
    #     options=['image_captioning', 'visual_question_answering', 'image_text_matching'],
    #     help="Choose a task.",
    # )
    # @cog.input(
    #     "question",
    #     type=str,
    #     default=None,
    #     help="Type question for the input image for visual question answering task.",
    # )
    # @cog.input(
    #     "caption",
    #     type=str,
    #     default=None,
    #     help="Type caption for the input image for image text matching task.",
    # )
    def predict(self, 
        image: Path = Input(description="input image"),
        task: str = Input(
            description='task',
            default='visual_question_answering',
            choices=['image_captioning', 'visual_question_answering', 'image_text_matching'],        
        ),
        question: str = Input(
            description='question',
            default="What is the age in years of the person?",
        )) -> str:
        im = load_image(image, image_size=480 if task == 'visual_question_answering' else 384, device=self.device)
        model = self.models[task]
        model.eval()
        model = model.to(self.device)

        if task == 'image_captioning':
            with torch.no_grad():
                caption = model.generate(im, sample=False, num_beams=3, max_length=20, min_length=5)
                return 'Caption: ' + caption[0]

        if task == 'visual_question_answering':
            with torch.no_grad():
                answer = model(im, question, train=False, inference='generate')
                # Write answer[0] to /outputs/answer
                try:
                    with open('/outputs/answer', 'w') as f:
                        f.write(answer[0])
                except:
                    pass
                return 'Answer: ' + answer[0]

        # image_text_matching
        itm_output = model(im, caption, match_head='itm')
        itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
        itc_score = model(im, caption, match_head='itc')
        return f'The image and text is matched with a probability of {itm_score.item():.4f}.\n' \
               f'The image feature and text feature has a cosine similarity of {itc_score.item():.4f}.'


def load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image
