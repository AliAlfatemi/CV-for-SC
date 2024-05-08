import json
from IPython.display import clear_output
clear_output()
import os

import datasets
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
from transformers import AutoTokenizer, GPT2Config, default_data_collator

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer



if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

os.environ["WANDB_DISABLED"] = "true"

class config:
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "EleutherAI/gpt-j-6b"
    # DECODER = "microsoft/phi-2"
    # DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    VAL_EPOCHS = 1
    LR = 5e-5
    SEED = 42
    MAX_LEN = 128
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    NUM_WORKERS = mp.cpu_count()
    EPOCHS = 3
    IMG_SIZE = (224, 224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95

rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    print("rouge2_precision", round(rouge_output.precision, 4) )
    print("rouge2_recall", round(rouge_output.recall, 4) )
    print("rouge2_fmeasure", round(rouge_output.fmeasure, 4) )
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token




image_transforms = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])




# Load JSON and create DataFrame
with open('/u/erdos/csga/aalfatemi/ImageC/cv/data/annotations/captions_train2017.json') as f:
    data = json.load(f)

# Extract image file names and captions
images = {img['id']: img['file_name'] for img in data['images']}
annotations = [{'image_id': ann['image_id'], 'caption': ann['caption'], 'image': images[ann['image_id']]} for ann in data['annotations']]
df = pd.DataFrame(annotations)

train_df, val_df = train_test_split(df, test_size=0.2)


class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = 50

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        img = Image.open(img_path).convert("RGB")
        # img = np.array(img) / 255.0  
        
        if self.transform is not None:
            img = self.transform(img)  
        
        # No need to call the feature extractor if transforms already handled normalization and tensor conversion
        captions = self.tokenizer(caption, padding='max_length', max_length=self.max_length, truncation=True).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        encoding = {"pixel_values": img, "labels": torch.tensor(captions)}  
        return encoding


train_dataset = ImgDataset(train_df, root_dir="/u/erdos/csga/aalfatemi/ImageC/cv/data/train2017", tokenizer=tokenizer, feature_extractor=feature_extractor, transform=image_transforms)
val_dataset = ImgDataset(val_df, root_dir="/u/erdos/csga/aalfatemi/ImageC/cv/data/train2017", tokenizer=tokenizer, feature_extractor=feature_extractor, transform=image_transforms)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER, config.DECODER)


model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

training_args = Seq2SeqTrainingArguments(
    output_dir='VIT_large_gpt2',
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,
    save_steps=2048,
    warmup_steps=1024,
    learning_rate=config.LR,
    num_train_epochs=config.EPOCHS,
    overwrite_output_dir=True,
    save_total_limit=1,
)


trainer = Seq2SeqTrainer(
    tokenizer=feature_extractor,
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)
trainer.train()
trainer.save_model('VIT_large_gpt2')



