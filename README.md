# Image Captioning with Vision Transformers and GPT-J

Image-captioning research implementation combining a ViT image encoder with a
GPT-J language decoder through Hugging Face's `VisionEncoderDecoderModel`.

**Paper:** [Vision-Language Integration for Image Captioning Using Vision Transformers and GPT-J](https://doi.org/10.1007/978-3-032-18474-0_8)

**Authors:** Ali Alfatemi, Mohamed Rahouti, Mohammed Aledhari, Nasir Ghani, Abdellah Chehri, Gwanggil Jeon

**Publication:** Applied Imagery Pattern Recognition (AIPR 2025), LNCS 16446, pp. 101–114 (2026).

## Implementation

[main.py](main.py) prepares COCO captions, constructs image-caption datasets,
configures the encoder-decoder model, and trains with `Seq2SeqTrainer`.
Evaluation computes ROUGE-2 precision, recall, and F-measure.

The default configuration uses `google/vit-base-patch16-224` and
`EleutherAI/gpt-j-6b`, three training epochs, and a batch size of eight.
Model and training parameters are defined in the script's `config` class.

## Environment

Dependencies include PyTorch, Torchvision, Transformers, Datasets, Pillow,
Pandas, NumPy, scikit-learn, Matplotlib, tqdm, and IPython. The script uses the
legacy Hugging Face `datasets.load_metric` and `ViTFeatureExtractor` APIs;
select dependency versions compatible with those interfaces.

## Run

```bash
git clone https://github.com/AliAlfatemi/CV-for-SC.git
cd CV-for-SC
```

Prepare the COCO 2017 training images and caption annotations. In `main.py`,
set the annotation-file path and both `ImgDataset` image-root paths to the local
dataset. Review the model, tokenizer, and training configuration for the selected
environment, then run:

```bash
python main.py
```

The trainer writes checkpoints and the final model to `VIT_large_gpt2`.

## Citation

See [CITATION.cff](CITATION.cff) for the paper citation and complete author list.
