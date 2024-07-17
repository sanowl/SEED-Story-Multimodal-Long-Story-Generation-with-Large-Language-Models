
# SEED-Story: Multimodal Long Story Generation with Large Language Models

## Overview

SEED-Story is a TensorFlow implementation of a multimodal story generation model based on the paper "SEED-Story: Multimodal Long Story Generation with Large Language Model". This model combines vision and language models to generate coherent, long-form stories from images and text prompts.

## Key Components

1. Vision Transformer (ViT): Extracts features from input images.
2. QFormer: Transforms image features for language model consumption.
3. Large Language Model (LLaMA): Generates story text.
4. Multimodal Attention Sink: Enables efficient processing of long sequences.

## Installation

```bash
pip install tensorflow tensorflow-addons transformers
```

## Usage

```python
from seed_story import SEEDStory

model = SEEDStory()

image = load_and_preprocess_image("example.jpg")
prompt = "Once upon a time,"
generated_story = model((image, prompt))
print(generated_story)
```

## Model Architecture

### QFormer

The QFormer uses learnable queries to transform image features:

- Multiple attention layers with feed-forward networks
- Positional encodings for queries
- Group Normalization for improved training stability
- Layer Scale to manage signal propagation

### Multimodal Attention Sink

Enables efficient processing of long sequences by retaining key tokens in the attention computation.

## Training

The model is trained end-to-end, including fine-tuning of the ViT layers. AdamW optimizer is used with weight decay for regularization.

## Implementation Details

1. ViT Fine-tuning: The last 4 layers of the ViT model are set to be trainable.
2. Positional Encodings: Added to queries in the QFormer for sequence awareness.
3. Group Normalization: Replaces Layer Normalization for better batch independence.
4. Layer Scale: Improves training stability in deep networks.

## Future Work

- Integration with a TensorFlow-compatible image generation model (e.g., SDXL equivalent)
- Hyperparameter tuning for optimal performance
- Extensive evaluation on diverse datasets

## Acknowledgements

We would like to express our sincere gratitude to the authors of the original SEED-Story paper:

Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen

Their groundbreaking work in multimodal story generation has been instrumental in the development of this implementation.  I encourage readers to refer to the original paper for a comprehensive understanding of the SEED-Story concept and methodology.

## Citation

If you use this implementation in your research, please cite the original SEED-Story paper.
