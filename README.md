

# SEED-Story: Multimodal Long Story Generation with Large Language Models (JAX Implementation)

## Overview

SEED-Story is a JAX/Flax implementation of a multimodal story generation model based on the paper "SEED-Story: Multimodal Long Story Generation with Large Language Model". This model combines vision and language models to generate coherent, long-form stories from images and text prompts.

## Key Components

1. Vision Transformer (ViT): Extracts features from input images.
2. QFormer: Transforms image features for language model consumption.
3. Large Language Model (LLaMA): Generates story text.
4. Multimodal Attention Sink: Enables efficient processing of long sequences.

## Installation

```bash
pip install jax jaxlib flax optax transformers datasets tqdm pillow requests
```

## Usage

```python
from seed_story import SEEDStory, SEEDStoryConfig, load_and_preprocess_image

config = SEEDStoryConfig()
model = SEEDStory(config)

image_path = "example.jpg"  # Can be a local file path or URL
image = load_and_preprocess_image(image_path, target_size=(config.image_size, config.image_size))
image = jnp.array(image)[None, ...]  # Add batch dimension
prompt = "Once upon a time,"
generated_story = model.apply({'params': model.params}, (image, prompt))
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

The model is trained end-to-end using JAX/Flax. AdamW optimizer from Optax is used with weight decay for regularization.

## Implementation Details

1. ViT Fine-tuning: Uses the Flax version of the ViT model.
2. Positional Encodings: Added to queries in the QFormer for sequence awareness.
3. Group Normalization: Replaces Layer Normalization for better batch independence.
4. Layer Scale: Improves training stability in deep networks.
5. JAX/Flax: Utilizes JAX's automatic differentiation and JIT compilation for improved performance.

## Future Work

- Integration with a JAX-compatible image generation model
- Hyperparameter tuning for optimal performance
- Extensive evaluation on diverse datasets
- Exploration of JAX-specific optimizations for further performance improvements

## Acknowledgements

We would like to express our sincere gratitude to the authors of the original SEED-Story paper:

Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen

Their groundbreaking work in multimodal story generation has been instrumental in the development of this implementation.

I encourage readers to refer to the original paper for a comprehensive understanding of the SEED-Story concept and methodology.

## Citation

If you use this implementation in your research, please cite the original SEED-Story paper.

## Notes on JAX Implementation

This JAX implementation offers several advantages:

1. Improved performance through JAX's JIT compilation
2. Easy parallelization across multiple devices
3. Functional programming style for cleaner, more maintainable code
4. Seamless integration with other JAX-based libraries in the ML ecosystem