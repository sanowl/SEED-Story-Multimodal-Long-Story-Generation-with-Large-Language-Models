import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Tuple, List, Optional
from dataclasses import dataclass
from transformers import FlaxViTModel, FlaxLlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
from io import BytesIO
from security import safe_requests

@dataclass
class SEEDStoryConfig:
    image_dim: int = 4096
    query_dim: int = 1024
    num_queries: int = 64
    max_length: int = 2048
    sink_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    layer_scale_init_value: float = 1e-5
    groups: int = 32
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    image_size: int = 224

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image from a file path or URL and preprocess it for the ViT model.
    
    Args:
    image_path (str): Local file path or URL of the image.
    target_size (tuple): Target size for resizing the image.
    
    Returns:
    np.ndarray: Preprocessed image as a numpy array.
    """
    if image_path.startswith(('http://', 'https://')):
        response = safe_requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to RGB if it's not
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Standardize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    return img_array

class GroupNorm(nn.Module):
    groups: int
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        return nn.GroupNorm(num_groups=self.groups, epsilon=self.epsilon)(x)

class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.constant(self.init_values), (self.dim,))
        return x * scale

class QFormerLayer(nn.Module):
    config: SEEDStoryConfig

    @nn.compact
    def __call__(self, x, queries, training=False):
        attn = nn.MultiHeadDotProductAttention(num_heads=self.config.num_heads)
        ffn = lambda x: nn.Dense(self.config.query_dim)(nn.gelu(nn.Dense(4 * self.config.query_dim)(x)))
        
        norm1 = GroupNorm(self.config.groups)
        norm2 = GroupNorm(self.config.groups)
        ls1 = LayerScale(self.config.query_dim, self.config.layer_scale_init_value)
        ls2 = LayerScale(self.config.query_dim, self.config.layer_scale_init_value)
        
        k, v = nn.Dense(self.config.query_dim)(x), nn.Dense(self.config.query_dim)(x)
        attn_output = attn(norm1(queries), k, v)
        queries = queries + nn.Dropout(self.config.dropout_rate, deterministic=not training)(ls1(attn_output))
        ffn_output = ffn(norm2(queries))
        return queries + nn.Dropout(self.config.dropout_rate, deterministic=not training)(ls2(ffn_output))

class QFormer(nn.Module):
    config: SEEDStoryConfig

    @nn.compact
    def __call__(self, x, training=False):
        queries = self.param('queries', nn.initializers.normal(), (self.config.num_queries, self.config.query_dim))
        queries = jnp.tile(queries[None, :, :], (x.shape[0], 1, 1))
        
        for _ in range(self.config.num_layers):
            queries = QFormerLayer(self.config)(x, queries, training)
        return queries

class MultimodalAttentionSink(nn.Module):
    max_length: int
    sink_size: int

    def __call__(self, x, sink_tokens):
        return jnp.concatenate([sink_tokens, x[:, -(self.max_length - self.sink_size):]], axis=1)

class SEEDStory(nn.Module):
    config: SEEDStoryConfig

    def setup(self):
        self.vit = FlaxViTModel.from_pretrained("google/vit-large-patch14-224-in21k")
        self.qformer = QFormer(self.config)
        self.attention_sink = MultimodalAttentionSink(self.config.max_length, self.config.sink_size)
        self.lm = FlaxLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def encode_image(self, image):
        features = self.vit(image).last_hidden_state
        return self.qformer(features)

    def generate_story(self, image_features, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors="jax")
        attention_mask = jnp.ones_like(input_ids)
        
        image_embeds = self.lm.variables['params']['model']['embed_tokens'](image_features)
        inputs_embeds = jnp.concatenate([image_embeds, self.lm.variables['params']['model']['embed_tokens'](input_ids)], axis=1)
        
        output = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )
        
        return self.tokenizer.decode(output[0])

    def __call__(self, inputs, training=False):
        image, prompt = inputs
        image_features = self.encode_image(image)
        sink_tokens = image_features[:, :self.config.sink_size]
        full_sequence = self.attention_sink(image_features, sink_tokens)
        story = self.generate_story(full_sequence, prompt)
        return story

def create_train_state(config, model, rng):
    params = model.init(rng, (jnp.zeros((1, config.image_size, config.image_size, 3)), ""))['params']
    tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, rng):
    def loss_fn(params):
        images, prompts, targets = batch
        logits = state.apply_fn({'params': params}, (images, prompts), rngs={'dropout': rng}, training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits

def evaluate_model(state, eval_ds):
    losses = []
    for batch in eval_ds:
        images, prompts, targets = batch
        logits = state.apply_fn({'params': state.params}, (images, prompts), training=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        losses.append(loss)
    return jnp.mean(jnp.array(losses))

def main():
    config = SEEDStoryConfig()
    model = SEEDStory(config)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(config, model, init_rng)

    # Load and preprocess dataset
    dataset = load_dataset("your_dataset_name")
    train_ds = dataset["train"].shuffle(seed=42).batch(config.batch_size)
    eval_ds = dataset["validation"].batch(config.batch_size)

    # Training loop
    for epoch in range(config.num_epochs):
        # Training
        with tqdm(total=len(train_ds), desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
            for batch in train_ds:
                rng, step_rng = jax.random.split(rng)
                state, loss, _ = train_step(state, batch, step_rng)
                pbar.update(1)
                pbar.set_postfix({"Loss": float(loss)})

        # Evaluation
        eval_loss = evaluate_model(state, eval_ds)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Eval Loss: {eval_loss}")

    # Example usage
    image_path = "example.jpg"  # This can be a local file path or a URL
    image = load_and_preprocess_image(image_path, target_size=(config.image_size, config.image_size))
    image = jnp.array(image)[None, ...]  # Add batch dimension
    prompt = "Once upon a time,"
    generated_story = model.apply({'params': state.params}, (image, prompt))
    print(generated_story)

if __name__ == "__main__":
    main()
