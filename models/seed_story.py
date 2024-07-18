import jax
import jax.numpy as jnp
from flax import linen as nn
from transformers import FlaxViTModel, FlaxLlamaForCausalLM, LlamaTokenizer
from typing import Any, Dict, Optional
from .qformer import QFormer
from .attention_sink import MultimodalAttentionSink

class SEEDStory(nn.Module):
    config: Any

    def setup(self):
        self.vit = FlaxViTModel.from_pretrained(self.config.vit_model_name)
        self.qformer = QFormer(self.config)
        self.attention_sink = MultimodalAttentionSink(
            max_length=self.config.max_length,
            sink_size=self.config.sink_size,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout_rate=self.config.dropout_rate
        )
        self.lm = FlaxLlamaForCausalLM.from_pretrained(self.config.llm_model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.llm_model_name)
        
        self.image_projection = nn.Dense(self.config.hidden_dim)
        self.prompt_projection = nn.Dense(self.config.hidden_dim)

    def encode_image(self, image, training: bool = False):
        features = self.vit(image, training=training).last_hidden_state
        features = self.image_projection(features)
        return self.qformer(features, training=training)

    def generate_story(self, image_features, prompt, max_length=100, generate_kwargs: Optional[Dict] = None):
        input_ids = self.tokenizer.encode(prompt, return_tensors="jax")
        attention_mask = jnp.ones_like(input_ids)
        
        image_embeds = self.lm.variables['params']['model']['embed_tokens'](image_features)
        prompt_embeds = self.prompt_projection(self.lm.variables['params']['model']['embed_tokens'](input_ids))
        inputs_embeds = jnp.concatenate([image_embeds, prompt_embeds], axis=1)
        
        default_generate_kwargs = {
            'max_length': max_length,
            'num_return_sequences': 1,
            'no_repeat_ngram_size': 3,
            'do_sample': True,
            'top_k': 50,
            'top_p': 0.95,
            'temperature': 0.8,
        }
        if generate_kwargs:
            default_generate_kwargs.update(generate_kwargs)
        
        output = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **default_generate_kwargs
        )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def __call__(self, inputs, training: bool = False, generate: bool = True):
        image, prompt = inputs
        image_features = self.encode_image(image, training=training)
        sink_tokens = image_features[:, :self.config.sink_size]
        full_sequence = self.attention_sink(image_features, sink_tokens, training=training)
        
        if generate:
            story = self.generate_story(full_sequence, prompt)
            return story
        else:
            # Return embeddings for further processing or loss computation
            return full_sequence

    def compute_loss(self, inputs, targets):
        image, prompt = inputs
        full_sequence = self(inputs, training=True, generate=False)
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="jax")
        target_ids = self.tokenizer.encode(targets, return_tensors="jax")
        
        lm_output = self.lm(inputs_embeds=full_sequence, labels=target_ids)
        return lm_output.loss

    @classmethod
    def from_pretrained(cls, config, model_path: str):
        model = cls(config)
        variables = jax.tree_util.tree_map(jnp.array, model.load_variables(model_path))
        return model.bind(variables)

    def save_pretrained(self, save_path: str):
        self.save_variables(save_path)