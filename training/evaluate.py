import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training.common_utils import shard
from typing import Any, Dict

@jax.jit
def eval_step(state: train_state.TrainState, batch: Any) -> Dict[str, float]:
    """Perform a single evaluation step."""
    images, prompts, targets = batch
    logits = state.apply_fn({'params': state.params}, (images, prompts), training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == targets)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics

def evaluate_model(state: train_state.TrainState, eval_ds: Any) -> Dict[str, float]:
    """Evaluate the model on the provided dataset."""
    metrics = []
    for batch in eval_ds:
        batch = shard(batch)  # Shard the batch for multi-device evaluation
        metrics.append(eval_step(state, batch))

    # Compute average metrics
    metrics = jax.tree_map(lambda *args: jnp.mean(jnp.array(args)), *metrics)
    return metrics