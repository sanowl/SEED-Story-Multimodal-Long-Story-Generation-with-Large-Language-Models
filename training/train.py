import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training.common_utils import shard
from tqdm import tqdm
import wandb
from typing import Any, Callable, Dict, Tuple

def create_train_state(config: Any, model: Any, rng: jnp.ndarray) -> train_state.TrainState:
    """Create initial training state."""
    params = model.init(rng, (jnp.zeros((1, config.image_size, config.image_size, 3)), ""))['params']
    tx = optax.adamw(
        learning_rate=config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=config.weight_decay,
        mask=lambda p: jax.tree_map(lambda x: x.ndim > 1, p)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics

@jax.jit
def train_step(state: train_state.TrainState, batch: Tuple, rng: jnp.ndarray) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """Perform a single training step."""
    def loss_fn(params):
        images, prompts, targets = batch
        logits = state.apply_fn({'params': params}, (images, prompts), rngs={'dropout': rng}, training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch[2])  # batch[2] is targets
    return state, metrics

def train_model(config: Any, model: Any, train_ds: Any, eval_ds: Any, eval_fn: Callable) -> train_state.TrainState:
    """Train the model."""
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(config, model, init_rng)

    # Initialize wandb
    wandb.init(project=config.project_name, config=config.__dict__)

    for epoch in range(config.num_epochs):
        # Training
        train_metrics = []
        with tqdm(total=len(train_ds), desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
            for batch in train_ds:
                rng, step_rng = jax.random.split(rng)
                batch = shard(batch)  # Shard the batch for multi-device training
                state, metrics = train_step(state, batch, step_rng)
                train_metrics.append(metrics)
                pbar.update(1)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

        # Compute average training metrics
        train_metrics = jax.tree_map(lambda *args: jnp.mean(jnp.array(args)), *train_metrics)

        # Evaluation
        eval_metrics = eval_fn(state, eval_ds)

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
        })

        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Eval Loss: {eval_metrics['loss']:.4f}, Eval Accuracy: {eval_metrics['accuracy']:.4f}")

    wandb.finish()
    return state