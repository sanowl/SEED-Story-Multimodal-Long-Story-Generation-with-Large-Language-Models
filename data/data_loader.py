from datasets import load_dataset

def load_data(dataset_name, config):
    dataset = load_dataset(dataset_name)
    train_ds = dataset["train"].shuffle(seed=42).batch(config.batch_size)
    eval_ds = dataset["validation"].batch(config.batch_size)
    return train_ds, eval_ds