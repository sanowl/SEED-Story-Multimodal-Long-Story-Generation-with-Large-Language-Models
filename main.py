import jax
from config import SEEDStoryConfig
from models.seed_story import SEEDStory
from data.data_loader import load_data
from train import train_model
from evaluate import evaluate_model
from logging_utils import setup_logger, log_exception
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train SEEDStory model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("SEEDStory")

    try:
        logger.info("Initializing SEEDStory training")
        config = SEEDStoryConfig.from_json(args.config)
        model = SEEDStory(config)
        
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            model = model.load_checkpoint(args.checkpoint)

        train_ds, eval_ds = load_data(config)
        
        if args.eval:
            logger.info("Running evaluation")
            eval_metrics = evaluate_model(model, eval_ds)
            logger.info(f"Evaluation metrics: {eval_metrics}")
        else:
            logger.info("Starting training")
            trained_state = train_model(config, model, train_ds, eval_ds, evaluate_model)
            
            logger.info("Saving trained model")
            trained_state.save_checkpoint(config.model_save_path)
            
        logger.info("Process completed successfully")
    
    except Exception:
        log_exception(logger, sys.exc_info())
        sys.exit(1)

if __name__ == "__main__":
    main()