from detectron2.engine import DefaultTrainer
import argparse
import os
from utils.det_utils import create_config, register_datasets
from detectron2.utils.logger import setup_logger
import concurrent.futures
import sys
import subprocess
import signal
import yaml

logger = setup_logger()

# TODO: Will be updated to use lauch for multi-distributed training
def train_model(cfg):
    """
    Train the model using the given configuration
    """
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg.OUTPUT_DIR

def main(args):
    # start tensorboard
    pid = os.fork()
    if pid == 0:
        os.setsid()
        os.system(f"tensorboard --logdir {args.output_dir} --port 6006")
        sys.exit(0)
    else:
        logger.info(f"Tensorboard started with PID {pid}")
    
    config_file = args.config_file
    cfg, original_config = create_config(config_file)
    # register the datasets train, test
    for dataset_name, dataset_path in zip(original_config['DATASETS']["TRAIN"], original_config['DATASETS']["TRAIN_DIR"]):
        register_datasets(dataset_dir=dataset_path, dataset_name=dataset_name)
    for dataset_name, dataset_path in zip(original_config['DATASETS']["TEST"], original_config['DATASETS']["TEST_DIR"]):
        register_datasets(dataset_dir=dataset_path, dataset_name=dataset_name)
    logger.info("Starting training run")
    train_model(cfg)
    # kill tensorboard
    os.kill(pid, signal.SIGTERM)

    # update the config file with the output directory
    # load the yaml file in the output directory
    config = yaml.safe_load(open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "r"))
    config["OUTPUT_DIR"] = cfg.OUTPUT_DIR # update the output directory
    config["MODEL"]["WEIGHTS"] = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # update the weights path
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    logger.info("Training run completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, help="Path to the config file")
    parser.add_argument("--output-dir", type=str, help="Path to the output directory", default="/home/training/")
    args = parser.parse_args()
    main(args=args)