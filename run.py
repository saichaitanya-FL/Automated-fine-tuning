#!/usr/bin/env python3
"""
Automated Fine-tuning System - Main Entry Point
Run: python run.py config.json
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finetuning_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def install_packages():
    """Install required packages with proper version management"""
    logger.info("Installing required packages...")
    
    packages = [
        "torch",
        "bitsandbytes",
        "accelerate",
        "peft",
        "trl<=0.23.1",
        "transformers",
        "datasets>=3.4.1,<4.0.0",
        "huggingface_hub>=0.34.0",
        "sentencepiece",
        "protobuf",
        "sentence-transformers",
        "pandas",
        "evaluate",
        "synthetic-data-kit==0.0.5",
        "openai==1.75.0",
        "nest-asyncio",
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth", "--no-deps"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth_zoo"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("All packages installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing packages: {e}")
        sys.exit(1)

# Install packages first
install_packages()

# Now import after installation
import argparse
import torch
from src.engine import RecommendationEngine


def main():
    parser = argparse.ArgumentParser(description='Automated Fine-tuning System')
    parser.add_argument('config_files', nargs='+', help='Path(s) to config JSON file(s)')
    parser.add_argument('--threshold-f1', type=float, default=0.2, help='F1 threshold (default: 0.2)')
    parser.add_argument('--threshold-latency', type=float, default=0.3, help='Latency threshold (default: 0.3)')
    parser.add_argument('--factors', nargs='+', default=['accuracy', 'latency'], help='Optimization factors (default: accuracy latency)')
    
    args = parser.parse_args()

    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("No GPU detected. Training will be slow on CPU.")
    
    logger.info(f"Config files: {', '.join(args.config_files)}")
    
    # Run experiments
    recommendation = RecommendationEngine(
        config_paths=args.config_files,
        threshold_f1=args.threshold_f1,
        threshold_latency=args.threshold_latency,
        factors=args.factors
    )
    
    result = recommendation.get_best_model()
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(result)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
