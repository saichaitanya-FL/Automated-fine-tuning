"""Utility functions for package installation"""

import subprocess
import sys
import logging

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
