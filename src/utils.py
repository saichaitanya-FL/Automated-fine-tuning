"""Package installation utilities with automatic GPU detection and CUDA support."""

import os
import subprocess
import sys
import logging
import platform
import re

logger = logging.getLogger(__name__)


def detect_cuda_version():
    """Detect CUDA version using nvidia-smi.
    
    Returns:
        str: CUDA version (e.g., "12.4") or None if not detected.
    """
    try:
        use_shell = platform.system() == "Windows"
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=15,
            shell=use_shell
        )
        
        if result.returncode == 0:
            # Try multiple regex patterns to find CUDA version
            patterns = [
                r'CUDA Version:\s*(\d+\.\d+)',
                r'CUDA\s+Version\s+(\d+\.\d+)',
                r'cuda\s+version[:\s]+(\d+\.\d+)',
            ]
            
            for pattern in patterns:
                cuda_match = re.search(pattern, result.stdout, re.IGNORECASE)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
                    logger.info(f"CUDA version detected: {cuda_version}")
                    return cuda_version
            
            logger.warning("Could not extract CUDA version from nvidia-smi output")
            return None
    except FileNotFoundError:
        logger.warning("nvidia-smi not found. GPU may not be available.")
    except Exception as e:
        logger.warning(f"Error detecting CUDA version: {str(e)}")
    
    return None


def get_pytorch_cuda_version(detected_cuda):
    """Get compatible PyTorch CUDA version (nearest smaller available version).
    
    PyTorch 2.7.1 supports: cu118, cu126, cu128
    
    Args:
        detected_cuda: Detected CUDA version string (e.g., "12.4")
        
    Returns:
        str: PyTorch CUDA version (e.g., "12.6", "11.8") or None for CPU.
    """
    if not detected_cuda:
        return None
    
    try:
        cuda_major, cuda_minor = map(float, detected_cuda.split('.'))
        
        # Available PyTorch CUDA versions (in order from smallest to largest)
        available_versions = [
            (11.8, "11.8"),
            (12.6, "12.6"),
            (12.8, "12.8"),
        ]
        
        detected_version = cuda_major + cuda_minor / 10.0
        
        # Find nearest smaller version (not equal, but smaller)
        selected_version = None
        for version_num, version_str in reversed(available_versions):  # Start from largest
            if detected_version > version_num:
                selected_version = version_str
                break
        
        # If detected version is smaller than or equal to all available, use smallest
        if selected_version is None:
            selected_version = "11.8"
        
        logger.info(f"Detected CUDA {detected_cuda}, using PyTorch CUDA {selected_version} (nearest smaller)")
        return selected_version
        
    except (ValueError, AttributeError) as e:
        logger.warning(f"Could not parse CUDA version '{detected_cuda}': {e}. Defaulting to CUDA 11.8")
        return "11.8"


def check_existing_pytorch():
    """Check if PyTorch is installed and whether it has CUDA support.
    
    Returns:
        dict: Installation status with version and CUDA availability.
    """
    try:
        import torch
        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None
        }


def uninstall_cpu_pytorch():
    """Uninstall CPU-only PyTorch packages.
    
    Returns:
        bool: True if uninstallation was successful or not needed, False otherwise.
    """
    logger.info("Checking for existing PyTorch installation...")
    existing_pytorch = check_existing_pytorch()
    
    if not existing_pytorch["installed"]:
        logger.info("PyTorch is not installed. Proceeding with fresh installation.")
        return True
    
    if existing_pytorch["cuda_available"]:
        cuda_status = f"CUDA {existing_pytorch['cuda_version']}"
        logger.info(f"PyTorch {existing_pytorch['version']} is already installed with {cuda_status} support.")
        logger.info("Skipping PyTorch installation.")
        return False  # Don't reinstall if CUDA is already available
    
    # PyTorch is installed but CUDA is not available (CPU version)
    logger.warning(f"PyTorch {existing_pytorch['version']} is installed but CUDA is not available (CPU version).")
    logger.info("Uninstalling CPU-only PyTorch to install CUDA version...")
    
    try:
        # Uninstall torch, torchvision, and related packages
        packages_to_uninstall = ["torch", "torchvision", "torchaudio", "xformers"]
        for package in packages_to_uninstall:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", package, "-y"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60
                )
            except Exception as e:
                logger.warning(f"Could not uninstall {package}: {e}")
        
        logger.info("CPU-only PyTorch uninstalled successfully")
        return True
    except Exception as e:
        logger.error(f"Error uninstalling PyTorch: {e}")
        raise


def install_pytorch_with_cuda(cuda_version):
    """Install PyTorch 2.7.1 with specified CUDA version, plus torchvision and xformers.
    
    - For CUDA builds, uses the official PyTorch CUDA index for torch/torchvision.
    - Installs xformers from PyPI with a version known to work:
      torch==2.7.1+cu118, torchvision==0.22.1+cu118, xformers==0.0.31.post1
    - For CPU builds, installs all from PyPI.
    
    Args:
        cuda_version: CUDA version string (e.g., "12.6", "11.8") or None for CPU.
    """
    try:
        if cuda_version is None:
            # CPU-only installation from PyPI
            logger.info("Installing PyTorch CPU version (from PyPI)...")
            cpu_packages = [
                "torch==2.7.1",
                "torchvision==0.22.1",
                "xformers==0.0.31.post1",
            ]
            cmd_cpu = [sys.executable, "-m", "pip", "install"] + cpu_packages
            result = subprocess.run(
                cmd_cpu,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            if result.returncode != 0:
                logger.error(f"PyTorch CPU installation failed:\n{result.stdout}")
                raise subprocess.CalledProcessError(result.returncode, cmd_cpu, result.stdout)
            logger.info("PyTorch, torchvision, and xformers (CPU) installed successfully")
            return
        
        # CUDA build installation
        cuda_version_map = {
            "12.8": "https://download.pytorch.org/whl/cu128",
            "12.6": "https://download.pytorch.org/whl/cu126",
            "11.8": "https://download.pytorch.org/whl/cu118",
        }
        index_url = cuda_version_map.get(cuda_version, cuda_version_map["11.8"])
        
        logger.info(f"Installing PyTorch 2.7.1 with CUDA {cuda_version} support")
        logger.info(f"Using index URL for torch/torchvision: {index_url}")
        
        # 1) Install torch and torchvision from the CUDA index
        torch_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "--index-url",
            index_url,
        ]
        result_torch = subprocess.run(
            torch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        if result_torch.returncode != 0:
            logger.error(f"torch/torchvision installation failed:\n{result_torch.stdout}")
            raise subprocess.CalledProcessError(result_torch.returncode, torch_cmd, result_torch.stdout)
        
        logger.info("torch==2.7.1 and torchvision==0.22.1 installed successfully")
        
        # 2) Install xformers from PyPI with a compatible version
        logger.info("Installing xformers==0.0.31.post1 from PyPI...")
        xformers_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "xformers==0.0.31.post1",
        ]
        result_xf = subprocess.run(
            xformers_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        if result_xf.returncode != 0:
            logger.error(f"xformers installation failed:\n{result_xf.stdout}")
            raise subprocess.CalledProcessError(result_xf.returncode, xformers_cmd, result_xf.stdout)
        
        logger.info("xformers==0.0.31.post1 installed successfully")
        logger.info("PyTorch, torchvision, and xformers (CUDA build) installed successfully")
    
    except subprocess.TimeoutExpired:
        logger.error("PyTorch installation timed out")
        raise
    except Exception as e:
        logger.error(f"Error installing PyTorch: {e}")
        raise


def install_packages():
    """Install required packages with OS-specific handling and GPU detection.
    
    Architecture:
    1. Detect OS (Windows, Linux, Mac)
    2. Windows: Install unsloth[windows] first, then detect CUDA and install PyTorch
    3. Linux/Mac: Install standard unsloth, then detect CUDA and install PyTorch
    4. Install common packages (evaluate, synthetic-data-kit, openai) on all OS
    """
    logger.info("=" * 60)
    logger.info("Package Installation with GPU Detection")
    logger.info("=" * 60)
    
    # Step 1: Detect OS
    os_name = platform.system()
    is_windows = os_name == "Windows"
    is_linux = os_name == "Linux"
    is_mac = os_name == "Darwin"
    
    logger.info(f"Operating System detected: {os_name}")
    
    # Step 2: Upgrade pip
    logger.info("Upgrading pip...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"pip upgrade failed: {e}")
    
    # Step 3: Install Unsloth (OS-specific)
    if is_windows:
        logger.info("Windows detected. Installing unsloth[windows]...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "unsloth[windows]"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600
            )
            if result.returncode != 0:
                logger.warning("unsloth[windows] installation failed, trying standard unsloth...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "unsloth"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
            else:
                logger.info("unsloth[windows] installed successfully")
            
            # Install triton-windows for Windows with specific version
            logger.info("Installing triton-windows==3.3.1.post21 for Windows...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "triton-windows==3.3.1.post21"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=600
                )
                logger.info("triton-windows installed successfully")
            except Exception as e:
                logger.warning(f"triton-windows installation failed (may not be critical): {e}")
        except Exception as e:
            logger.error(f"Failed to install unsloth[windows]: {e}")
            raise
    else:
        logger.info("Installing standard unsloth package...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "unsloth"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            logger.info("unsloth installed successfully")
        except Exception as e:
            logger.error(f"Failed to install unsloth: {e}")
            raise
    
    # Step 4: Detect CUDA version using nvidia-smi
    logger.info("Detecting CUDA version...")
    detected_cuda = detect_cuda_version()
    
    # Step 5: Determine PyTorch CUDA version (nearest smaller available)
    pytorch_cuda = get_pytorch_cuda_version(detected_cuda)
    
    # Step 6: Check existing PyTorch installation and uninstall CPU version if needed
    if pytorch_cuda is not None:  # Only check if we're installing CUDA version
        should_install = uninstall_cpu_pytorch()
        if not should_install:
            logger.info("PyTorch with CUDA is already installed. Skipping installation.")
        else:
            # Step 7: Install PyTorch 2.7.1 with appropriate CUDA version (includes torchvision and xformers)
            install_pytorch_with_cuda(pytorch_cuda)
    else:
        # No CUDA detected, install CPU version
        logger.info("No CUDA detected. Installing PyTorch CPU version...")
        install_pytorch_with_cuda(None)
    
    # Step 8: Install other required packages (all OS)
    logger.info("Installing other required packages...")
    final_packages = [
        "synthetic-data-kit==0.0.5",
        "openai==1.75.0",
        "sentence-transformers"
    ]
    
    # Install packages one by one for better error handling
    for package in final_packages:
        try:
            logger.info(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600
            )
            if result.returncode == 0:
                logger.info(f"{package} installed successfully")
            else:
                logger.warning(f"{package} installation had issues (return code: {result.returncode})")
                logger.warning(f"Output: {result.stdout[:500]}")  # Show first 500 chars
        except subprocess.TimeoutExpired:
            logger.error(f"Installation of {package} timed out")
            raise
        except Exception as e:
            logger.warning(f"Failed to install {package}: {e}. Continuing with other packages...")
    
    logger.info("Final packages installation completed")
    
    # Step 10: Verify installation
    logger.info("Verifying installation...")
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available (CPU mode)")
    except ImportError:
        logger.warning("Could not verify PyTorch installation")
    
    logger.info("=" * 60)
    logger.info("Installation completed successfully")
    logger.info("=" * 60)
