#!/usr/bin/env python3
"""
Hugging Face Model Downloader for Ollama Deployment
Downloads GGUF models from Hugging Face repositories
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(args):
    """Download model from Hugging Face"""
    try:
        logger.info(f"Downloading model from Hugging Face: {args.repo_id}")
        logger.info(f"Model file pattern: {args.filename}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # List all files in the repository
        logger.info("Listing repository files...")
        all_files = list_repo_files(
            repo_id=args.repo_id,
            token=args.hf_token
        )
        
        # Filter for GGUF files
        gguf_files = [f for f in all_files if f.endswith('.gguf')]
        
        if not gguf_files:
            logger.error("No GGUF files found in repository")
            sys.exit(1)
        
        logger.info(f"Found {len(gguf_files)} GGUF file(s):")
        for f in gguf_files:
            logger.info(f"  - {f}")
        
        # Select file to download
        if args.filename:
            # User specified a file
            target_file = args.filename
            if target_file not in all_files:
                logger.error(f"File {target_file} not found in repository")
                logger.info("Available GGUF files:")
                for f in gguf_files:
                    logger.info(f"  - {f}")
                sys.exit(1)
        else:
            # Use first GGUF file or specific quantization
            if args.quantization:
                matching_files = [f for f in gguf_files if args.quantization.upper() in f.upper() and not f.startswith('mmproj')]
                if matching_files:
                    target_file = matching_files[0]
                    logger.info(f"Selected file with quantization {args.quantization}: {target_file}")
                else:
                    logger.warning(f"No file found with quantization {args.quantization}, using first GGUF file")
                    target_file = [f for f in gguf_files if not f.startswith('mmproj')][0]
            else:
                target_file = [f for f in gguf_files if not f.startswith('mmproj')][0]
                logger.info(f"Using first GGUF file: {target_file}")
        
        # Download the main model file
        logger.info(f"Downloading {target_file}...")
        logger.info(f"This may take a while depending on file size and network speed...")
        
        downloaded_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=target_file,
            token=args.hf_token,
            cache_dir=output_dir / ".cache",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Model downloaded successfully to: {downloaded_path}")
        
        # Check for and download mmproj files (for vision models)
        mmproj_files = [f for f in gguf_files if f.startswith('mmproj')]
        if mmproj_files:
            logger.info(f"Found {len(mmproj_files)} mmproj file(s) for vision model")
            # Download the first mmproj file (typically F16 or BF16)
            mmproj_file = mmproj_files[0]
            logger.info(f"Downloading mmproj file: {mmproj_file}...")
            
            mmproj_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=mmproj_file,
                token=args.hf_token,
                cache_dir=output_dir / ".cache",
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            logger.info(f"Mmproj downloaded successfully to: {mmproj_path}")
        
        # Verify file exists and get size
        file_size = os.path.getsize(downloaded_path)
        logger.info(f"File size: {file_size / (1024**3):.2f} GB")
        
        # Create a simple metadata file
        metadata_file = output_dir / "model_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"Repository: {args.repo_id}\n")
            f.write(f"File: {target_file}\n")
            f.write(f"Size: {file_size} bytes\n")
            f.write(f"Downloaded: {downloaded_path}\n")
        
        logger.info("Download complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Download GGUF models from Hugging Face for Ollama'
    )
    
    # Hugging Face specific arguments
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='Hugging Face repository ID (e.g., unsloth/Qwen3-VL-8B-Instruct-GGUF)'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=os.environ.get('HF_TOKEN'),
        help='Hugging Face access token'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='Specific file to download (if not specified, first GGUF file will be used)'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        default='Q4_K_M',
        help='Quantization level preference (e.g., Q4_K_M, Q5_K_M, Q8_0)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for downloaded model'
    )
    
    # Legacy arguments for compatibility with existing deployment
    parser.add_argument('--model-name', type=str, help='Model name (for compatibility)')
    parser.add_argument('--model-version', type=str, help='Model version (for compatibility)')
    parser.add_argument('--backend-base-url', type=str, help='Backend URL (ignored)')
    parser.add_argument('--backend-bearer-token', type=str, help='Bearer token (ignored)')
    parser.add_argument('--backend-catalog-id', type=str, help='Catalog ID (ignored)')
    parser.add_argument('--no-backup', action='store_true', help='No backup flag (ignored)')
    
    args = parser.parse_args()
    
    # Validate token
    if not args.hf_token:
        logger.error("Hugging Face token is required. Set HF_TOKEN environment variable or use --hf-token")
        sys.exit(1)
    
    # Download model
    return download_model(args)


if __name__ == '__main__':
    sys.exit(main())
