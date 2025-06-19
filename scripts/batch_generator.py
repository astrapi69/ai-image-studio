#!/usr/bin/env python3
"""
AI Image Batch Generator
Processes prompt files and generates images with full logging and metadata.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm
import pandas as pd

from ..utils.file_helpers import load_config, save_metadata, ensure_directory
from ..utils.image_helpers import add_metadata_to_image, validate_image


class ImageGenerator:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = load_config(config_path)
        self.pipeline = None
        self.generation_log = []

    def initialize_pipeline(self, model_name: Optional[str] = None):
        """Initialize the Stable Diffusion pipeline."""
        model = model_name or self.config["generation"]["default_model"]
        device = self.config["generation"]["device"]

        print(f"Loading model: {model}")
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(model)
            self.pipeline.to(device)
            print(f"✅ Pipeline loaded on {device}")
        except Exception as e:
            fallback = self.config["generation"]["fallback_device"]
            print(f"⚠️  Failed to load on {device}, trying {fallback}")
            self.pipeline = StableDiffusionPipeline.from_pretrained(model)
            self.pipeline.to(fallback)

    def generate_from_prompts(self, prompt_file: str, project_name: str):
        """Generate images from a prompt file."""
        # Load prompts
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)

        prompts = prompt_data.get("prompts", [])
        settings = prompt_data.get("settings", {})

        # Create project directory
        output_dir = Path(self.config["paths"]["output_dir"]) / project_name
        ensure_directory(output_dir)

        # Process each prompt
        results = []
        for i, prompt_item in enumerate(tqdm(prompts, desc="Generating images")):
            try:
                result = self._generate_single_image(
                    prompt_item, output_dir, i, settings
                )
                results.append(result)
                self.generation_log.append(result)
            except Exception as e:
                error_result = {
                    "index": i,
                    "prompt": prompt_item.get("text", ""),
                    "filename": prompt_item.get("filename", f"image_{i:04d}.png"),
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)
                print(f"❌ Error generating image {i}: {e}")

        # Save generation report
        self._save_generation_report(results, output_dir, project_name)
        return results

    def _generate_single_image(self, prompt_item: Dict, output_dir: Path,
                               index: int, global_settings: Dict) -> Dict:
        """Generate a single image and save with metadata."""
        # Extract prompt details
        prompt_text = prompt_item.get("text", "")
        filename = prompt_item.get("filename", f"image_{index:04d}.png")
        local_settings = prompt_item.get("settings", {})

        # Merge settings (local overrides global)
        merged_settings = {**global_settings, **local_settings}

        # Generation parameters
        params = {
            "prompt": prompt_text,
            "num_inference_steps": merged_settings.get("steps", self.config["generation"]["default_steps"]),
            "guidance_scale": merged_settings.get("guidance", self.config["generation"]["default_guidance"]),
            "height": merged_settings.get("height", self.config["generation"]["default_size"][1]),
            "width": merged_settings.get("width", self.config["generation"]["default_size"][0]),
        }

        # Add seed if specified
        if "seed" in merged_settings:
            generator = torch.Generator().manual_seed(merged_settings["seed"])
            params["generator"] = generator

        # Generate image
        start_time = datetime.now()
        image = self.pipeline(**params).images[0]
        generation_time = (datetime.now() - start_time).total_seconds()

        # Save image
        image_path = output_dir / filename
        image.save(image_path, format="PNG")

        # Create metadata
        metadata = {
            "prompt": prompt_text,
            "filename": filename,
            "generation_params": params,
            "generation_time_seconds": generation_time,
            "timestamp": start_time.isoformat(),
            "model": self.config["generation"]["default_model"],
            "image_size": [params["width"], params["height"]],
            "file_size_bytes": image_path.stat().st_size
        }

        # Save metadata
        metadata_path = output_dir / f"{Path(filename).stem}_metadata.json"
        save_metadata(metadata, metadata_path)

        return {
            "index": index,
            "prompt": prompt_text,
            "filename": filename,
            "status": "success",
            "generation_time": generation_time,
            "timestamp": start_time.isoformat(),
            "metadata_file": str(metadata_path)
        }

    def _save_generation_report(self, results: List[Dict], output_dir: Path, project_name: str):
        """Save a comprehensive generation report."""
        report_data = {
            "project_name": project_name,
            "generation_date": datetime.now().isoformat(),
            "total_images": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results
        }

        # Save JSON report
        report_path = output_dir / f"{project_name}_generation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Save CSV summary
        df = pd.DataFrame(results)
        csv_path = output_dir / f"{project_name}_generation_summary.csv"
        df.to_csv(csv_path, index=False)

        print(f"📊 Generation report saved: {report_path}")
        print(f"📈 Summary CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Image Batch Generator")
    parser.add_argument("prompt_file", help="Path to prompt JSON file")
    parser.add_argument("project_name", help="Name for this generation project")
    parser.add_argument("--model", help="Override default model")
    parser.add_argument("--config", default="config/settings.json", help="Config file path")

    args = parser.parse_args()

    # Initialize generator
    generator = ImageGenerator(args.config)
    generator.initialize_pipeline(args.model)

    # Generate images
    results = generator.generate_from_prompts(args.prompt_file, args.project_name)

    # Print summary
    successful = len([r for r in results if r["status"] == "success"])
    total = len(results)
    print(f"\n🎉 Generation complete: {successful}/{total} images successful")


if __name__ == "__main__":
    main()