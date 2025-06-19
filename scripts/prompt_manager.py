#!/usr/bin/env python3
"""
Prompt Management System
Create, validate, and manage prompt collections for AI image generation.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)

    def create_project(self, project_name: str, prompts: List[str],
                       settings: Optional[Dict] = None) -> str:
        """Create a new prompt project file."""
        project_data = {
            "project_name": project_name,
            "created_date": datetime.now().isoformat(),
            "description": f"AI image generation project: {project_name}",
            "settings": settings or {
                "steps": 30,
                "guidance": 7.5,
                "width": 512,
                "height": 512
            },
            "prompts": []
        }

        # Convert prompts to structured format
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                prompt_item = {
                    "text": prompt,
                    "filename": f"{project_name}_{i + 1:04d}.png",
                    "settings": {}
                }
            else:
                prompt_item = prompt

            project_data["prompts"].append(prompt_item)

        # Save to file
        project_file = self.prompts_dir / "projects" / f"{project_name}.json"
        project_file.parent.mkdir(parents=True, exist_ok=True)

        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Created project: {project_file}")
        return str(project_file)

    def add_prompts_from_csv(self, csv_path: str, project_name: str):
        """Create project from CSV file."""
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Validate CSV format
        required_columns = ["prompt"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        prompts = []
        for _, row in df.iterrows():
            prompt_item = {
                "text": row["prompt"],
                "filename": row.get("filename", ""),
                "settings": {}
            }

            # Add optional settings from CSV
            for col in ["steps", "guidance", "seed", "width", "height"]:
                if col in df.columns and pd.notna(row[col]):
                    prompt_item["settings"][col] = row[col]

            prompts.append(prompt_item)

        return self.create_project(project_name, prompts)


def main():
    parser = argparse.ArgumentParser(description="Prompt Management System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create project command
    create_parser = subparsers.add_parser("create", help="Create new project")
    create_parser.add_argument("project_name", help="Name of the project")
    create_parser.add_argument("--prompts", nargs="+", required=True, help="List of prompts")
    create_parser.add_argument("--steps", type=int, default=30, help="Generation steps")
    create_parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")

    # CSV import command
    csv_parser = subparsers.add_parser("from-csv", help="Create project from CSV")
    csv_parser.add_argument("csv_file", help="Path to CSV file")
    csv_parser.add_argument("project_name", help="Name of the project")

    args = parser.parse_args()

    manager = PromptManager()

    if args.command == "create":
        settings = {
            "steps": args.steps,
            "guidance": args.guidance
        }
        manager.create_project(args.project_name, args.prompts, settings)

    elif args.command == "from-csv":
        manager.add_prompts_from_csv(args.csv_file, args.project_name)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()