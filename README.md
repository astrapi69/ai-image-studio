# 🎨 AI Image Studio

A complete workflow for managing AI image generation projects with Stable Diffusion.

## 🚀 Quick Start

1. **Clone and setup**
   ```bash
   git clone <your-repo-url>
   cd ai-image-studio
   poetry install
   poetry shell
   ```

2. **Create your first project**
   ```bash
   ai-prompts create my_project --prompts "A red dragon" "A blue castle" "A green forest"
   ```

3. **Generate images**
   ```bash
   ai-generate prompts/projects/my_project.json my_project
   ```

## 📁 Project Structure

- `ai_image_studio/` - Main package with scripts and utilities
- `prompts/` - Organized prompt collections
- `output/` - Generated images with metadata
- `config/` - Configuration files
- `docs/` - Documentation and guides

## 🛠️ Available Commands

Poetry provides convenient CLI commands:

### Prompt Management

```bash
# Create new project
ai-prompts create PROJECT_NAME --prompts "prompt1" "prompt2"

# Import from CSV
ai-prompts from-csv data.csv PROJECT_NAME
```

### Image Generation

```bash
# Basic generation
ai-generate prompts/projects/PROJECT.json PROJECT_NAME

# With custom model
ai-generate prompts/projects/PROJECT.json PROJECT_NAME --model "custom/model"
```

### Development Commands

```bash
# Run tests
poetry run pytest

# Format code
poetry run black ai_image_studio/

# Type checking
poetry run mypy ai_image_studio/
```

## 📊 Output Structure

Each generation creates:

- **Raw images** in `output/PROJECT_NAME/`
- **Metadata files** for each image
- **Generation report** with statistics
- **CSV summary** for analysis

## 🔧 Configuration

Edit `config/settings.json` to customize:

- Default generation parameters
- Output directories
- Quality settings
- Model configurations

## 📚 Documentation

- [Setup Guide](docs/SETUP.md)
- [Usage Examples](docs/USAGE.md)
- [API Reference](docs/API.md)