# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChirpMinds is an AI toolkit for analyzing bird behavior through video processing and annotation. The project uses a hybrid Nix + Python development setup with Docker containers for Label Studio annotation workflows.

## Development Environment

This project uses **Nix flakes** for reproducible development environments with CUDA support:

```bash
# Enter development shell
nix develop

# Development shell includes:
# - Python 3.11 with CUDA-enabled packages
# - uv package manager (recommended over pip)
# - FFmpeg for video processing
# - Arion for Docker container orchestration
# - Just for task automation
# - Pueue for job queue management
```

## Package Management

The project uses **uv** for Python package management with PyTorch index configurations:

```bash
# Install dependencies
uv sync

# Install with CUDA support (default)
uv sync --extra cu124

# Install with CPU-only PyTorch
uv sync --extra cpu

# Install development dependencies
uv sync --extra dev
```

Dependencies are defined in `pyproject.toml` and lockfile is `uv.lock`.

## Common Development Tasks

### Running the Development Environment
```bash
# Start Label Studio and ML backend
just dev

# Or run components separately:
just studio    # Starts Label Studio container
just studioml  # Starts ML backend container
```

### Video Processing Workflow
The project uses Jupyter notebooks in two formats:
- `notebooks/ipynb/` - Standard Jupyter notebooks
- `notebooks/pypct/` - Python percent format (managed by Jupytext)

Key notebooks:
- `000_download_gdrive.py` - Downloads video data from Google Drive
- `001_process_video.py` - Extracts frames and processes video files
- `002_setup_annotation_project.py` - Sets up Label Studio annotation projects

### Working with Notebooks
```bash
# Convert between formats (handled automatically by Jupytext)
# Edit either .py or .ipynb files - they stay in sync

# Export notebook code to modules
# Notebooks use nbdev export functionality
```

## Architecture

### Key Components

1. **Video Processing Pipeline** (`notebooks/pypct/001_process_video.py`)
   - Uses FFmpeg for frame extraction and video manipulation
   - Handles interlaced video with bwdif filter
   - Parallel processing for batch operations
   - Outputs to `scratch/` directory structure

2. **Annotation System** (`nix/arion/`)
   - Label Studio container setup via Arion
   - SAM (Segment Anything Model) ML backend integration
   - Container orchestration with GPU support

3. **Data Flow**
   - Raw video files → `scratch/raw_data/`
   - Processed clips → `scratch/clips/`
   - Extracted frames → `scratch/frames/`
   - Container data → `scratch/container_data/`

### Dependencies

Core ML/Video libraries:
- **ultralytics** - YOLO models
- **fiftyone** - Dataset management
- **label-studio** - Annotation platform
- **ffmpeg-python** - Video processing
- **supervision** - Computer vision utilities
- **moviepy** - Video editing

Development tools:
- **ruff** - Linting and formatting
- **pytest** - Testing
- **nbdev** - Notebook development
- **jupytext** - Notebook synchronization

## Container Services

The project uses Label Studio with SAM backend:

```bash
# Label Studio accessible at http://localhost:8080
# ML Backend at http://localhost:9090
# Access token: b0429a22dacfad6c4fe937b8d9ddcf7b8dce7de7
```

## File Organization

- `notebooks/` - Jupyter notebooks for data processing
- `nix/` - Nix configuration and container definitions
- `scratch/` - Working directory for processed data
- `old/` - Legacy code and documentation
- `src/` - Source code modules (if any)

## Code Quality

Run linting and formatting:
```bash
ruff check .
ruff format .
```

The project excludes `src` and `notebooks` directories from pyright type checking.