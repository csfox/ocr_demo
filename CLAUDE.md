# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an OCR (Optical Character Recognition) demo project built with Python 3.12. The project uses `uv` for dependency management and package operations.

## Development Commands

### Running the Application
```bash
uv run main.py
```

### Python Environment
- Python version: 3.12 (specified in `.python-version`)
- Package manager: `uv` (configured via `pyproject.toml`)

### Installing Dependencies
```bash
uv sync
```

### Adding New Dependencies
```bash
uv add <package-name>
```

## Project Structure

- `main.py` - Entry point with a simple main() function
- `pyproject.toml` - Project metadata and dependencies (uv configuration)
- `.python-version` - Specifies Python 3.12 requirement

## Architecture Notes

This is a minimal starter project. The main application logic is currently a basic "Hello World" style entry point in `main.py:1`. As OCR functionality is added, consider organizing code into:
- OCR processing modules
- Image preprocessing utilities
- Output formatting/export functions
