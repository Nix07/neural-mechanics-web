# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static website for **Special Topics in AI: The Neural Mechanics of Concepts** at Northeastern University (Spring 2026).

### Course Goals

The course teaches research methods to localize and characterize rich concept representations within large language models (LLMs). Students work in interdisciplinary teams to:

1. Choose an important concept category from a field outside CS (e.g., law, psychology, medicine, sociology, literature, mathematics)
2. Apply methods of mechanistic interpretability to probe, characterize, and localize neural representations of those concepts within LLM internals
3. Draft a NeurIPS-level research paper submission by the end of the semester

**Target Audience**: PhD students pursuing research in machine learning, or interdisciplinary PhD students with expertise in a concept to characterize (enrollment with permission of professor).

### Website Purpose

The site serves as the central hub for course materials. Weekly pages will include:
- Tutorial material on interpretability methods
- Assigned readings from mechanistic interpretability research
- Coding exercises demonstrating techniques
- Project homework challenges to develop research skills

## Build System

Simple Makefile-based system that copies files from `src/` to `public/`:

- **Build**: `make` or `make all` - copies modified files from src/ to public/
- **Deploy**: `make deploy` - uses rsync to sync all src/ to public/

## Repository Structure

```
src/
  index.html           # Main course homepage with schedule
  week0.html, week1.html, etc.  # Individual week pages (to be created)
  imgs/                # Images and assets
public/                # Build output (auto-generated, don't edit)
```

## Content Development

### Main Homepage (`src/index.html`)
- Course staff information
- Course description emphasizing interdisciplinary research approach
- Weekly lecture schedule (currently covers weeks 0-12 with topics like benchmarking, steering, representation visualization, causation/patching, circuits, probes, etc.)
- Links to individual week pages

### Weekly Pages (to be developed)
Each week focuses on specific interpretability methods and should include:
- Tutorial explanations of the week's method/concept
- Research paper readings
- Hands-on coding exercises (likely using Python/PyTorch)
- Project milestones or homework challenges
- Consistent styling with main site

## Development Workflow

1. Edit files in `src/` directory only (never edit `public/` directly)
2. Run `make` to build changes
3. When creating new week pages, update the schedule table in `src/index.html` to link to them

## Git Commits

- **Run `make` before pushing** - changes must be built to `public/` before pushing for them to be available on the live site
- **No emojis in commit messages** - the deployment server cannot handle non-ASCII characters
