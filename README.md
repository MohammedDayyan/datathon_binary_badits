# Image Dehazing Pipeline

A complete Image Dehazing pipeline for Indian winter conditions, focusing on fog and smog removal for improved visibility in transportation, surveillance, and smart-city applications.

## Problem Overview
In the Indian context, image dehazing is especially important during North Indian winters, when dense fog and smog severely reduce visibility due to stubble burning and temperature inversion.

## Core Objectives
1. **Model Selection & Inference Pipeline**: Implement pre-trained Image Dehazing architectures
2. **Web Dashboard Deployment**: Interactive web application for model interaction
3. **Bonus**: Haze generation and dehazing pipeline

## Evaluation Parameters
- **Datasets**: I-Haze, N-Haze, Dense-Haze (NTIRE challenge datasets)
- **Metrics**: PSNR and SSIM
- **Quality Check**: Structural fidelity verification against hallucination

## Project Structure
```
├── models/           # Model implementations and weights
├── data/            # Dataset storage and processing
├── src/             # Source code for inference pipeline
├── web/             # Web dashboard application
├── evaluation/      # Evaluation scripts and metrics
├── notebooks/       # Research and experimentation
└── requirements.txt # Dependencies
```

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Download pre-trained models to `models/` directory
3. Run web dashboard: `python web/app.py`
4. Upload hazy images through the web interface for dehazing