# Diffusion Model Pipeline

This project implements a pipeline for training unconditional diffusion models, such as DDPM (Denoising Diffusion Probabilistic Model) or DDIM (Denoising Diffusion Implicit Model) for generating realistic images. The pipeline includes data acquisition, analysis, and preparation steps for two datasets: CelebA and Flowers102.

## Team info:

- Team name: ImGenDiff
- Team members: Baczó Domonkos Z9EGIM, Varsányi Máté IPBZX7

## Prerequisites

- Docker
- Docker Compose
- Up to date NVIDIA drivers
- NVIDIA Container Toolkit

## Project Structure

```
project_directory/
│
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── diffusion_model_pipeline.ipynb
└── README.md
```

## Getting Started

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. Build and run the Docker container:
   ```
   docker-compose up --build
   ```

3. Once the container is running, you'll see a URL in the console output that looks like:
   ```
   http://127.0.0.1:8888/lab?token=<some_long_token>
   ```
   Copy and paste this URL into your web browser.

4. In the Jupyter Lab interface, open `diffusion_model_pipeline.ipynb`.

5. Run the cells in the notebook to perform data acquisition, analysis, and preparation, and to train the baseline model.

## Notebook Contents

The `diffusion_model_pipeline.ipynb` notebook contains the following sections:

1. Data Acquisition
   - Downloads and extracts the CelebA dataset
   - Downloads the Flowers102 dataset

2. Data Analysis
   - Analyzes image sizes, aspect ratios, and distributions for both datasets

3. Data Cleansing and Preparation
   - Prepares and cleans the datasets for training

4. Baseline model
   - Creates a VAE baseline model
   - Trains the model on the Flowers dataset
   - Generates some images based on the Flowers dataset 

## Stopping the Container

To stop the Docker container, use `Ctrl+C` in the terminal where you ran `docker-compose up`, or run `docker-compose down` in another terminal in the same directory.

