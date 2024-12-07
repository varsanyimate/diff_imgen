import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
import time
from torch import nn, einsum
from inspect import isfunction
from functools import partial
from src.model_unet import *
device = "cuda" if torch.cuda.is_available() else "cpu"

class Evaluator:
    def __init__(self, model, device='cuda'):
        print("Initializing evaluation metrics...")
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=model.to(self.device)
        print(self.device)
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.inception_score = InceptionScore(normalize=True).to(device)
        self.kid = KernelInceptionDistance(normalize=True, subset_size=50).to(device)
        print("✓ Metrics initialized successfully")

    def num_to_groups(self,num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr

    @torch.no_grad()
    def generate_images(self, num_images, image_size=64, batch_size=32):
        # Ensure model is in evaluation mode and on the correct device
        self.model.eval()
        #model = model.to(self.device)
        batches = self.num_to_groups(num_images, batch_size)
        
        # Sample images for each batch
        all_images_list = []
        for n in batches:
            # Generate images for this batch
            # Ensure the shape tensor is on the same device as the model
            batch_images = sample(self.model, image_size, diffusion_params=diffusion_params, batch_size=32, channels=3)
            
            # Take the last image in the sampling process (fully denoised)
            batch_images = batch_images[-1].to(self.device)
            all_images_list.append(batch_images)
        
        # Concatenate images
        all_images = torch.cat(all_images_list, dim=0)
        
        # Move to CPU for saving
        all_images = all_images.cpu()
        
        # Normalize to [0, 1] range
        all_images = (all_images + 1) * 0.5
        
        # Ensure images are on CPU and clipped to [0, 1]
        all_images = torch.clamp(all_images, 0, 1)
        
        # Save images
        #save_image(all_images, 'generated_images.png', nrow=8)
        
        return all_images


        
    @torch.no_grad()
    def evaluate_samples(self, real_dataloader, num_samples=100, batch_size=32):
        """
        Enhanced evaluation with KID score and progress logging
        """
        start_time = time.time()
        print(f"\n📊 Starting evaluation with {num_samples} samples...")
    
        print("\n1️⃣ Collecting real images...")
        real_images = []
        for batch, _ in real_dataloader:
            real_images.append(batch)
            if len(torch.cat(real_images)) >= num_samples:
                break
        real_images = torch.cat(real_images)[:num_samples].to(self.device)
        print(f"✓ Collected {len(real_images)} real images")

        print("\n2️⃣ Generating samples...")
        generated_images = self.generate_images(num_images=num_samples, image_size=64)
        generated_images = generated_images.to(self.device)
        print(f"✓ Generated {len(generated_images)} images")
        
        # Calculate metrics
        metrics = {}
        print("\n3️⃣ Computing metrics...")
        
        print("Computing FID score...")
        self.fid.reset()
        self.fid.update(real_images, real=True)
        self.fid.update(generated_images, real=False)
        metrics['fid'] = self.fid.compute().item()
        print(f"✓ FID Score: {metrics['fid']:.2f}")
        
        print("\nComputing Inception Score...")
        self.inception_score.reset()
        self.inception_score.update(generated_images)
        is_mean, is_std = self.inception_score.compute()
        metrics['inception_score_mean'] = is_mean.item()
        metrics['inception_score_std'] = is_std.item()
        print(f"✓ Inception Score: {metrics['inception_score_mean']:.2f} ± {metrics['inception_score_std']:.2f}")
        
        print("\nComputing KID score...")
        self.kid.reset()
        self.kid.update(real_images, real=True)
        self.kid.update(generated_images, real=False)
        kid_mean, kid_std = self.kid.compute()
        metrics['kid_mean'] = kid_mean.item()
        metrics['kid_std'] = kid_std.item()
        print(f"✓ KID Score: {metrics['kid_mean']:.4f} ± {metrics['kid_std']:.4f}")
        
        print("\nComputing diversity score...")
        if len(generated_images) >= 2:
            diversity_score = self.calculate_diversity(
                generated_images, 
                num_pairs=min(100, num_samples * 2)
            )
            metrics['diversity_score'] = diversity_score
            print(f"✓ Diversity Score: {metrics['diversity_score']:.2f}")
        
        elapsed_time = time.time() - start_time
        print(f"\n✨ Evaluation completed in {elapsed_time:.2f} seconds")
            
        return metrics
    
    def calculate_diversity(self, images, num_pairs=100):
        """
        Calculate diversity score with random pairs
        """
        num_images = len(images)
        idx1 = torch.randint(0, num_images, (num_pairs,))
        idx2 = torch.randint(0, num_images, (num_pairs,))
        
        images_flat = images.view(len(images), -1)
        distances = torch.norm(
            images_flat[idx1] - images_flat[idx2], 
            dim=1, 
            p=2
        )
        
        return distances.mean().item()
