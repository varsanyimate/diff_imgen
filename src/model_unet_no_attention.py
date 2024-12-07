import math
from inspect import isfunction
from functools import partial

#%matplotlib inline
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader, TensorDataset


class Residual(nn.Module):
    """
    A generic residual wrapper that adds the input to the output of a function.
    
    This allows any module to be transformed into a residual connection,
    where the original input is added to the transformed output.
    
    Args:
        fn (callable): The transformation function to be wrapped
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # Store the transformation function
    
    def forward(self, x, *args, **kwargs):
        """
        Apply the transformation and add it to the original input.
        
        Args:
            x (torch.Tensor): Input tensor
            *args: Additional positional arguments for the transformation
            **kwargs: Additional keyword arguments for the transformation
        
        Returns:
            torch.Tensor: Input + transformed input
        """
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    """
    Create an upsampling layer using transposed convolution.
    
    Args:
        dim (int): Number of input/output channels
    
    Returns:
        nn.ConvTranspose2d: Upsampling layer that doubles spatial dimensions
    """
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)
    # 4: kernel size
    # 2: stride (doubles spatial dimensions)
    # 1: padding

def Downsample(dim):
    """
    Create a downsampling layer using strided convolution.
    
    Args:
        dim (int): Number of input/output channels
    
    Returns:
        nn.Conv2d: Downsampling layer that halves spatial dimensions
    """
    return nn.Conv2d(dim, dim, 4, 2, 1)
    # 4: kernel size
    # 2: stride (halves spatial dimensions)
    # 1: padding
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generate sinusoidal embeddings for time steps.
    
    This technique creates unique, periodic embeddings that help neural networks
    understand temporal information across different scales.
    
    Args:
        dim (int): Dimension of the embedding vector
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        """
        Convert time steps into sinusoidal embeddings.
        
        Args:
            time (torch.Tensor): Time steps to embed
        
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (batch_size, dim)
        """
        # Determine the device of the input tensor
        device = time.device
        
        # Calculate half the dimension (we'll use half for sin, half for cos)
        half_dim = self.dim // 2
        
        # Create a geometric sequence of frequencies
        # This helps capture different scales of temporal information
        frequency_scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -frequency_scale)
        
        # Broadcast time steps to create scaled frequency representations
        scaled_times = time[:, None] * frequencies[None, :]
        
        # Combine sine and cosine of the scaled times to create embeddings
        # This creates a unique embedding for each time step
        embeddings = torch.cat([
            scaled_times.sin(), 
            scaled_times.cos()
        ], dim=-1)
        
        return embeddings

class Block(nn.Module):
    """
    A basic convolutional block with normalization and activation.
    
    Args:
        dim (int): Input channel dimension
        dim_out (int): Output channel dimension
        groups (int, optional): Number of groups for GroupNorm. Defaults to 8.
    """
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)  # 3x3 convolution
        self.norm = nn.GroupNorm(groups, dim_out)  # Group normalization
        self.activation = nn.SiLU()  # Sigmoid Linear Unit activation
    
    def forward(self, x, scale_shift=None):
        """
        Forward pass with optional adaptive scaling and shifting.
        
        Args:
            x (torch.Tensor): Input tensor
            scale_shift (tuple, optional): Scaling and shifting parameters
        
        Returns:
            torch.Tensor: Processed tensor
        """
        x = self.proj(x)  # Convolutional projection
        x = self.norm(x)  # Normalization
        
        # Optional adaptive scaling and shifting
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.activation(x)  # Activation
        return x


class ResnetBlock(nn.Module):
    """
    Residual Network Block with time-conditional processing.
    
    Implements the core idea from 'Deep Residual Learning for Image Recognition'
    with additional time embedding support.
    
    Args:
        dim (int): Input channel dimension
        dim_out (int): Output channel dimension
        time_emb_dim (int, optional): Dimension of time embeddings
        groups (int, optional): Number of groups for normalization. Defaults to 8.
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # Optional time embedding processing
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )
        
        # Two consecutive convolutional blocks
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        
        # Residual connection handling
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    def forward(self, x, time_emb=None):
        """
        Forward pass with optional time embedding conditioning.
        
        Args:
            x (torch.Tensor): Input tensor
            time_emb (torch.Tensor, optional): Time embedding
        
        Returns:
            torch.Tensor: Processed tensor with residual connection
        """
        h = self.block1(x)
        
        # Conditionally process time embeddings
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
            h = h + time_emb
        
        h = self.block2(h)
        return h + self.res_conv(x)  # Residual connection


class ConvNextBlock(nn.Module):
    """
    ConvNeXt Block with time-conditional processing.
    
    Implements the architecture from 'A ConvNet for the 2020s' paper
    with time embedding support.
    
    Args:
        dim (int): Input channel dimension
        dim_out (int): Output channel dimension
        time_emb_dim (int, optional): Dimension of time embeddings
        mult (int, optional): Expansion multiplier for intermediate layers
        norm (bool, optional): Whether to use group normalization
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        # Optional time embedding processing
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if time_emb_dim is not None
            else None
        )
        
        # Depthwise separable convolution
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        # Complex convolution network
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        
        # Residual connection handling
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    def forward(self, x, time_emb=None):
        """
        Forward pass with optional time embedding conditioning.
        
        Args:
            x (torch.Tensor): Input tensor
            time_emb (torch.Tensor, optional): Time embedding
        
        Returns:
            torch.Tensor: Processed tensor with residual connection
        """
        h = self.ds_conv(x)
        
        # Conditionally process time embeddings
        if self.mlp is not None and time_emb is not None:
            condition = self.mlp(time_emb)
            condition = condition.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
            h = h + condition
        
        h = self.net(h)
        return h + self.res_conv(x)  # Residual connection


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        """
        Unet architecture for generative models with flexible block types.
        
        Args:
            dim (int): Base feature dimension
            init_dim (int, optional): Initial feature dimension
            out_dim (int, optional): Output channel dimension
            dim_mults (tuple): Multipliers for feature dimensions at each scale
            channels (int): Input image channels
            with_time_emb (bool): Whether to use time embeddings
            resnet_block_groups (int): Groups for ResNet blocks
            use_convnext (bool): Use ConvNext blocks instead of ResNet
            convnext_mult (int): Multiplier for ConvNext blocks
        """
        super().__init__()
        
        # Configure basic parameters
        self.channels = channels
        
        # Determine initial dimension
        init_dim = init_dim or (dim // 3 * 2)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        
        # Compute feature dimensions for each scale
        dims = [init_dim]
        for mult in dim_mults:
            dims.append(dim * mult)
        
        # Select block type
        if use_convnext:
            def create_block(dim_in, dim_out):
                return ConvNextBlock(dim_in, dim_out, 
                                     time_emb_dim=time_dim, 
                                     mult=convnext_mult)
        else:
            def create_block(dim_in, dim_out):
                return ResnetBlock(dim_in, dim_out, 
                                   time_emb_dim=time_dim, 
                                   groups=resnet_block_groups)
        
        # Time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
        
        # Downsample path
        self.downs = nn.ModuleList()
        for i in range(len(dims) - 1):
            dim_in, dim_out = dims[i], dims[i+1]
            is_last = (i == len(dims) - 2)
            
            down_blocks = nn.ModuleList([
                create_block(dim_in, dim_out),
                create_block(dim_out, dim_out),
                Downsample(dim_out) if not is_last else nn.Identity()
            ])
            self.downs.append(down_blocks)
        
        # Middle (bottleneck) blocks
        mid_dim = dims[-1]
        self.mid_block1 = create_block(mid_dim, mid_dim)
        self.mid_block2 = create_block(mid_dim, mid_dim)
        
        # Upsample path
        self.ups = nn.ModuleList()
        in_out = list(zip(dims[:-1], dims[1:]))
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            
            up_blocks = nn.ModuleList([
                create_block(dim_out * 2, dim_in),
                create_block(dim_in, dim_in),
                Upsample(dim_in) if not is_last else nn.Identity()
            ])
            self.ups.append(up_blocks)
        
        # Final convolution
        out_dim = out_dim or channels
        self.final_conv = nn.Sequential(
            create_block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        # Initial convolution
        x = self.init_conv(x)
        
        # Process time embeddings
        t = self.time_mlp(time) if self.time_mlp is not None else None
        
        # Skip connections for residual blocks
        skip_connections = []
        
        # Downsample path
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            skip_connections.append(x)
            x = downsample(x)
        
        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # Upsample path
        for block1, block2, upsample in self.ups:
            # Concatenate with skip connection
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)
        
        # Final convolution
        return self.final_conv(x)


class DiffusionSchedule:
    """
    Utility class for creating and managing noise schedules in diffusion models.
    
    This class provides multiple schedule types for generating beta values
    that control the noise addition process in diffusion models.
    """
    
    @staticmethod
    def cosine_beta_schedule(timesteps, smoothing=0.008):
        """
        Generate a cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672
        
        Args:
            timesteps (int): Number of diffusion steps
            smoothing (float): Smoothing parameter to prevent singularities
        
        Returns:
            torch.Tensor: Beta values for each timestep
        """
        # Create steps including the initial step
        total_steps = timesteps + 1
        x = torch.linspace(0, timesteps, total_steps)
        
        # Compute cumulative product of alphas using cosine function
        # This creates a non-linear noise schedule that starts slow and accelerates
        alphas_cumprod = torch.cos(
            ((x / timesteps) + smoothing) / (1 + smoothing) * math.pi * 0.5
        ) ** 2
        
        # Normalize the cumulative product
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Compute betas from the difference in cumulative alphas
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clip betas to prevent extreme values
        return torch.clamp(betas, 0.0001, 0.9999)
    
    @staticmethod
    def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
        """
        Generate a linear beta schedule.
        
        Args:
            timesteps (int): Number of diffusion steps
            start (float): Initial beta value
            end (float): Final beta value
        
        Returns:
            torch.Tensor: Linearly spaced beta values
        """
        return torch.linspace(start, end, timesteps)
    
    @staticmethod
    def quadratic_beta_schedule(timesteps, start=0.0001, end=0.02):
        """
        Generate a quadratic beta schedule.
        
        Args:
            timesteps (int): Number of diffusion steps
            start (float): Initial beta value
            end (float): Final beta value
        
        Returns:
            torch.Tensor: Quadratically spaced beta values
        """
        # Take square root before interpolation, then square
        interpolated = torch.linspace(start**0.5, end**0.5, timesteps)
        return interpolated ** 2
    
    @staticmethod
    def sigmoid_beta_schedule(timesteps, start=0.0001, end=0.02):
        """
        Generate a sigmoid-based beta schedule.
        
        Args:
            timesteps (int): Number of diffusion steps
            start (float): Initial beta value
            end (float): Final beta value
        
        Returns:
            torch.Tensor: Sigmoid-scaled beta values
        """
        # Create a range of values and apply sigmoid scaling
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (end - start) + start
    
    @staticmethod
    def compute_diffusion_parameters(betas):
        """
        Compute additional parameters needed for the diffusion process.
        
        Args:
            betas (torch.Tensor): Beta values for each timestep
        
        Returns:
            dict: Computed diffusion parameters
        """
        # Compute alphas (1 - beta)
        alphas = 1.0 - betas
        
        # Cumulative product of alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Pad alphas_cumprod to include initial step
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Compute various useful transformations
        return {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'alphas_cumprod_prev': alphas_cumprod_prev,
            'sqrt_recip_alphas': torch.sqrt(1.0 / alphas),
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
            'posterior_variance': betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        }

# Example usage
timesteps = 200
betas = DiffusionSchedule.linear_beta_schedule(timesteps).clone()
diffusion_params = DiffusionSchedule.compute_diffusion_parameters(betas)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion
def q_sample(x_start, t, diffusion_params=diffusion_params, noise=None):
    """
    Forward diffusion process (using the nice property).
    
    Args:
        x_start (torch.Tensor): Initial input
        t (torch.Tensor): Timestep
        diffusion_params (dict): Diffusion parameters from DiffusionSchedule
        noise (torch.Tensor, optional): Noise tensor. Defaults to None.
    
    Returns:
        torch.Tensor: Noisy input at timestep t
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(
        diffusion_params['sqrt_alphas_cumprod'], t, x_start.shape
    )
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, diffusion_params=diffusion_params, noise=None, loss_type="l1"):
    """
    Compute loss for the denoising process.
    
    Args:
        denoise_model (nn.Module): Denoising neural network
        x_start (torch.Tensor): Initial input
        t (torch.Tensor): Timestep
        diffusion_params (dict): Diffusion parameters from DiffusionSchedule
        noise (torch.Tensor, optional): Noise tensor. Defaults to None.
        loss_type (str, optional): Type of loss. Defaults to "l1".
    
    Returns:
        torch.Tensor: Computed loss
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_noisy = q_sample(x_start=x_start, t=t, diffusion_params=diffusion_params, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    
    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index, diffusion_params=diffusion_params):
    """
    Reverse diffusion sampling process.
    
    Args:
        model (nn.Module): Noise prediction model
        x (torch.Tensor): Current noisy input
        t (torch.Tensor): Current timestep
        t_index (int): Index of the current timestep
        diffusion_params (dict): Diffusion parameters from DiffusionSchedule
    
    Returns:
        torch.Tensor: Denoised sample
    """
    betas_t = extract(diffusion_params['betas'], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = extract(diffusion_params['sqrt_recip_alphas'], t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(diffusion_params['posterior_variance'], t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample(model, image_size, diffusion_params, batch_size=32, channels=3):
    """
    Generate samples using a diffusion probabilistic model through a reverse process.
    
    Args:
        model (nn.Module): The trained diffusion model
        image_size (int): Size of the generated image (assumed square)
        diffusion_params (dict): Parameters controlling the diffusion process
        batch_size (int, optional): Number of images to generate. Defaults to 32.
        channels (int, optional): Number of color channels. Defaults to 3.
    
    Returns:
        torch.Tensor: Generated images with shape (batch_size, channels, image_size, image_size)
    """
    # Determine the device (GPU/CPU) based on the model's parameters
    device = next(model.parameters()).device
    
    # Define the complete shape of the images to be generated
    shape = (batch_size, channels, image_size, image_size)
    
    # Initialize the image with pure random noise
    # This serves as the starting point for the reverse diffusion process
    img = torch.randn(shape, device=device)
    
    # List to store intermediate images during the sampling process (optional, for visualization/debugging)
    imgs = []
    
    # Iterate through timesteps in reverse order (from noisy to clean)
    # Uses tqdm for a progress bar during the sampling loop
    for i in tqdm(reversed(range(0, timesteps)), 
                  desc='sampling loop time step', 
                  total=timesteps):
        
        # Perform a single denoising step using the p_sample function
        img = p_sample(
            model, 
            img, 
            torch.full((batch_size,), i, device=device, dtype=torch.long),  # Time tensor
            i,  # Current timestep 
            diffusion_params
        )
        
        # Optionally store intermediate images
        imgs.append(img.detach().clone())
    
    # Return either the final denoised image or the stack of intermediate images
    return torch.stack(imgs)




        