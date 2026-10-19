import torch
import torch.nn as nn
import math

#seed = 42
#torch.manual_seed(seed=seed)
#torch.use_deterministic_algorithms(True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        motion_dim: int = 56,
        hidden_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 120
    ):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Motion embedding
        self.motion_embedding = nn.Linear(motion_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4*hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, motion_dim)
        
    def to(self, device):
        # Ensure all components move to the correct device
        super().to(device)
        return self
        
    def forward(
        self, 
        x: torch.Tensor,           # [batch_size, sequence_length, motion_dim]
        timesteps: torch.Tensor    # [batch_size]
    ) -> torch.Tensor:
        # Ensure input tensors are on the same device as model
        device = next(self.parameters()).device
        x = x.to(device)
        timesteps = timesteps.float().to(device)
        
        # Time embedding
        time_emb = self.time_mlp(timesteps.unsqueeze(-1))
        
        # Embed motion sequence
        x = self.motion_embedding(x)  # [batch_size, sequence_length, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Add time embedding to each position
        x = x + time_emb.unsqueeze(1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project back to motion dimension
        x = self.output(x)  # [batch_size, sequence_length, motion_dim]
        
        return x

class GaussianDiffusion:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        self.timesteps = timesteps
        
        # Define beta schedule (keep on CPU initially)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # Pre-calculate diffusion parameters (keep on CPU initially)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Ensure t is on CPU since parameters are on CPU
        t = t.cpu()
        
        # Get values from parameters (kept on CPU)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Move results to same device as input
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(x_start.device)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int
    ) -> torch.Tensor:
        """Single step of reverse diffusion sampling"""
        # Ensure t is on CPU since parameters are on CPU
        t = t.cpu()
        
        # Get values from parameters (kept on CPU)
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Move to same device as input
        betas_t = betas_t.to(x.device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(x.device)
        
        # Predict noise
        predicted_noise = model(x, t)
        
        # No noise at step 0
        if t_index == 0:
            noise = 0
        else:
            noise = torch.randn_like(x)
            
        return 1 / torch.sqrt(1. - betas_t) * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        ) + torch.sqrt(betas_t) * noise
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        sequence_length: int,
        motion_dim: int,
        device: str = "cuda"
    ) -> torch.Tensor:
        """Generate samples"""
        model.eval()
        
        # Start from pure noise
        x = torch.randn(batch_size, sequence_length, motion_dim).to(device)
        
        # Gradually denoise
        for t_index in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), t_index, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, t_index)
            
        return x


