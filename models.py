import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class DeepLINK(nn.Module):
    """DeepLINK architecture for feature selection with knockoffs."""
    def __init__(self, input_dim, hidden_dims=[100, 50], output_dim=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim

        self.filter_weights = nn.Parameter(torch.ones(input_dim, 2) * 0.5)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, x_tilde):
        z = self.filter_weights[:, 0]
        z_tilde = self.filter_weights[:, 1]

        norm = torch.abs(z) + torch.abs(z_tilde) + 1e-8
        z = z / norm
        z_tilde = z_tilde / norm

        filtered_features = x * z.unsqueeze(0) + x_tilde * z_tilde.unsqueeze(0)
        output = self.mlp(filtered_features)
        return output

    def get_filter_weights(self):
        z = self.filter_weights[:, 0]
        z_tilde = self.filter_weights[:, 1]

        norm = torch.abs(z) + torch.abs(z_tilde) + 1e-8
        z = z / norm
        z_tilde = z_tilde / norm

        return z, z_tilde


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalLayerNorm(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Linear(time_dim, dim)
        self.shift = nn.Linear(time_dim, dim)

    def forward(self, x, t):
        normalized = self.norm(x)
        scale = self.scale(t).unsqueeze(1)
        shift = self.shift(t).unsqueeze(1)
        return normalized * (1 + scale) + shift


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=None, dropout=0.1, time_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        mlp_dim = mlp_dim or dim * 4

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm1 = ConditionalLayerNorm(dim, time_dim) if time_dim else nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.norm2 = ConditionalLayerNorm(dim, time_dim) if time_dim else nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, t=None):
        if t is not None:
            h = self.norm1(x, t)
        else:
            h = self.norm1(x)

        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], -1)
        x = x + self.to_out(out)

        if t is not None:
            h = self.norm2(x, t)
        else:
            h = self.norm2(x)
        x = x + self.mlp(h)

        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, depth=6, heads=8, dim_head=32,
                 mlp_dim=None, dropout=0.1, time_dim=128):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout, time_dim)
            for _ in range(depth)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        batch_size, seq_len, _ = x.shape

        t_emb = self.time_mlp(t)
        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :seq_len, :]

        for block in self.transformer_blocks:
            h = block(h, t_emb)

        h = self.output_norm(h)
        output = self.output_proj(h)
        return output


class ImprovedDDPM:
    """Improved Denoising Diffusion Probabilistic Model with cosine scheduling."""
    def __init__(self, model, beta_start=0.0001, beta_end=0.02, num_timesteps=1000,
                 loss_type='l2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.device = device

        self.betas = self._cosine_beta_schedule(num_timesteps, beta_start, beta_end).to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]]).to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(device)

    def _cosine_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        if self.loss_type == 'l2':
            loss = nn.functional.mse_loss(predicted_noise, noise)
        elif self.loss_type == 'l1':
            loss = nn.functional.l1_loss(predicted_noise, noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.betas.gather(-1, t).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.gather(-1, t).view(-1, 1, 1)

        model_output = self.model(x, t)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance.gather(-1, t).view(-1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape, return_intermediates=False):
        device = self.device
        batch_size = shape[0]

        x = torch.randn(shape, device=device)
        intermediates = [x]

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i)

            if return_intermediates and i % 100 == 0:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates

        return x


class Autoencoder(nn.Module):
    """Simple autoencoder for knockoff generation."""
    def __init__(self, input_dim, bottleneck_dim, activation='relu'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ELU() if activation == 'elu' else nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.ELU() if activation == 'elu' else nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
