import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
import os
import matplotlib
matplotlib.rcParams['savefig.format'] = 'pdf'

# Create result directory
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result_modified_new_function')
os.makedirs(RESULT_DIR, exist_ok=True)


# ============= DeepLINK Components =============

class DeepLINK(nn.Module):
    """DeepLINK architecture for feature selection with knockoffs - optimized for TPM"""
    def __init__(self, input_dim, hidden_dims=[100, 50], output_dim=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        
        # Pairwise filter layer - properly initialized
        self.filter_weights = nn.Parameter(torch.ones(input_dim, 2) * 0.5)
        
        # MLP layers with LayerNorm for TPM data
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Add LayerNorm for TPM data
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x, x_tilde):
        """Forward pass with original features x and knockoffs x_tilde"""
        # Get filter weights
        z = self.filter_weights[:, 0]
        z_tilde = self.filter_weights[:, 1]
        
        # Normalize filters to sum to 1 (antisymmetric constraint)
        norm = torch.abs(z) + torch.abs(z_tilde) + 1e-8
        z = z / norm
        z_tilde = z_tilde / norm
        
        # Compute filtered features
        filtered_features = x * z.unsqueeze(0) + x_tilde * z_tilde.unsqueeze(0)
        
        # Pass through MLP
        output = self.mlp(filtered_features)
        
        return output
    
    def get_filter_weights(self):
        """Get the normalized filter weights for knockoff statistics"""
        z = self.filter_weights[:, 0]
        z_tilde = self.filter_weights[:, 1]
        
        # Normalize
        norm = torch.abs(z) + torch.abs(z_tilde) + 1e-8
        z = z / norm
        z_tilde = z_tilde / norm
        
        return z, z_tilde

# ============= Diffusion based Knockoff =============
class DiffKnock:
    """Diffusion-based Knockoff Generation for TPM data"""
    def __init__(self, input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.device = device
        self.diffusion = None
        self.deepLINK = None
        self.scaler = None
        
        
    def train_diffusion_for_knockoffs(self, X_data, n_epochs=500, batch_size=64, lr=1e-4):
        """Train diffusion model to learn the data distribution"""
        print("Training diffusion model for knockoff generation...")
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_data).unsqueeze(1)  # Add sequence dimension
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              pin_memory=torch.cuda.is_available())
        
        # Initialize diffusion model
        model = DiffusionTransformer(
            input_dim=self.input_dim,
            hidden_dim=256,
            depth=6,
            heads=8,
            dim_head=32,
            dropout=0.1
        )
        
        self.diffusion = ImprovedDDPM(model, num_timesteps=1000, device=self.device)
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        # Training loop
        losses = []
        model.train()
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_idx, (batch,) in enumerate(dataloader):
                batch = batch.to(self.device)
                batch_size_actual = batch.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.diffusion.num_timesteps, 
                                 (batch_size_actual,), device=self.device).long()
                
                # Compute loss
                loss = self.diffusion.p_losses(batch, t)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def generate_knockoffs(self, X_data):
        print("Generating knockoff variables...")
        
        with torch.no_grad():
            # Generate same number of samples as original data
            n_samples = X_data.shape[0]
            X_knockoffs = self.diffusion.sample((n_samples, 1, self.input_dim))
            X_knockoffs = X_knockoffs.squeeze(1).cpu().numpy()
        
        
        return X_knockoffs
    
    def compute_gradient_based_statistics(self, X, X_tilde, y, model, criterion, batch_size=100):
        """Compute gradient-based importance statistics with proper batching"""
        print("Computing gradient-based statistics...")
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Initialize gradient accumulators
        grad_X_accum = np.zeros(n_features)
        grad_X_tilde_accum = np.zeros(n_features)
        
        # Process in batches to avoid memory issues
        model.eval()
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            
            # Get batch
            X_batch = torch.FloatTensor(X[i:end_idx]).to(self.device).requires_grad_(True)
            X_tilde_batch = torch.FloatTensor(X_tilde[i:end_idx]).to(self.device).requires_grad_(True)
            y_batch = torch.FloatTensor(y[i:end_idx]).to(self.device)
            
            # Forward pass
            output = model(X_batch, X_tilde_batch)
            loss = criterion(output.squeeze(), y_batch)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, [X_batch, X_tilde_batch], create_graph=False)
            
            # Accumulate absolute gradients
            grad_X_accum += torch.abs(grads[0]).sum(dim=0).detach().cpu().numpy()
            grad_X_tilde_accum += torch.abs(grads[1]).sum(dim=0).detach().cpu().numpy()
        
        # Average gradients
        G_j = grad_X_accum / n_samples
        G_j_tilde = grad_X_tilde_accum / n_samples
        
        # Compute knockoff statistics
        W_grad = G_j - G_j_tilde
        
        return W_grad
    
    def compute_deeplink_statistics(self, model):
        """Compute filter-based statistics - ORIGINAL VERSION"""
        print("Computing filter-based statistics...")
        
        # Get filter weights
        z, z_tilde = model.get_filter_weights()
        
        # Get MLP weights to compute importance (matching original DeepLINK)
        with torch.no_grad():
            # Extract Linear layers from the MLP
            linear_layers = []
            for module in model.mlp:
                if isinstance(module, nn.Linear):
                    linear_layers.append(module)
            
            # Compute combined weight w by multiplying through the network
            # Start from output layer and work backwards
            w = linear_layers[-1].weight.data  # Shape: (1, hidden_dim_last)
            
            # Multiply through all linear layers (excluding the last one we already have)
            for layer in reversed(linear_layers[:-1]):
                w = w @ layer.weight.data  # Matrix multiplication through the network
            
            # w now has shape (1, input_dim) where input_dim = p_genes
            w = w.squeeze()  # Remove the batch dimension, shape: (p_genes,)
        
        # Compute knockoff statistics as in original paper: W = (w * z)² - (w * z_tilde)²
        w = w.cpu().numpy()
        z = z.detach().cpu().numpy()  # ADD detach() here
        z_tilde = z_tilde.detach().cpu().numpy()  # ADD detach() here
        
        W_filt = (w * z) ** 2 - (w * z_tilde) ** 2
        
        return W_filt

    
    def select_features(self, W_j, fdr_threshold=0.2):
        def knockoff_plus_threshold(W, q):
            # Get the absolute values of statistics
            abs_W = np.abs(W)
            
            # Sort in descending order
            sorted_indices = np.argsort(abs_W)[::-1]
            sorted_W = W[sorted_indices]
            sorted_abs_W = abs_W[sorted_indices]
            
            # Find threshold using the knockoff+ procedure
            threshold = np.inf
            
            # Consider all possible thresholds (including positive values)
            unique_vals = np.unique(sorted_abs_W[sorted_abs_W > 0])
            
            if len(unique_vals) == 0:
                # No non-zero statistics, can't select anything
                return np.inf
            
            # Add a small value below the minimum to consider selecting all
            thresholds_to_try = np.append(unique_vals, [0, unique_vals.min() / 2])
            thresholds_to_try = np.sort(thresholds_to_try)[::-1]  # Descending order
            
            for t in thresholds_to_try:
                # Count discoveries and false discoveries
                selected = W >= t
                n_selected = np.sum(selected)
                
                if n_selected == 0:
                    continue
                    
                # Knockoff+ estimate of false discoveries
                n_negative = np.sum(W <= -t)
                
                # FDP estimate with offset=1 for knockoff+
                fdp_estimate = (1 + n_negative) / max(n_selected, 1)
                
                # Check if this threshold achieves target FDR
                if fdp_estimate <= q:
                    threshold = t
                    break
            
            return threshold
        
        threshold = knockoff_plus_threshold(W_j, fdr_threshold)
        selected_features = np.where(W_j >= threshold)[0]
        
        return selected_features

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps"""
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
    """Conditional Layer Normalization with time embedding"""
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
    """Transformer block with cross-attention and time conditioning"""
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=None, dropout=0.1, time_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        mlp_dim = mlp_dim or dim * 4
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # Self-attention components
        self.norm1 = ConditionalLayerNorm(dim, time_dim) if time_dim else nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # MLP components
        self.norm2 = ConditionalLayerNorm(dim, time_dim) if time_dim else nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, t=None):
        # Self-attention
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
        
        # MLP
        if t is not None:
            h = self.norm2(x, t)
        else:
            h = self.norm2(x)
        x = x + self.mlp(h)
        
        return x

class DiffusionTransformer(nn.Module):
    """State-of-the-art Diffusion Transformer for TPM data"""
    def __init__(self, input_dim, hidden_dim=256, depth=6, heads=8, dim_head=32, 
                 mlp_dim=None, dropout=0.1, time_dim=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Positional encoding for sequence elements
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout, time_dim)
            for _ in range(depth)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t):
        # x shape: (batch_size, seq_len, input_dim)
        # t shape: (batch_size,)
        
        batch_size, seq_len, _ = x.shape
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Project input
        h = self.input_proj(x)
        
        # Add positional encoding
        h = h + self.pos_embedding[:, :seq_len, :]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            h = block(h, t_emb)
        
        # Output projection
        h = self.output_norm(h)
        output = self.output_proj(h)
        
        return output

class ImprovedDDPM:
    """Improved Denoising Diffusion Probabilistic Model with advanced scheduling"""
    def __init__(self, model, beta_start=0.0001, beta_end=0.02, num_timesteps=1000,
                 loss_type='l2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.device = device
        
        # Improved noise schedule (cosine schedule)
        self.betas = self._cosine_beta_schedule(num_timesteps, beta_start, beta_end).to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]]).to(device)
        
        # Pre-compute values for efficiency
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(device)
        
    def _cosine_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """Cosine schedule for better sampling quality"""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        """Compute training loss"""
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
        """Reverse diffusion process - single step"""
        betas_t = self.betas.gather(-1, t).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.gather(-1, t).view(-1, 1, 1)
        
        # Predict noise
        model_output = self.model(x, t)
        
        # Compute mean
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
        """Generate samples from noise"""
        device = self.device
        batch_size = shape[0]
        
        # Start from pure noise
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

class DiffusionXKnockoff:
    """
    End-to-end pipeline:
      Diffusion knockoffs -> train DeepLINK -> compute W (gradient/filter) -> knockoff+ selection

    Inputs match other pipelines:
      X_processed
      y: np.ndarray (n,)
    """
    def __init__(self, input_dim, device=None):
        self.input_dim = input_dim
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Core diffusion object (generator + stats + selection utilities)
        self.diff = DiffKnock(input_dim=input_dim, device=self.device)

        # DeepLINK model (black box)
        self.deepLINK = None

    def run(
        self,
        X_processed, y,
        fdr_threshold=0.2,

        # diffusion training
        diffusion_epochs=300,
        diffusion_batch_size=64,
        diffusion_lr=1e-4,

        # DeepLINK training
        hidden_dims=[50, 20],
        dropout=0.1,
        deeplink_epochs=300,
        deeplink_batch_size=64,
        deeplink_lr=1e-3,
        deeplink_weight_decay=1e-4,

        # which stats
        use_gradient=True,
        use_filter=True
    ):
        


        # --- train diffusion + generate knockoffs ---
        self.diff.train_diffusion_for_knockoffs(
            X_processed,
            n_epochs=diffusion_epochs,
            batch_size=diffusion_batch_size,
            lr=diffusion_lr
        )
        X_knock = self.diff.generate_knockoffs(X_processed)

        # --- train DeepLINK once ---
        self.deepLINK = DeepLINK(input_dim=self.input_dim, hidden_dims=hidden_dims, dropout=dropout).to(self.device)

        dataset = TensorDataset(
            torch.FloatTensor(X_processed),
            torch.FloatTensor(X_knock),
            torch.FloatTensor(y)
        )
        dataloader = DataLoader(dataset, batch_size=deeplink_batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.deepLINK.parameters(), lr=deeplink_lr, weight_decay=deeplink_weight_decay)

        self.deepLINK.train()
        for epoch in range(deeplink_epochs):
            for Xb, Xkb, yb in dataloader:
                Xb = Xb.to(self.device)
                Xkb = Xkb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                out = self.deepLINK(Xb, Xkb)
                loss = criterion(out.squeeze(), yb)
                loss.backward()
                optimizer.step()

        self.deepLINK.eval()

        # --- compute statistics + select ---
        out = {
            "X_processed": X_processed,
            "X_knockoffs": X_knock,
            "W": {},
            "selected": {}
        }

        if use_gradient:
            W_grad = self.diff.compute_gradient_based_statistics(X_processed, X_knock, y, self.deepLINK, criterion)
            out["W"]["gradient"] = W_grad
            out["selected"]["gradient"] = self.diff.select_features(W_grad, fdr_threshold=fdr_threshold)

        if use_filter:
            W_filt = self.diff.compute_deeplink_statistics(self.deepLINK)
            out["W"]["filter"] = W_filt
            out["selected"]["filter"] = self.diff.select_features(W_filt, fdr_threshold=fdr_threshold)

        return out

# ============= Autoencoder Knockoff =============
class Autoencoder(nn.Module):
    """Simple autoencoder for knockoff generation"""
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


class AutoencoderXKnockoff:
    """
    Autoencoder-based knockoff generation + DeepLINK black-box analysis.
    FIXED: DeepLINK is trained ONCE per dataset (X, X_knockoff, y) and reused for:
      - filter-based W
      - gradient-based W
    """
    def __init__(self, input_dim, r_factors=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.r_factors = r_factors
        self.device = device
        self.autoencoder = None
        self.deepLINK = None
        self._criterion = None  # store criterion after training
        self._trained_signature = None  # simple cache key to avoid accidental reuse across datasets

    def knockoff_construct_autoencoder(self, X, activation='relu', epochs=500, lr=1e-3, verbose=True):
        """Construct knockoffs using autoencoder (original DeepLINK method)"""
        n, p = X.shape

        # Build and train autoencoder
        self.autoencoder = Autoencoder(p, self.r_factors, activation).to(self.device)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Train
        self.autoencoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.autoencoder(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Autoencoder Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Get C estimate
        self.autoencoder.eval()
        with torch.no_grad():
            C = self.autoencoder(X_tensor).cpu().numpy()

        # Construct knockoffs
        E = X - C
        sigma = np.sqrt(np.sum(E ** 2) / (n * p))
        X_knockoff = C + sigma * np.random.randn(n, p)

        return X_knockoff

    # -------------------------
    # NEW: Train DeepLINK once
    # -------------------------
    def train_deeplink(self, X, X_knockoff, y,
                           epochs=500, lr=0.001, batch_size=64,
                           verbose=True, hidden_dims=[50, 20], dropout=0.1):
        """
        Train DeepLINK once on (X, X_knockoff, y) and cache the trained model + criterion.
        Call this once per dataset before computing filter/gradient statistics.
        """
        n, p = X.shape

        # Cache signature (shape + simple checksum) to avoid accidental reuse
        # (lightweight; you can remove if you prefer)
        signature = (X.shape, X_knockoff.shape, y.shape, float(np.mean(X)), float(np.mean(X_knockoff)), float(np.mean(y)))
        self._trained_signature = signature

        # Build DeepLINK model
        self.deepLINK = DeepLINK(input_dim=p, hidden_dims=hidden_dims, dropout=dropout).to(self.device)
        optimizer = optim.Adam(self.deepLINK.parameters(), lr=lr, weight_decay=1e-4)
        self._criterion = nn.MSELoss()

        # Dataloader with (X, X_knockoff, y)
        dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.FloatTensor(X_knockoff).to(self.device),
            torch.FloatTensor(y).to(self.device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        self.deepLINK.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, Xk_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = self.deepLINK(X_batch, Xk_batch)
                loss = self._criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if verbose and (epoch + 1) % 100 == 0:
                print(f"DeepLINK(AE) Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        self.deepLINK.eval()
        return self

    def _ensure_trained(self, X, X_knockoff, y,
                        epochs=500, lr=0.001, batch_size=64,
                        verbose=True, hidden_dims=[50, 20], dropout=0.1):
        """Train DeepLINK once if not trained or if dataset changed."""
        if self.deepLINK is None or self._criterion is None or self._trained_signature is None:
            self.train_deeplink(X, X_knockoff, y,
                                     epochs=epochs, lr=lr, batch_size=batch_size,
                                     verbose=verbose, hidden_dims=hidden_dims, dropout=dropout)
            return

        signature = (X.shape, X_knockoff.shape, y.shape, float(np.mean(X)), float(np.mean(X_knockoff)), float(np.mean(y)))
        if signature != self._trained_signature:
            # Dataset changed: retrain
            self.train_deeplink(X, X_knockoff, y,
                                     epochs=epochs, lr=lr, batch_size=batch_size,
                                     verbose=verbose, hidden_dims=hidden_dims, dropout=dropout)

    # -------------------------
    # Filter-based statistics (NO training here)
    # -------------------------
    def compute_knockoff_stats_filter(self, X, X_knockoff, y, activation='relu',
                                      epochs=500, lr=0.001, batch_size=64,
                                      verbose=True, hidden_dims=[50, 20], dropout=0.1):
        """
        Compute filter-based knockoff statistics using the trained DeepLINK.
        FIXED: trains DeepLINK once (if needed) and reuses it.
        """
        self._ensure_trained(X, X_knockoff, y,
                             epochs=epochs, lr=lr, batch_size=batch_size,
                             verbose=verbose, hidden_dims=hidden_dims, dropout=dropout)

        # Extract filter weights z, z_tilde
        z, z_tilde = self.deepLINK.get_filter_weights()

        # Compute combined MLP weight vector w
        with torch.no_grad():
            linear_layers = [m for m in self.deepLINK.mlp if isinstance(m, nn.Linear)]
            w = linear_layers[-1].weight.data  # (1, hidden_last)
            for layer in reversed(linear_layers[:-1]):
                w = w @ layer.weight.data
            w = w.squeeze().cpu().numpy()
            z = z.detach().cpu().numpy()
            z_tilde = z_tilde.detach().cpu().numpy()

        # Knockoff statistic
        W_filt = (w * z) ** 2 - (w * z_tilde) ** 2
        return W_filt

    # -------------------------
    # Gradient-based statistics (NO training here)
    # -------------------------
    def compute_knockoff_stats_gradient(self, X, X_knockoff, y,
                                        epochs=500, lr=0.001, batch_size=64,
                                        verbose=True, hidden_dims=[50, 20], dropout=0.1):
        """
        Compute gradient-based knockoff statistics using the trained DeepLINK.
        FIXED: trains DeepLINK once (if needed) and reuses it.
        """
        self._ensure_trained(X, X_knockoff, y,
                             epochs=epochs, lr=lr, batch_size=batch_size,
                             verbose=verbose, hidden_dims=hidden_dims, dropout=dropout)

        # Use DiffKnock's existing gradient statistic utility
        diff_util = DiffKnock(input_dim=X.shape[1])
        diff_util.device = self.device

        self.deepLINK.eval()
        W_grad = diff_util.compute_gradient_based_statistics(
            X, X_knockoff, y, self.deepLINK, self._criterion
        )
        return W_grad

    def select_features(self, W_j, fdr_threshold=0.2):
        """Feature selection - IDENTICAL TO DIFFUSION FRAMEWORK"""
        def knockoff_plus_threshold(W, q):
            abs_W = np.abs(W)
            sorted_indices = np.argsort(abs_W)[::-1]
            sorted_abs_W = abs_W[sorted_indices]

            threshold = np.inf
            unique_vals = np.unique(sorted_abs_W[sorted_abs_W > 0])
            if len(unique_vals) == 0:
                return np.inf

            thresholds_to_try = np.append(unique_vals, [0, unique_vals.min() / 2])
            thresholds_to_try = np.sort(thresholds_to_try)[::-1]

            for t in thresholds_to_try:
                selected = W >= t
                n_selected = np.sum(selected)
                if n_selected == 0:
                    continue

                n_negative = np.sum(W <= -t)
                fdp_estimate = (1 + n_negative) / max(n_selected, 1)

                if fdp_estimate <= q:
                    threshold = t
                    break

            return threshold

        threshold = knockoff_plus_threshold(W_j, fdr_threshold)
        selected_features = np.where(W_j >= threshold)[0]
        return selected_features

# ============= Classic Gaussian Knockoff =============
class Gaussian:
    """
    Model-X Gaussian knockoffs baseline (Barber & Candès style).
    Assumes X is approximately Gaussian; best used on X_processed (quantile-normalized).
    
    Usage:
        gx = Gaussian()
        gx.fit(X_processed)
        X_tilde = gx.generate_knockoffs(X_processed)
    """
    def __init__(self, ridge=1e-6, s_mode="equi"):
        self.ridge = ridge
        self.s_mode = s_mode
        self.Sigma_ = None
        self.Sigma_inv_ = None
        self.s_ = None
        self.C_ = None

    def fit(self, X):
        X = np.asarray(X)
        n, p = X.shape

        # Center X (important)
        Xc = X - X.mean(axis=0, keepdims=True)

        # Sample covariance with ridge for stability
        Sigma = (Xc.T @ Xc) / max(n - 1, 1)
        Sigma = Sigma + self.ridge * np.eye(p)

        # Invert covariance
        Sigma_inv = np.linalg.inv(Sigma)

        # Choose s (equi-correlated by default)
        if self.s_mode != "equi":
            raise NotImplementedError("Only s_mode='equi' implemented for speed/stability.")

        # Equi-correlated: s = min(2 * lambda_min(Sigma), 1) * 1_p
        # We cap at 1 to ensure diag(s) <= 2*Sigma PSD constraints in practice.
        evals = np.linalg.eigvalsh(Sigma)
        lam_min = float(np.min(evals))
        s_val = min(2.0 * lam_min, 1e-3)
        if s_val <= 0:
            # If covariance is nearly singular, fall back to small s
            s_val = 1e-3
        s = s_val * np.ones(p)

        # Compute C s.t. C^T C = 2 diag(s) - diag(s) Sigma^{-1} diag(s)
        D = np.diag(s)
        M = 2 * D - D @ Sigma_inv @ D
        # Numerical symmetrization + jitter
        M = 0.5 * (M + M.T) + self.ridge * np.eye(p)

        # Cholesky might fail if not PSD due to numerical issues; do eigen fallback
        try:
            C = np.linalg.cholesky(M).T  # so that C^T C = M
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(M)
            w = np.clip(w, 0.0, None)
            C = (V * np.sqrt(w)) @ V.T  # symmetric square root; valid for sampling

        self.Sigma_ = Sigma
        self.Sigma_inv_ = Sigma_inv
        self.s_ = s
        self.C_ = C
        return self

    def generate_knockoffs(self, X):
        if self.Sigma_inv_ is None or self.C_ is None or self.s_ is None:
            raise RuntimeError("Call fit(X) before generate_knockoffs(X).")

        X = np.asarray(X)
        n, p = X.shape
        Xc = X - X.mean(axis=0, keepdims=True)

        U = np.random.standard_normal(size=(n, p))

        D = np.diag(self.s_)
        # X_tilde = X(I - Sigma^{-1} D) + U C
        X_tilde = Xc @ (np.eye(p) - self.Sigma_inv_ @ D) + U @ self.C_

        # Add back mean so knockoffs live on same location scale as X input
        X_tilde = X_tilde + X.mean(axis=0, keepdims=True)
        return X_tilde

class GaussianXKnockoff:
    """
    End-to-end pipeline:
      Gaussian Model-X knockoffs -> train DeepLINK -> compute W -> knockoff+ selection

    Inputs match DiffKnock usage:
      X_processed: np.ndarray (n, p)
      y: np.ndarray (n,)
    """
    def __init__(self, input_dim, device=None, ridge=1e-1, s_mode="equi"):
        self.input_dim = input_dim
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Knockoff generator
        self.gx = Gaussian(ridge=ridge, s_mode=s_mode)

        # DeepLINK model (black box)
        self.deepLINK = None

        # We reuse DiffKnock utilities for:
        # - compute_gradient_based_statistics
        # - compute_deeplink_statistics
        # - select_features
        self._diff_util = DiffKnock(input_dim=input_dim)
        self._diff_util.device = self.device  # ensure consistent device

    def fit_generator(self, X_processed):
        """Fit Gaussian Model-X generator on X_processed."""
        self.gx.fit(X_processed)
        return self

    def generate_knockoffs(self, X_processed):
        """Generate Gaussian knockoffs for X_processed."""
        return self.gx.generate_knockoffs(X_processed)

    def train_deeplink(self, X_processed, X_knockoffs, y,
                       hidden_dims=[50, 20], dropout=0.1,
                       epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-4):
        """Train DeepLINK (black box) on (X, X_knockoffs, y)."""
        self.deepLINK = DeepLINK(input_dim=self.input_dim, hidden_dims=hidden_dims, dropout=dropout).to(self.device)

        dataset = TensorDataset(
            torch.FloatTensor(X_processed),
            torch.FloatTensor(X_knockoffs),
            torch.FloatTensor(y)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.deepLINK.parameters(), lr=lr, weight_decay=weight_decay)

        self.deepLINK.train()
        for epoch in range(epochs):
            for Xb, Xkb, yb in dataloader:
                Xb = Xb.to(self.device)
                Xkb = Xkb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                out = self.deepLINK(Xb, Xkb)
                loss = criterion(out.squeeze(), yb)
                loss.backward()
                optimizer.step()

        return criterion

    def compute_statistics(self, X_processed, X_knockoffs, y, criterion, use_gradient=True, use_filter=True):
        """
        Compute knockoff statistics W.
        - Gradient-based uses DiffKnock.compute_gradient_based_statistics
        - Filter-based uses DiffKnock.compute_deeplink_statistics
        """
        if self.deepLINK is None:
            raise RuntimeError("Call train_deeplink(...) before compute_statistics(...).")

        stats = {}

        if use_gradient:
            W_grad = self._diff_util.compute_gradient_based_statistics(
                X_processed, X_knockoffs, y, self.deepLINK, criterion
            )
            stats["gradient"] = W_grad

        if use_filter:
            W_filt = self._diff_util.compute_deeplink_statistics(self.deepLINK)
            stats["filter"] = W_filt

        return stats

    def select(self, W, fdr_threshold=0.2):
        """Knockoff+ selection using same method as DiffKnock."""
        return self._diff_util.select_features(W, fdr_threshold=fdr_threshold)

    def run(self, X_processed, y,
            fdr_threshold=0.2,
            hidden_dims=[50, 20], dropout=0.1,
            train_epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-4,
            use_gradient=True, use_filter=True):
        """
        Full run:
          fit generator -> generate knockoffs -> train DeepLINK -> compute W -> select
        Returns:
          dict with knockoffs, W, selected sets.
        """
        self.fit_generator(X_processed)
        X_knockoffs = self.generate_knockoffs(X_processed)
        criterion = self.train_deeplink(
            X_processed, X_knockoffs, y,
            hidden_dims=hidden_dims, dropout=dropout,
            epochs=train_epochs, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay
        )
        stats = self.compute_statistics(
            X_processed, X_knockoffs, y, criterion,
            use_gradient=use_gradient, use_filter=use_filter
        )

        out = {"X_knockoffs": X_knockoffs, "W": {}, "selected": {}}

        if use_gradient:
            out["W"]["gradient"] = stats["gradient"]
            out["selected"]["gradient"] = self.select(stats["gradient"], fdr_threshold=fdr_threshold)

        if use_filter:
            out["W"]["filter"] = stats["filter"]
            out["selected"]["filter"] = self.select(stats["filter"], fdr_threshold=fdr_threshold)

        return out

# ============= GAN-Based Knockoff =============
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

class GAN:
    """
    Practical KnockoffGAN-style baseline:
      - Generator G(x, z) -> x_tilde
      - Critic D(x, x_tilde) scores pairs
      - Train D to distinguish (x, x_tilde) vs swapped (x_tilde, x)
      - Train G to fool D + moment matching regularizer
    
    Usage:
        kg = GAN(input_dim=p, z_dim=64)
        kg.train_gan_for_knockoffs(X_processed, n_epochs=500, batch_size=128)
        X_tilde = kg.generate_knockoffs(X_processed)
    """
    def __init__(
        self,
        input_dim,
        z_dim=64,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        g_hidden=(64, 64),
        d_hidden=(64, 64),
        dropout=0.1,
        lr_g=2e-4,
        lr_d=2e-4,
        n_critic=2,
        lambda_mom=0.5,
    ):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.device = device
        self.n_critic = n_critic
        self.lambda_mom = lambda_mom

        # Generator: takes [x, z] and outputs x_tilde
        self.G = _MLP(input_dim + z_dim, input_dim, hidden_dims=g_hidden, dropout=dropout).to(device)
        # Critic: takes [x, x_tilde] and outputs a scalar score
        self.D = _MLP(2 * input_dim, 1, hidden_dims=d_hidden, dropout=dropout).to(device)

        self.optG = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.optD = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.9))


    def _sample_z(self, n):
        return torch.randn(n, self.z_dim, device=self.device)

    @staticmethod
    def _moment_loss(x, x_tilde):
        """
        Simple moment matching: match mean and covariance (second moment).
        Uses batch estimates for efficiency.
        """
        x0 = x - x.mean(dim=0, keepdim=True)
        xt0 = x_tilde - x_tilde.mean(dim=0, keepdim=True)

        mean_loss = (x.mean(dim=0) - x_tilde.mean(dim=0)).pow(2).mean()

        # Covariance mismatch (Frobenius, normalized)
        cov_x = (x0.T @ x0) / max(x0.shape[0] - 1, 1)
        cov_xt = (xt0.T @ xt0) / max(xt0.shape[0] - 1, 1)
        cov_loss = (cov_x - cov_xt).pow(2).mean()

        return mean_loss + cov_loss

    def train_gan_for_knockoffs(self, X_data, n_epochs=500, batch_size=128, verbose=True):
        X = np.asarray(X_data, dtype=np.float32)
        n, p = X.shape
        if p != self.input_dim:
            raise ValueError(f"X has p={p}, but GAN initialized with input_dim={self.input_dim}")

        ds = TensorDataset(torch.from_numpy(X))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

        bce = nn.BCEWithLogitsLoss()

        for epoch in range(n_epochs):
            for (x_batch,) in dl:
                x_batch = x_batch.to(self.device)

                # ---------------------
                # Train Critic / Discriminator
                # ---------------------
                for _ in range(self.n_critic):
                    self.D.train()
                    self.optD.zero_grad()

                    z = self._sample_z(x_batch.shape[0])
                    x_tilde = self.G(torch.cat([x_batch, z], dim=1)).detach()

                    # D tries to tell (x, x_tilde) from swapped (x_tilde, x)
                    real_pairs = torch.cat([x_batch, x_tilde], dim=1)
                    swap_pairs = torch.cat([x_tilde, x_batch], dim=1)

                    logits_real = self.D(real_pairs)
                    logits_swap = self.D(swap_pairs)

                    # Labels: real_pairs=1, swap_pairs=0
                    lossD = bce(logits_real, torch.ones_like(logits_real)) + \
                            bce(logits_swap, torch.zeros_like(logits_swap))

                    lossD.backward()
                    self.optD.step()

                # ---------------------
                # Train Generator
                # ---------------------
                self.G.train()
                self.optG.zero_grad()

                z = self._sample_z(x_batch.shape[0])
                x_tilde = self.G(torch.cat([x_batch, z], dim=1))

                real_pairs = torch.cat([x_batch, x_tilde], dim=1)
                logits_real = self.D(real_pairs)

                # Generator wants D(real_pairs) -> 0.5 (i.e., confuse with swap)
                # Here: make D label them as 0 (or equivalently minimize ability to distinguish)
                # Using "swap" label to push towards invariance.
                lossG_adv = bce(logits_real, torch.zeros_like(logits_real))

                lossG_mom = self._moment_loss(x_batch, x_tilde)
                lossG = lossG_adv + self.lambda_mom * lossG_mom

                lossG.backward()
                self.optG.step()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"[KnockoffGAN] Epoch {epoch+1}/{n_epochs} | lossG={lossG.item():.4f} (adv={lossG_adv.item():.4f}, mom={lossG_mom.item():.4f})")

        return self

    @torch.no_grad()
    def generate_knockoffs(self, X_data, batch_size=512):
        """
        Generate knockoffs for input X_data. Shape preserved: (n, p).
        """
        X = np.asarray(X_data, dtype=np.float32)
        n, p = X.shape
        if p != self.input_dim:
            raise ValueError(f"X has p={p}, but GAN initialized with input_dim={self.input_dim}")

        self.G.eval()
        out = np.zeros_like(X, dtype=np.float32)

        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(self.device)
            z = self._sample_z(xb.shape[0])
            xt = self.G(torch.cat([xb, z], dim=1))
            out[i:i+batch_size] = xt.cpu().numpy()

        return out

class GANXKnockoff:
    """
    End-to-end pipeline:
      KnockoffGAN -> train DeepLINK -> compute W (gradient/filter) -> knockoff+ selection

    Inputs match DiffKnock usage:
      X_processed: np.ndarray (n, p)
      y: np.ndarray (n,)
    """
    def __init__(
        self,
        input_dim,
        device=None,
        z_dim=64,
        g_hidden=(256, 256),
        d_hidden=(256, 256),
        dropout=0.1,
        lr_g=2e-4,
        lr_d=2e-4,
        n_critic=2,
        lambda_mom=10.0,
    ):
        self.input_dim = input_dim
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Knockoff generator
        self.gan = GAN(
            input_dim=input_dim,
            z_dim=z_dim,
            device=self.device,
            g_hidden=g_hidden,
            d_hidden=d_hidden,
            dropout=dropout,
            lr_g=lr_g,
            lr_d=lr_d,
            n_critic=n_critic,
            lambda_mom=lambda_mom
        )

        # DeepLINK model (black box)
        self.deepLINK = None

        # Reuse DiffKnock utilities for:
        # - compute_gradient_based_statistics
        # - compute_deeplink_statistics
        # - select_features
        self._diff_util = DiffKnock(input_dim=input_dim)
        self._diff_util.device = self.device  # ensure consistent device

    def fit_generator(self, X_processed, n_epochs=500, batch_size=128, verbose=True):
        """Train GAN knockoff generator on X_processed."""
        self.gan.train_gan_for_knockoffs(X_processed, n_epochs=n_epochs, batch_size=batch_size, verbose=verbose)
        return self

    def generate_knockoffs(self, X_processed, batch_size=512):
        """Generate knockoffs using trained GAN."""
        return self.gan.generate_knockoffs(X_processed, batch_size=batch_size)

    def train_deeplink(
        self,
        X_processed, X_knockoffs, y,
        hidden_dims=[50, 20], dropout=0.1,
        epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-4
    ):
        """Train DeepLINK (black box) on (X, X_knockoffs, y)."""
        self.deepLINK = DeepLINK(input_dim=self.input_dim, hidden_dims=hidden_dims, dropout=dropout).to(self.device)

        dataset = TensorDataset(
            torch.FloatTensor(X_processed),
            torch.FloatTensor(X_knockoffs),
            torch.FloatTensor(y)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.deepLINK.parameters(), lr=lr, weight_decay=weight_decay)

        self.deepLINK.train()
        for epoch in range(epochs):
            for Xb, Xkb, yb in dataloader:
                Xb = Xb.to(self.device)
                Xkb = Xkb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                out = self.deepLINK(Xb, Xkb)
                loss = criterion(out.squeeze(), yb)
                loss.backward()
                optimizer.step()

        return criterion

    def compute_statistics(self, X_processed, X_knockoffs, y, criterion, use_gradient=True, use_filter=True):
        """
        Compute knockoff statistics W:
          - Gradient-based: DiffKnock.compute_gradient_based_statistics(...)
          - Filter-based:   DiffKnock.compute_deeplink_statistics(deeplink_model)
        """
        if self.deepLINK is None:
            raise RuntimeError("Call train_deeplink(...) before compute_statistics(...).")

        stats = {}

        if use_gradient:
            W_grad = self._diff_util.compute_gradient_based_statistics(
                X_processed, X_knockoffs, y, self.deepLINK, criterion
            )
            stats["gradient"] = W_grad

        if use_filter:
            W_filt = self._diff_util.compute_deeplink_statistics(self.deepLINK)
            stats["filter"] = W_filt

        return stats

    def select(self, W, fdr_threshold=0.2):
        """Knockoff+ selection using DiffKnock's select_features (same as diffusion baseline)."""
        return self._diff_util.select_features(W, fdr_threshold=fdr_threshold)

    def run(
        self,
        X_processed, y,
        fdr_threshold=0.2,

        # GAN training
        gan_epochs=500,
        gan_batch_size=128,
        gan_verbose=True,
        gan_sample_batch_size=512,

        # DeepLINK training
        hidden_dims=[50, 20],
        dropout=0.1,
        train_epochs=300,
        train_batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,

        # Stats toggles
        use_gradient=True,
        use_filter=True
    ):
        """
        Full pipeline:
          train GAN -> generate knockoffs -> train DeepLINK -> compute W -> select
        Returns:
          dict with X_knockoffs, W (gradient/filter), selected (gradient/filter)
        """
        self.fit_generator(X_processed, n_epochs=gan_epochs, batch_size=gan_batch_size, verbose=gan_verbose)
        X_knockoffs = self.generate_knockoffs(X_processed, batch_size=gan_sample_batch_size)

        criterion = self.train_deeplink(
            X_processed, X_knockoffs, y,
            hidden_dims=hidden_dims, dropout=dropout,
            epochs=train_epochs, batch_size=train_batch_size,
            lr=lr, weight_decay=weight_decay
        )

        stats = self.compute_statistics(
            X_processed, X_knockoffs, y, criterion,
            use_gradient=use_gradient, use_filter=use_filter
        )

        out = {"X_knockoffs": X_knockoffs, "W": {}, "selected": {}}

        if use_gradient:
            out["W"]["gradient"] = stats["gradient"]
            out["selected"]["gradient"] = self.select(stats["gradient"], fdr_threshold=fdr_threshold)

        if use_filter:
            out["W"]["filter"] = stats["filter"]
            out["selected"]["filter"] = self.select(stats["filter"], fdr_threshold=fdr_threshold)

        return out

# ============= Simulation Functions for TPM Data =============

def generate_tpm_rnaseq_data(n_samples=1000, p_genes=50, s_causal=5,
                             signal_strength=2.0, noise_level=1.0,
                             outcome_type='tanh'):
    """
    Synthetic data generator with HARD-X scenarios intended to favor diffusion knockoffs.

    NEW outcome_type options (recommended):
      - 'mix_manifold'    : multimodal mixture of nonlinear manifolds (mode structure)
      - 'tail_dependence' : heavy tails + tail dependence (Gaussian fails hard)
      - 'spike_slab'      : spike-and-slab discrete-continuous mixture (AE residual breaks spikes)
      - 'nonlinear_dag'   : nonlinear DAG propagation (conditional structure)
      - 'switching_ar'    : regime-switching latent AR with jumps (nonstationary/multimodal)

    Optional:
      - 'phase_coupled'   : multiscale frequency + phase coupling
      - 'hetero_mixture'  : heteroskedastic mixture-of-Gaussians (conditional variance shifts)

    Backward compatibility:
      If outcome_type is not recognized, falls back to a generic non-Gaussian mixture.
    """
    import numpy as np
    rng = np.random.default_rng()

    # ---- choose causal features ----
    causal_genes = rng.choice(p_genes, size=s_causal, replace=False)
    beta = np.zeros(p_genes)
    beta[causal_genes] = rng.normal(0.0, signal_strength, size=s_causal)

    def _zscore(X):
        return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    # =========================
    # HARD-X scenarios
    # =========================
    if outcome_type == "mix_manifold":
        # Multimodal, nonlinear manifolds (GAN mode drop; Gaussian mismatch; AE residual breaks)
        z = rng.normal(size=(n_samples, 3))
        comp = rng.integers(0, 3, size=n_samples)

        X = np.zeros((n_samples, p_genes))
        W0 = rng.normal(size=(3, p_genes))
        W1 = rng.normal(size=(3, p_genes))
        W2 = rng.normal(size=(3, p_genes))

        idx = comp == 0
        X[idx] = np.tanh(z[idx]) @ W0 + 0.15 * rng.standard_t(df=3, size=(idx.sum(), p_genes))

        idx = comp == 1
        X[idx] = (np.sign(z[idx]) * np.sqrt(np.abs(z[idx]) + 1e-8)) @ W1 + 0.15 * rng.standard_t(df=3, size=(idx.sum(), p_genes))

        idx = comp == 2
        X[idx] = np.sin(2 * np.pi * z[idx]) @ W2 + 0.15 * rng.standard_t(df=3, size=(idx.sum(), p_genes))

        X = _zscore(X)

    elif outcome_type == "tail_dependence":
        # Heavy tails + shared factor => tail dependence (Gaussian knockoffs should break)
        shared = rng.standard_t(df=2.5, size=(n_samples, 1))
        indiv = rng.standard_t(df=2.5, size=(n_samples, p_genes))
        X = 0.85 * shared + 0.55 * indiv
        X = np.tanh(X) + 0.2 * X  # non-elliptical
        X = _zscore(X)

    elif outcome_type == "spike_slab":
        # Discrete-continuous mixture with many exact zeros (hard for AE+Gaussian noise)
        slab = np.exp(rng.normal(0, 1.0, size=(n_samples, p_genes)))  # positive slab
        spike = (rng.random((n_samples, p_genes)) < 0.45)             # many zeros
        X = slab
        X[spike] = 0.0
        # mild global coupling and noise
        X += 0.05 * rng.standard_t(df=3, size=X.shape)
        X = _zscore(X)

    elif outcome_type == "nonlinear_dag":
        # Nonlinear propagation through a sparse DAG (conditional structure)
        A = rng.uniform(0, 1, size=(p_genes, p_genes))
        A = np.tril(A, k=-1)
        A[A < 0.88] = 0.0  # sparse DAG

        X = rng.normal(size=(n_samples, p_genes))
        for _ in range(4):
            X = np.tanh(X @ A.T) + 0.15 * rng.standard_t(df=4, size=(n_samples, p_genes))
        X = _zscore(X)

    elif outcome_type == "switching_ar":
        # Regime switching AR with jumps (multimodal + nonstationary)
        X = np.zeros((n_samples, p_genes))
        state = rng.normal(size=p_genes)

        for i in range(n_samples):
            state = 0.75 * state + 0.25 * np.tanh(state) + rng.normal(0, 0.4, size=p_genes)
            if rng.random() < 0.08:
                state += rng.normal(0, 3.0, size=p_genes)  # jump
            X[i] = state

        X = _zscore(X)

    elif outcome_type == "phase_coupled":
        # Multiscale frequency with nonlinear phase coupling
        t = rng.uniform(0, 1, size=(n_samples, 1))
        freqs = np.array([1.0, 3.0, 7.0, 15.0])
        phases = rng.uniform(0, 2*np.pi, size=(n_samples, len(freqs)))

        base = np.zeros((n_samples, 1))
        for i, f in enumerate(freqs):
            base += np.sin(2*np.pi*f*t + phases[:, i:i+1])

        base2 = np.cos(2*np.pi*5*t + 0.6 * base)  # coupling
        W = rng.normal(size=(2, p_genes))
        lat = np.concatenate([base, base2], axis=1)
        X = lat @ W + 0.35 * rng.normal(size=(n_samples, p_genes))
        X = _zscore(X)

    elif outcome_type == "hetero_mixture":
        # Conditional heteroskedastic mixture (AE residual breaks, Gaussian mismatch)
        g = rng.normal(size=(n_samples, 1))
        gate = (g[:, 0] > np.quantile(g[:, 0], 0.55))

        X = rng.normal(size=(n_samples, p_genes))
        X[~gate] = 0.7 * X[~gate] + rng.normal(0, 0.3, size=(np.sum(~gate), p_genes))
        X[gate] = 1.6 * X[gate] + rng.normal(0, 1.2, size=(np.sum(gate), p_genes)) + 1.5

        X = np.sign(X) * np.log1p(np.abs(X))
        X = _zscore(X)

    else:
        # Generic non-Gaussian fallback (still harder than plain Gaussian)
        X = rng.normal(size=(n_samples, p_genes))
        mix = rng.random(n_samples) < 0.35
        X[mix] = 1.3 * X[mix] + rng.normal(0, 1.0, size=(mix.sum(), p_genes)) + 0.9
        X = np.sign(X) * np.sqrt(np.abs(X) + 1e-8)
        X = _zscore(X)

    # =========================
    # y generation (keep simple/identifiable)
    # =========================
    y_base = X @ beta
    if s_causal >= 2:
        y_base = y_base + 0.25 * X[:, causal_genes[0]] * X[:, causal_genes[1]]

    y = y_base + noise_level * rng.standard_t(df=4, size=n_samples)
    y = (y - y.mean()) / (y.std() + 1e-8)

    return X, y, beta, causal_genes, outcome_type

def evaluate_selection_performance(selected_features, true_causal_features, p_features):
    """Evaluate feature selection performance"""
    selected_set = set(selected_features)
    true_set = set(true_causal_features)
    
    tp = len(selected_set & true_set)
    fp = len(selected_set - true_set)
    fn = len(true_set - selected_set)
    tn = p_features - tp - fp - fn
    
    power = tp / len(true_set) if len(true_set) > 0 else 0
    fdr = fp / len(selected_set) if len(selected_set) > 0 else 0
    
    return {
        'power': power,
        'fdr': fdr,
        'n_selected': len(selected_set),
        'true_positives': tp,
        'false_positives': fp
    }

def preprocess_tpm_data(X_tpm):
    """Preprocess TPM data using log transformation and quantile normalization."""
    X_log = np.log1p(X_tpm)
    scaler = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=min(1000, X_tpm.shape[0])
    )
    X_processed = scaler.fit_transform(X_log)
    return X_processed

# ============= Main Execution =============

def knockoff_quality_check():
    """Check knockoff quality for 4 methods and save plots + diagnostics."""
    print("=== Knockoff Quality Check: Diffusion vs Gaussian vs GAN vs Autoencoder ===")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # -------------------------
    # Generate TPM RNA-seq data
    # -------------------------
    n_samples = 1000
    p_genes = 50
    s_causal = 5

    X_tpm, y, beta_true, causal_genes, _ = generate_tpm_rnaseq_data(
        n_samples=n_samples,
        p_genes=p_genes,
        s_causal=s_causal,
        signal_strength=1.5,
        noise_level=1.0
    )

    print(f"\nData generated:")
    print(f"  Samples: {n_samples}")
    print(f"  Features (genes): {p_genes}")
    print(f"  Causal genes: {causal_genes}")
    print(f"  TPM range: [{X_tpm.min():.2f}, {X_tpm.max():.2f}]")
    print(f"  Log-TPM mean: {np.mean(np.log1p(X_tpm)):.2f}")

    # -------------------------
    # Preprocess ONCE for fairness
    # -------------------------
    # Use your existing preprocessing function if present; otherwise use DiffKnock's method.
    X_processed = X_tpm

    print(f"  Processed data range: [{X_processed.min():.2f}, {X_processed.max():.2f}]")

    # -------------------------
    # Helper: plot per-method quality (same layout as your diffusion plot)
    # -------------------------
    def _plot_quality(method_name, X_ref, X_knock):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Expression distributions (first 3 genes)
        for i in range(3):
            axes[0, i].hist(X_ref[:, i], bins=50, alpha=0.5, density=True, label='Original')
            axes[0, i].hist(X_knock[:, i], bins=50, alpha=0.5, density=True, label='Knockoff')
            axes[0, i].set_title(f'{method_name}: Gene {i} Distribution')
            axes[0, i].set_xlabel('Normalized Expression')
            axes[0, i].legend()

        # Correlation matrices
        corr_orig = np.corrcoef(X_ref.T)
        corr_knock = np.corrcoef(X_knock.T)

        im1 = axes[1, 0].imshow(corr_orig, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Original Correlation')
        plt.colorbar(im1, ax=axes[1, 0])

        im2 = axes[1, 1].imshow(corr_knock, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Knockoff Correlation')
        plt.colorbar(im2, ax=axes[1, 1])

        im3 = axes[1, 2].imshow(np.abs(corr_orig - corr_knock), cmap='Reds', vmin=0, vmax=1)
        axes[1, 2].set_title('|Corr diff|')
        plt.colorbar(im3, ax=axes[1, 2])

        plt.tight_layout()
        outpath = os.path.join(RESULT_DIR, f'knockoff_quality_{method_name.lower().replace(" ", "_")}.pdf')
        plt.savefig(outpath)
        plt.close()
        return outpath

    # -------------------------
    # Helper: summary comparison heatmap (corr diff magnitude)
    # -------------------------
    def _plot_summary_corrdiff(method_to_knock):
        # Compute scalar summaries: mean abs corr diff
        corr_ref = np.corrcoef(X_processed.T)
        rows = []
        for name, Xk in method_to_knock.items():
            corr_k = np.corrcoef(Xk.T)
            mean_abs_diff = float(np.mean(np.abs(corr_ref - corr_k)))
            rows.append((name, mean_abs_diff))

        # Bar plot summary
        plt.figure(figsize=(10, 5))
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        plt.bar(names, vals)
        plt.ylabel('Mean |Corr(X) - Corr(X̃)|')
        plt.title('Knockoff Dependence Preservation Summary')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        outpath = os.path.join(RESULT_DIR, 'knockoff_quality_summary_corrdiff.pdf')
        plt.savefig(outpath)
        plt.close()
        return outpath

    # -------------------------
    # Generate knockoffs for 4 methods on SAME X_processed
    # -------------------------
    method_knockoffs = {}

    # (1) Diffusion
    print("\n--- Generating Diffusion knockoffs ---")
    diff = DiffKnock(input_dim=p_genes)
    # Train diffusion on processed data
    diff_losses = diff.train_diffusion_for_knockoffs(X_processed, n_epochs=300, batch_size=64)
    # Save diffusion loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(diff_losses)
    plt.title('Diffusion Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, 'diffusion_training_loss.pdf'))
    plt.close()

    Xk_diff = diff.generate_knockoffs(X_processed)
    method_knockoffs["Diffusion"] = Xk_diff

    # (2) Gaussian Model-X (fit on processed data)
    print("\n--- Generating Gaussian knockoffs ---")
    gx = Gaussian()
    gx.fit(X_processed)
    Xk_gauss = gx.generate_knockoffs(X_processed)
    method_knockoffs["Gaussian"] = Xk_gauss

    # (3) GAN knockoffs (train on processed data)
    print("\n--- Generating GAN knockoffs ---")
    gan = GAN(input_dim=p_genes, device=('cuda' if torch.cuda.is_available() else 'cpu'))
    gan.train_gan_for_knockoffs(X_processed, n_epochs=300, batch_size=64, verbose=True)
    Xk_gan = gan.generate_knockoffs(X_processed)
    method_knockoffs["GAN"] = Xk_gan

    # (4) Autoencoder knockoffs (train AE on processed data)
    print("\n--- Generating Autoencoder knockoffs ---")
    ae = AutoencoderXKnockoff(input_dim=p_genes, r_factors=1, device=('cuda' if torch.cuda.is_available() else 'cpu'))
    Xk_ae = ae.knockoff_construct_autoencoder(X_processed, epochs=300, lr=1e-3, verbose=True)
    method_knockoffs["Autoencoder"] = Xk_ae

    # -------------------------
    # Plot quality + run exchangeability diagnostics for each method
    # -------------------------
    diag_results = {}

    for method_name, Xk in method_knockoffs.items():
        print(f"\n=== Quality plots & diagnostics: {method_name} ===")

        # 1) Quality plot (distributions + correlations)
        plot_path = _plot_quality(method_name, X_processed, Xk)
        print(f"Saved quality plot: {plot_path}")

        # 2) Exchangeability diagnostics (KS + correlation prints)
        # NOTE: analyze_knockoff_exchangeability saves 'knockoff_exchangeability.pdf' each time;
        #       we temporarily rename by saving then moving, so each method is preserved.
        ks_stats = analyze_knockoff_exchangeability(X_processed, Xk)

        # Rename the output file produced by analyze_knockoff_exchangeability
        src = os.path.join(RESULT_DIR, 'knockoff_exchangeability.pdf')
        dst = os.path.join(RESULT_DIR, f'knockoff_exchangeability_{method_name.lower()}.pdf')
        if os.path.exists(src):
            try:
                os.replace(src, dst)
                print(f"Saved exchangeability plot: {dst}")
            except Exception:
                # If replace fails, just leave it; user can manually inspect
                print(f"WARNING: Could not rename {src} to {dst}")

        diag_results[method_name] = {
            "ks_mean": float(np.mean(ks_stats)),
            "ks_max": float(np.max(ks_stats)),
        }
        print(f"{method_name} KS mean={diag_results[method_name]['ks_mean']:.4f}, max={diag_results[method_name]['ks_max']:.4f}")

    # Summary plot across methods
    summary_path = _plot_summary_corrdiff(method_knockoffs)
    print(f"\nSaved summary comparison plot: {summary_path}")

    # Optional: diffusion process visualization (diffusion only)
    # Uses your existing function; keep it for diffusion-specific inspection.
    try:
        visualize_diffusion_process(diff, X_processed)
        print(f"Saved diffusion process plot: {os.path.join(RESULT_DIR, 'diffusion_process.pdf')}")
    except Exception as e:
        print(f"WARNING: visualize_diffusion_process failed: {e}")

    return method_knockoffs, diag_results

def analyze_knockoff_exchangeability(X, X_knockoffs):
    """Analyze the exchangeability property of knockoffs"""
    print("\n=== Analyzing knockoff exchangeability ===")
    
    n_features = X.shape[1]
    
    # Test marginal distributions
    ks_statistics = []
    for j in range(n_features):
        ks_stat, p_value = stats.ks_2samp(X[:, j], X_knockoffs[:, j])
        ks_statistics.append(ks_stat)
    
    # Plot KS statistics
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_features), ks_statistics)
    plt.xlabel('Gene Index')
    plt.ylabel('KS Statistic')
    plt.title('Kolmogorov-Smirnov Test: Original vs Knockoff Marginals')
    plt.axhline(y=np.mean(ks_statistics), color='red', linestyle='--', 
               label=f'Mean: {np.mean(ks_statistics):.3f}')
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, 'knockoff_exchangeability.pdf'))
    plt.close()
    
    # Check correlation preservation
    corr_orig = np.corrcoef(X.T)
    corr_knock = np.corrcoef(X_knockoffs.T)
    corr_cross = np.corrcoef(X.T, X_knockoffs.T)[:n_features, n_features:]
    
    print(f"Mean absolute correlation difference: {np.mean(np.abs(corr_orig - corr_knock)):.4f}")
    print(f"Mean cross-correlation: {np.mean(np.abs(corr_cross)):.4f}")
    
    return ks_statistics

def visualize_diffusion_process(deeplink, X_sample):
    """Visualize the diffusion process for knockoff generation"""
    print("\n=== Visualizing diffusion process ===")
    
    # Generate with intermediates
    with torch.no_grad():
        shape = (100, 1, X_sample.shape[1])
        samples, intermediates = deeplink.diffusion.sample(shape, return_intermediates=True)
    
    # Select features to visualize
    feature_indices = [0, 1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Show denoising at different timesteps
    timesteps = [0, len(intermediates)//3, 2*len(intermediates)//3]
    titles = ['Pure Noise (t=1000)', 'Early Denoising (t≈666)', 'Late Denoising (t≈333)']
    
    for idx, feature_idx in enumerate(feature_indices):
        for j, (t_idx, title) in enumerate(zip(timesteps, titles)):
            if t_idx < len(intermediates):
                data = intermediates[t_idx].cpu().numpy().squeeze()
                axes[idx, j].hist(data[:, feature_idx], bins=30, alpha=0.7, density=True)
                axes[idx, j].set_title(f'{title}\nGene {feature_idx}')
                axes[idx, j].set_xlim(-4, 4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'diffusion_process.pdf'))
    plt.close()

# ============= Comparison Function =============
def run_multiple_simulations_comparison(n_simulations=50):
    """
    Compare 4 knockoff generators (Diffusion / Gaussian / GAN / Autoencoder)
    with 2 statistics each (Gradient / Filter) across different outcome types,
    and save power/FDR curves + txt outputs as before.
    """
    print("\n=== Comparing 4 Knockoff Generators x 2 Statistics Across Outcome Types ===")

    outcome_scenarios = ["mix_manifold","tail_dependence","spike_slab","nonlinear_dag","switching_ar"]
    signal_amplitudes = np.linspace(0.5, 8, 10)

    # Metadata
    metadata_file = os.path.join(RESULT_DIR, 'simulation_metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write(f"n_simulations: {n_simulations}\n")
        f.write(f"n_samples: 1000\n")
        f.write(f"p_genes: 50\n")
        f.write(f"s_causal: 5\n")
        f.write(f"fdr_threshold: 0.2\n")
        f.write(f"signal_amplitudes: {signal_amplitudes.tolist()}\n")
        f.write("methods: diffusion_gradient diffusion_filter gaussian_gradient gaussian_filter "
                "gan_gradient gan_filter autoencoder_gradient autoencoder_filter\n")

    all_results = {}

    # Fixed parameters
    n_samples = 1000
    p_genes = 50
    s_causal = 5
    fdr_threshold = 0.2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Helper: compute FDP/TDP
    def _fdp_tdp(selected, causal):
        if len(selected) > 0:
            fdp = len(set(selected) - set(causal)) / len(selected)
        else:
            fdp = 0.0
        tdp = len(set(selected) & set(causal)) / s_causal
        return fdp, tdp

    # Helper: train DeepLINK once (black box) for a given (X, Xk, y)
    def _train_deeplink_blackbox(X_proc, Xk, y_vec, epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-4):
        model = DeepLINK(input_dim=p_genes, hidden_dims=[50, 20]).to(device)
        dataset = TensorDataset(
            torch.FloatTensor(X_proc),
            torch.FloatTensor(Xk),
            torch.FloatTensor(y_vec)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model.train()
        for _ in range(epochs):
            for Xb, Xkb, yb in dataloader:
                Xb = Xb.to(device)
                Xkb = Xkb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                out = model(Xb, Xkb)
                loss = criterion(out.squeeze(), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        return model, criterion

    # Loop scenarios
    for outcome_type in outcome_scenarios:
        print(f"\n{'='*60}")
        print(f"Testing Outcome Type: {outcome_type.upper()}")
        print(f"{'='*60}")

        # Storage for results for this outcome type
        methods = [
            'diffusion_gradient', 'diffusion_filter',
            'gaussian_gradient', 'gaussian_filter',
            'gan_gradient', 'gan_filter',
            'autoencoder_gradient', 'autoencoder_filter'
        ]
        results = {m: {'power': [], 'fdr': []} for m in methods}

        for amp in tqdm(signal_amplitudes, desc=f"Testing {outcome_type}"):
            amp_results = {m: {'power': [], 'fdr': []} for m in methods}

            for sim in range(n_simulations):
                # Generate data
                X_tpm, y, beta_true, causal_genes, _ = generate_tpm_rnaseq_data(
                    n_samples=n_samples,
                    p_genes=p_genes,
                    s_causal=s_causal,
                    signal_strength=amp,
                    noise_level=1.0,
                    outcome_type=outcome_type
                )

                # Preprocess ONCE for fairness
                X_processed = X_tpm

                # -----------------------------
                # (1) Diffusion knockoffs
                # -----------------------------
                diff_util = DiffKnock(input_dim=p_genes, device=device)
                diff_util.train_diffusion_for_knockoffs(X_processed, n_epochs=300, batch_size=64)
                Xk_diff = diff_util.generate_knockoffs(X_processed)

                deeplink_diff, crit_diff = _train_deeplink_blackbox(X_processed, Xk_diff, y, epochs=300)
                W_diff_grad = diff_util.compute_gradient_based_statistics(X_processed, Xk_diff, y, deeplink_diff, crit_diff)
                W_diff_filt = diff_util.compute_deeplink_statistics(deeplink_diff)

                sel_diff_grad = diff_util.select_features(W_diff_grad, fdr_threshold=fdr_threshold)
                sel_diff_filt = diff_util.select_features(W_diff_filt, fdr_threshold=fdr_threshold)

                fdp, tdp = _fdp_tdp(sel_diff_grad, causal_genes)
                amp_results['diffusion_gradient']['fdr'].append(fdp)
                amp_results['diffusion_gradient']['power'].append(tdp)

                fdp, tdp = _fdp_tdp(sel_diff_filt, causal_genes)
                amp_results['diffusion_filter']['fdr'].append(fdp)
                amp_results['diffusion_filter']['power'].append(tdp)

                # -----------------------------
                # (2) Gaussian knockoffs
                # -----------------------------
                gx = Gaussian(ridge=1e-1)
                gx.fit(X_processed)
                Xk_g = gx.generate_knockoffs(X_processed)

                deeplink_g, crit_g = _train_deeplink_blackbox(X_processed, Xk_g, y, epochs=300)
                W_g_grad = diff_util.compute_gradient_based_statistics(X_processed, Xk_g, y, deeplink_g, crit_g)
                W_g_filt = diff_util.compute_deeplink_statistics(deeplink_g)

                sel_g_grad = diff_util.select_features(W_g_grad, fdr_threshold=fdr_threshold)
                sel_g_filt = diff_util.select_features(W_g_filt, fdr_threshold=fdr_threshold)

                fdp, tdp = _fdp_tdp(sel_g_grad, causal_genes)
                amp_results['gaussian_gradient']['fdr'].append(fdp)
                amp_results['gaussian_gradient']['power'].append(tdp)

                fdp, tdp = _fdp_tdp(sel_g_filt, causal_genes)
                amp_results['gaussian_filter']['fdr'].append(fdp)
                amp_results['gaussian_filter']['power'].append(tdp)

                # -----------------------------
                # (3) GAN knockoffs
                # -----------------------------
                gan_gen = GAN(input_dim=p_genes, device=device, n_critic=1, lambda_mom=0.5)
                gan_gen.train_gan_for_knockoffs(X_processed, n_epochs=100, batch_size=64, verbose=False)
                Xk_gan = gan_gen.generate_knockoffs(X_processed)

                deeplink_gan, crit_gan = _train_deeplink_blackbox(X_processed, Xk_gan, y, epochs=300)
                W_gan_grad = diff_util.compute_gradient_based_statistics(X_processed, Xk_gan, y, deeplink_gan, crit_gan)
                W_gan_filt = diff_util.compute_deeplink_statistics(deeplink_gan)

                sel_gan_grad = diff_util.select_features(W_gan_grad, fdr_threshold=fdr_threshold)
                sel_gan_filt = diff_util.select_features(W_gan_filt, fdr_threshold=fdr_threshold)

                fdp, tdp = _fdp_tdp(sel_gan_grad, causal_genes)
                amp_results['gan_gradient']['fdr'].append(fdp)
                amp_results['gan_gradient']['power'].append(tdp)

                fdp, tdp = _fdp_tdp(sel_gan_filt, causal_genes)
                amp_results['gan_filter']['fdr'].append(fdp)
                amp_results['gan_filter']['power'].append(tdp)

                # -----------------------------
                # (4) Autoencoder knockoffs
                # -----------------------------
                ae = AutoencoderXKnockoff(input_dim=p_genes, r_factors=1, device=device)
                Xk_ae = ae.knockoff_construct_autoencoder(X_processed, activation='relu', epochs=100, verbose=False)

                # Train DeepLINK once inside AE (cached) and reuse for both stats
                ae.train_deeplink(X_processed, Xk_ae, y, epochs=100, lr=0.001, batch_size=64,
                                  verbose=False, hidden_dims=[50, 20], dropout=0.1)

                W_ae_filt = ae.compute_knockoff_stats_filter(X_processed, Xk_ae, y, epochs=100, lr=0.001,
                                                             batch_size=64, verbose=False, hidden_dims=[50, 20], dropout=0.1)
                W_ae_grad = ae.compute_knockoff_stats_gradient(X_processed, Xk_ae, y, epochs=100, lr=0.001,
                                                               batch_size=64, verbose=False, hidden_dims=[50, 20], dropout=0.1)

                sel_ae_grad = ae.select_features(W_ae_grad, fdr_threshold=fdr_threshold)
                sel_ae_filt = ae.select_features(W_ae_filt, fdr_threshold=fdr_threshold)

                fdp, tdp = _fdp_tdp(sel_ae_grad, causal_genes)
                amp_results['autoencoder_gradient']['fdr'].append(fdp)
                amp_results['autoencoder_gradient']['power'].append(tdp)

                fdp, tdp = _fdp_tdp(sel_ae_filt, causal_genes)
                amp_results['autoencoder_filter']['fdr'].append(fdp)
                amp_results['autoencoder_filter']['power'].append(tdp)

            # Average across simulations for this amplitude
            for m in methods:
                results[m]['fdr'].append(float(np.mean(amp_results[m]['fdr'])))
                results[m]['power'].append(float(np.mean(amp_results[m]['power'])))

        # Store results
        all_results[outcome_type] = results

        # Save power results
        power_file = os.path.join(RESULT_DIR, f'{outcome_type}_power.txt')
        with open(power_file, 'w') as f:
            f.write("# amp " + " ".join(methods) + "\n")
            for i, amp in enumerate(signal_amplitudes):
                row = [f"{amp:.4f}"] + [f"{results[m]['power'][i]:.6f}" for m in methods]
                f.write(" ".join(row) + "\n")

        # Save FDR results
        fdr_file = os.path.join(RESULT_DIR, f'{outcome_type}_fdr.txt')
        with open(fdr_file, 'w') as f:
            f.write("# amp " + " ".join(methods) + "\n")
            for i, amp in enumerate(signal_amplitudes):
                row = [f"{amp:.4f}"] + [f"{results[m]['fdr'][i]:.6f}" for m in methods]
                f.write(" ".join(row) + "\n")

        print(f"  Results saved to {power_file} and {fdr_file}")

        # -------- Plotting (same style, more curves) --------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Power
        for m in methods:
            ax1.plot(signal_amplitudes, results[m]['power'], linewidth=2, marker='o', markersize=3, label=m)
        ax1.set_xlabel('Signal Amplitude', fontsize=12)
        ax1.set_ylabel('Power+', fontsize=12)
        ax1.set_title('Power vs Signal Amplitude', fontsize=14)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        # FDR
        for m in methods:
            ax2.plot(signal_amplitudes, results[m]['fdr'], linewidth=2, marker='o', markersize=3, label=m)
        ax2.axhline(y=fdr_threshold, color='black', linestyle='--', linewidth=1.5, label=f'Target FDR = {fdr_threshold}')
        ax2.set_xlabel('Signal Amplitude', fontsize=12)
        ax2.set_ylabel('FDR+', fontsize=12)
        ax2.set_title('FDR vs Signal Amplitude', fontsize=14)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 0.3)

        plt.suptitle(
            f'Outcome Type: {outcome_type.upper()}\n'
            f'(n={n_samples}, p={p_genes}, s={s_causal}, {n_simulations} simulations per point)',
            fontsize=14, y=1.02
        )
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f'comparison_{outcome_type}.pdf'), dpi=300, bbox_inches='tight')
        plt.show()

        # Summary print
        print(f"\n=== Summary for {outcome_type.upper()} ===")
        print(f"At largest amplitude (A={signal_amplitudes[-1]:.1f}):")
        for m in methods:
            print(f"  {m}: FDR={results[m]['fdr'][-1]:.3f}, Power={results[m]['power'][-1]:.3f}")

    return all_results

if __name__ == "__main__":
    # 1) Knockoff quality check (all 4 generators)
    method_knockoffs, diag_results = knockoff_quality_check()

    # 2) Run comparison across nonlinear outcome scenarios
    print("="*70)
    print("DEEPLINK METHODS COMPARISON WITH MULTIPLE OUTCOME SCENARIOS")
    print("="*70)
    print("\nComparing four knockoff generators with two statistics each:")
    print("Generators: Diffusion, Gaussian, GAN, Autoencoder")
    print("Statistics: Gradient-based, Filter-based")
    print("\nOutcome scenarios to test:")
    print("- information_bottleneck")
    print("- multiscale_frequency")
    print("- network_propagation")
    print("- temporal_memory")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nPyTorch device: {device}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: Running on CPU - this will be significantly slower!")

    all_results = run_multiple_simulations_comparison(n_simulations=50)

    # 3) Final summary (updated for 8 method-curves)
    print("\n" + "="*70)
    print("FINAL COMPREHENSIVE SUMMARY")
    print("="*70)

    outcome_scenarios = ["mix_manifold","tail_dependence","spike_slab","nonlinear_dag","switching_ar"]
    methods = [
        'diffusion_gradient','diffusion_filter',
        'gaussian_gradient','gaussian_filter',
        'gan_gradient','gan_filter',
        'autoencoder_gradient','autoencoder_filter'
    ]

    print("\n=== Power at Maximum Signal Amplitude ===")
    print(f"{'Outcome Type':<22} {'Best Method':<22} {'Power':<8} {'FDR':<8}")
    print("-"*70)

    for outcome_type in outcome_scenarios:
        res = all_results[outcome_type]
        # find best by power at max amp
        best_method = max(methods, key=lambda m: res[m]['power'][-1])
        best_power = res[best_method]['power'][-1]
        best_fdr = res[best_method]['fdr'][-1]
        print(f"{outcome_type:<22} {best_method:<22} {best_power:<8.3f} {best_fdr:<8.3f}")

    print("\n=== FDR at Maximum Signal Amplitude (all methods) ===")
    for outcome_type in outcome_scenarios:
        res = all_results[outcome_type]
        print(f"\n{outcome_type.upper()}")
        for m in methods:
            print(f"  {m:<22} FDR={res[m]['fdr'][-1]:.3f}  Power={res[m]['power'][-1]:.3f}")

    print("\n" + "="*70)
    print("All analyses complete!")
    print(f"Results saved to: {RESULT_DIR}")
    print("="*70)
