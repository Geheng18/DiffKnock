import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models import (
    DeepLINK, DiffusionTransformer, ImprovedDDPM,
    Autoencoder, _MLP
)


# ============= Diffusion-based Knockoff Generator =============

class DiffKnock:
    """Diffusion-based knockoff generation."""
    def __init__(self, input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.device = device
        self.diffusion = None
        self.deepLINK = None
        self.scaler = None

    def train_diffusion_for_knockoffs(self, X_data, n_epochs=500, batch_size=64, lr=1e-4):
        print("Training diffusion model for knockoff generation...")

        X_tensor = torch.FloatTensor(X_data).unsqueeze(1)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=torch.cuda.is_available())

        model = DiffusionTransformer(
            input_dim=self.input_dim,
            hidden_dim=256,
            depth=6,
            heads=8,
            dim_head=32,
            dropout=0.1
        )

        self.diffusion = ImprovedDDPM(model, num_timesteps=1000, device=self.device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        losses = []
        model.train()

        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_idx, (batch,) in enumerate(dataloader):
                batch = batch.to(self.device)
                batch_size_actual = batch.shape[0]

                t = torch.randint(0, self.diffusion.num_timesteps,
                                  (batch_size_actual,), device=self.device).long()

                loss = self.diffusion.p_losses(batch, t)

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
            n_samples = X_data.shape[0]
            X_knockoffs = self.diffusion.sample((n_samples, 1, self.input_dim))
            X_knockoffs = X_knockoffs.squeeze(1).cpu().numpy()

        return X_knockoffs

    def compute_gradient_based_statistics(self, X, X_tilde, y, model, criterion, batch_size=100):
        print("Computing gradient-based statistics...")

        n_samples = X.shape[0]
        n_features = X.shape[1]

        grad_X_accum = np.zeros(n_features)
        grad_X_tilde_accum = np.zeros(n_features)

        model.eval()

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            X_batch = torch.FloatTensor(X[i:end_idx]).to(self.device).requires_grad_(True)
            X_tilde_batch = torch.FloatTensor(X_tilde[i:end_idx]).to(self.device).requires_grad_(True)
            y_batch = torch.FloatTensor(y[i:end_idx]).to(self.device)

            output = model(X_batch, X_tilde_batch)
            loss = criterion(output.squeeze(), y_batch)

            grads = torch.autograd.grad(loss, [X_batch, X_tilde_batch], create_graph=False)

            grad_X_accum += torch.abs(grads[0]).sum(dim=0).detach().cpu().numpy()
            grad_X_tilde_accum += torch.abs(grads[1]).sum(dim=0).detach().cpu().numpy()

        G_j = grad_X_accum / n_samples
        G_j_tilde = grad_X_tilde_accum / n_samples

        W_grad = G_j - G_j_tilde
        return W_grad

    def compute_deeplink_statistics(self, model):
        print("Computing filter-based statistics...")

        z, z_tilde = model.get_filter_weights()

        with torch.no_grad():
            linear_layers = []
            for module in model.mlp:
                if isinstance(module, nn.Linear):
                    linear_layers.append(module)

            w = linear_layers[-1].weight.data
            for layer in reversed(linear_layers[:-1]):
                w = w @ layer.weight.data

            w = w.squeeze()

        w = w.cpu().numpy()
        z = z.detach().cpu().numpy()
        z_tilde = z_tilde.detach().cpu().numpy()

        W_filt = (w * z) ** 2 - (w * z_tilde) ** 2
        return W_filt

    def select_features(self, W_j, fdr_threshold=0.2):
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


class DiffusionXKnockoff:
    """End-to-end pipeline: Diffusion knockoffs -> DeepLINK -> W -> selection."""
    def __init__(self, input_dim, device=None):
        self.input_dim = input_dim
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.diff = DiffKnock(input_dim=input_dim, device=self.device)
        self.deepLINK = None

    def run(self, X_processed, y, fdr_threshold=0.2,
            diffusion_epochs=300, diffusion_batch_size=64, diffusion_lr=1e-4,
            hidden_dims=[50, 20], dropout=0.1,
            deeplink_epochs=300, deeplink_batch_size=64,
            deeplink_lr=1e-3, deeplink_weight_decay=1e-4,
            use_gradient=True, use_filter=True):

        self.diff.train_diffusion_for_knockoffs(
            X_processed, n_epochs=diffusion_epochs,
            batch_size=diffusion_batch_size, lr=diffusion_lr
        )
        X_knock = self.diff.generate_knockoffs(X_processed)

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

        out = {"X_processed": X_processed, "X_knockoffs": X_knock, "W": {}, "selected": {}}

        if use_gradient:
            W_grad = self.diff.compute_gradient_based_statistics(X_processed, X_knock, y, self.deepLINK, criterion)
            out["W"]["gradient"] = W_grad
            out["selected"]["gradient"] = self.diff.select_features(W_grad, fdr_threshold=fdr_threshold)

        if use_filter:
            W_filt = self.diff.compute_deeplink_statistics(self.deepLINK)
            out["W"]["filter"] = W_filt
            out["selected"]["filter"] = self.diff.select_features(W_filt, fdr_threshold=fdr_threshold)

        return out


# ============= Gaussian Knockoff Generator =============

class Gaussian:
    """Model-X Gaussian knockoffs (Barber & Candes style)."""
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

        Xc = X - X.mean(axis=0, keepdims=True)

        Sigma = (Xc.T @ Xc) / max(n - 1, 1)
        Sigma = Sigma + self.ridge * np.eye(p)

        Sigma_inv = np.linalg.inv(Sigma)

        if self.s_mode != "equi":
            raise NotImplementedError("Only s_mode='equi' implemented.")

        evals = np.linalg.eigvalsh(Sigma)
        lam_min = float(np.min(evals))
        s_val = min(2.0 * lam_min, 1e-3)
        if s_val <= 0:
            s_val = 1e-3
        s = s_val * np.ones(p)

        D = np.diag(s)
        M = 2 * D - D @ Sigma_inv @ D
        M = 0.5 * (M + M.T) + self.ridge * np.eye(p)

        try:
            C = np.linalg.cholesky(M).T
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(M)
            w = np.clip(w, 0.0, None)
            C = (V * np.sqrt(w)) @ V.T

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
        X_tilde = Xc @ (np.eye(p) - self.Sigma_inv_ @ D) + U @ self.C_
        X_tilde = X_tilde + X.mean(axis=0, keepdims=True)
        return X_tilde


class GaussianXKnockoff:
    """End-to-end pipeline: Gaussian knockoffs -> DeepLINK -> W -> selection."""
    def __init__(self, input_dim, device=None, ridge=1e-1, s_mode="equi"):
        self.input_dim = input_dim
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gx = Gaussian(ridge=ridge, s_mode=s_mode)
        self.deepLINK = None
        self._diff_util = DiffKnock(input_dim=input_dim)
        self._diff_util.device = self.device

    def fit_generator(self, X_processed):
        self.gx.fit(X_processed)
        return self

    def generate_knockoffs(self, X_processed):
        return self.gx.generate_knockoffs(X_processed)

    def train_deeplink(self, X_processed, X_knockoffs, y,
                       hidden_dims=[50, 20], dropout=0.1,
                       epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-4):
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
        return self._diff_util.select_features(W, fdr_threshold=fdr_threshold)

    def run(self, X_processed, y, fdr_threshold=0.2,
            hidden_dims=[50, 20], dropout=0.1,
            train_epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-4,
            use_gradient=True, use_filter=True):

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


# ============= GAN-Based Knockoff Generator =============

class GAN:
    """KnockoffGAN-style knockoff generator."""
    def __init__(self, input_dim, z_dim=64,
                 device=("cuda" if torch.cuda.is_available() else "cpu"),
                 g_hidden=(64, 64), d_hidden=(64, 64), dropout=0.1,
                 lr_g=2e-4, lr_d=2e-4, n_critic=2, lambda_mom=0.5):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.device = device
        self.n_critic = n_critic
        self.lambda_mom = lambda_mom

        self.G = _MLP(input_dim + z_dim, input_dim, hidden_dims=g_hidden, dropout=dropout).to(device)
        self.D = _MLP(2 * input_dim, 1, hidden_dims=d_hidden, dropout=dropout).to(device)

        self.optG = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.optD = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.9))

    def _sample_z(self, n):
        return torch.randn(n, self.z_dim, device=self.device)

    @staticmethod
    def _moment_loss(x, x_tilde):
        x0 = x - x.mean(dim=0, keepdim=True)
        xt0 = x_tilde - x_tilde.mean(dim=0, keepdim=True)

        mean_loss = (x.mean(dim=0) - x_tilde.mean(dim=0)).pow(2).mean()

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

                for _ in range(self.n_critic):
                    self.D.train()
                    self.optD.zero_grad()

                    z = self._sample_z(x_batch.shape[0])
                    x_tilde = self.G(torch.cat([x_batch, z], dim=1)).detach()

                    real_pairs = torch.cat([x_batch, x_tilde], dim=1)
                    swap_pairs = torch.cat([x_tilde, x_batch], dim=1)

                    logits_real = self.D(real_pairs)
                    logits_swap = self.D(swap_pairs)

                    lossD = bce(logits_real, torch.ones_like(logits_real)) + \
                            bce(logits_swap, torch.zeros_like(logits_swap))

                    lossD.backward()
                    self.optD.step()

                self.G.train()
                self.optG.zero_grad()

                z = self._sample_z(x_batch.shape[0])
                x_tilde = self.G(torch.cat([x_batch, z], dim=1))

                real_pairs = torch.cat([x_batch, x_tilde], dim=1)
                logits_real = self.D(real_pairs)

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
    """End-to-end pipeline: GAN knockoffs -> DeepLINK -> W -> selection."""
    def __init__(self, input_dim, device=None, z_dim=64,
                 g_hidden=(256, 256), d_hidden=(256, 256), dropout=0.1,
                 lr_g=2e-4, lr_d=2e-4, n_critic=2, lambda_mom=10.0):
        self.input_dim = input_dim
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.gan = GAN(
            input_dim=input_dim, z_dim=z_dim, device=self.device,
            g_hidden=g_hidden, d_hidden=d_hidden, dropout=dropout,
            lr_g=lr_g, lr_d=lr_d, n_critic=n_critic, lambda_mom=lambda_mom
        )

        self.deepLINK = None
        self._diff_util = DiffKnock(input_dim=input_dim)
        self._diff_util.device = self.device

    def fit_generator(self, X_processed, n_epochs=500, batch_size=128, verbose=True):
        self.gan.train_gan_for_knockoffs(X_processed, n_epochs=n_epochs, batch_size=batch_size, verbose=verbose)
        return self

    def generate_knockoffs(self, X_processed, batch_size=512):
        return self.gan.generate_knockoffs(X_processed, batch_size=batch_size)

    def train_deeplink(self, X_processed, X_knockoffs, y,
                       hidden_dims=[50, 20], dropout=0.1,
                       epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-4):
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
        return self._diff_util.select_features(W, fdr_threshold=fdr_threshold)

    def run(self, X_processed, y, fdr_threshold=0.2,
            gan_epochs=500, gan_batch_size=128, gan_verbose=True, gan_sample_batch_size=512,
            hidden_dims=[50, 20], dropout=0.1,
            train_epochs=300, train_batch_size=64, lr=1e-3, weight_decay=1e-4,
            use_gradient=True, use_filter=True):

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


# ============= Autoencoder-Based Knockoff Generator =============

class AutoencoderXKnockoff:
    """Autoencoder-based knockoff generation."""
    def __init__(self, input_dim, r_factors=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.r_factors = r_factors
        self.device = device
        self.autoencoder = None

    def knockoff_construct_autoencoder(self, X, activation='relu', epochs=500, lr=1e-3, verbose=True):
        n, p = X.shape

        self.autoencoder = Autoencoder(p, self.r_factors, activation).to(self.device)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.autoencoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.autoencoder(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Autoencoder Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        self.autoencoder.eval()
        with torch.no_grad():
            C = self.autoencoder(X_tensor).cpu().numpy()

        E = X - C
        sigma = np.sqrt(np.sum(E ** 2) / (n * p))
        X_knockoff = C + sigma * np.random.randn(n, p)

        return X_knockoff
