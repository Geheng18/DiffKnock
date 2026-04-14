import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import os
import torch

from models import DeepLINK
from knockoff_generators import DiffKnock, Gaussian, GAN, AutoencoderXKnockoff
from data_generation import generate_tpm_rnaseq_data

matplotlib.rcParams['savefig.format'] = 'pdf'

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

# Unified training configuration
GENERATOR_EPOCHS = 300
DEEPLINK_EPOCHS = 300


def analyze_knockoff_exchangeability(X, X_knockoffs, result_dir=None):
    """Analyze the exchangeability property of knockoffs."""
    print("\n=== Analyzing knockoff exchangeability ===")
    save_dir = result_dir or RESULT_DIR

    n_features = X.shape[1]

    ks_statistics = []
    for j in range(n_features):
        ks_stat, p_value = stats.ks_2samp(X[:, j], X_knockoffs[:, j])
        ks_statistics.append(ks_stat)

    plt.figure(figsize=(10, 6))
    plt.bar(range(n_features), ks_statistics)
    plt.xlabel('Gene Index')
    plt.ylabel('KS Statistic')
    plt.title('Kolmogorov-Smirnov Test: Original vs Knockoff Marginals')
    plt.axhline(y=np.mean(ks_statistics), color='red', linestyle='--',
                label=f'Mean: {np.mean(ks_statistics):.3f}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'knockoff_exchangeability.pdf'))
    plt.close()

    corr_orig = np.corrcoef(X.T)
    corr_knock = np.corrcoef(X_knockoffs.T)
    corr_cross = np.corrcoef(X.T, X_knockoffs.T)[:n_features, n_features:]

    print(f"Mean absolute correlation difference: {np.mean(np.abs(corr_orig - corr_knock)):.4f}")
    print(f"Mean cross-correlation: {np.mean(np.abs(corr_cross)):.4f}")

    return ks_statistics


def visualize_diffusion_process(diffknock, X_sample, result_dir=None):
    """Visualize the diffusion process for knockoff generation."""
    print("\n=== Visualizing diffusion process ===")
    save_dir = result_dir or RESULT_DIR

    with torch.no_grad():
        shape = (100, 1, X_sample.shape[1])
        samples, intermediates = diffknock.diffusion.sample(shape, return_intermediates=True)

    feature_indices = [0, 1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    timesteps = [0, len(intermediates)//3, 2*len(intermediates)//3]
    titles = ['Pure Noise (t=1000)', 'Early Denoising (t~666)', 'Late Denoising (t~333)']

    for idx, feature_idx in enumerate(feature_indices):
        for j, (t_idx, title) in enumerate(zip(timesteps, titles)):
            if t_idx < len(intermediates):
                data = intermediates[t_idx].cpu().numpy().squeeze()
                axes[idx, j].hist(data[:, feature_idx], bins=30, alpha=0.7, density=True)
                axes[idx, j].set_title(f'{title}\nGene {feature_idx}')
                axes[idx, j].set_xlim(-4, 4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diffusion_process.pdf'))
    plt.close()


def knockoff_quality_check(result_dir=None):
    """Check knockoff quality for all 4 methods with equal training budgets."""
    save_dir = result_dir or RESULT_DIR
    print("=== Knockoff Quality Check: Diffusion vs Gaussian vs GAN vs Autoencoder ===")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

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

    X_processed = X_tpm

    print(f"  Processed data range: [{X_processed.min():.2f}, {X_processed.max():.2f}]")

    def _plot_quality(method_name, X_ref, X_knock):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for i in range(3):
            axes[0, i].hist(X_ref[:, i], bins=50, alpha=0.5, density=True, label='Original')
            axes[0, i].hist(X_knock[:, i], bins=50, alpha=0.5, density=True, label='Knockoff')
            axes[0, i].set_title(f'{method_name}: Gene {i} Distribution')
            axes[0, i].set_xlabel('Normalized Expression')
            axes[0, i].legend()

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
        outpath = os.path.join(save_dir, f'knockoff_quality_{method_name.lower().replace(" ", "_")}.pdf')
        plt.savefig(outpath)
        plt.close()
        return outpath

    def _plot_summary_corrdiff(method_to_knock):
        corr_ref = np.corrcoef(X_processed.T)
        rows = []
        for name, Xk in method_to_knock.items():
            corr_k = np.corrcoef(Xk.T)
            mean_abs_diff = float(np.mean(np.abs(corr_ref - corr_k)))
            rows.append((name, mean_abs_diff))

        plt.figure(figsize=(10, 5))
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        plt.bar(names, vals)
        plt.ylabel('Mean |Corr(X) - Corr(X~)|')
        plt.title('Knockoff Dependence Preservation Summary')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        outpath = os.path.join(save_dir, 'knockoff_quality_summary_corrdiff.pdf')
        plt.savefig(outpath)
        plt.close()
        return outpath

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    method_knockoffs = {}

    # (1) Diffusion
    print("\n--- Generating Diffusion knockoffs ---")
    diff = DiffKnock(input_dim=p_genes, device=device)
    diff_losses = diff.train_diffusion_for_knockoffs(X_processed, n_epochs=GENERATOR_EPOCHS, batch_size=64)

    plt.figure(figsize=(10, 5))
    plt.plot(diff_losses)
    plt.title('Diffusion Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'diffusion_training_loss.pdf'))
    plt.close()

    Xk_diff = diff.generate_knockoffs(X_processed)
    method_knockoffs["Diffusion"] = Xk_diff

    # (2) Gaussian Model-X
    print("\n--- Generating Gaussian knockoffs ---")
    gx = Gaussian()
    gx.fit(X_processed)
    Xk_gauss = gx.generate_knockoffs(X_processed)
    method_knockoffs["Gaussian"] = Xk_gauss

    # (3) GAN knockoffs
    print("\n--- Generating GAN knockoffs ---")
    gan = GAN(input_dim=p_genes, device=device)
    gan.train_gan_for_knockoffs(X_processed, n_epochs=GENERATOR_EPOCHS, batch_size=64, verbose=True)
    Xk_gan = gan.generate_knockoffs(X_processed)
    method_knockoffs["GAN"] = Xk_gan

    # (4) Autoencoder knockoffs
    print("\n--- Generating Autoencoder knockoffs ---")
    ae = AutoencoderXKnockoff(input_dim=p_genes, r_factors=1, device=device)
    Xk_ae = ae.knockoff_construct_autoencoder(X_processed, epochs=GENERATOR_EPOCHS, lr=1e-3, verbose=True)
    method_knockoffs["Autoencoder"] = Xk_ae

    diag_results = {}

    for method_name, Xk in method_knockoffs.items():
        print(f"\n=== Quality plots & diagnostics: {method_name} ===")

        plot_path = _plot_quality(method_name, X_processed, Xk)
        print(f"Saved quality plot: {plot_path}")

        ks_stats = analyze_knockoff_exchangeability(X_processed, Xk, result_dir=save_dir)

        src = os.path.join(save_dir, 'knockoff_exchangeability.pdf')
        dst = os.path.join(save_dir, f'knockoff_exchangeability_{method_name.lower()}.pdf')
        if os.path.exists(src):
            try:
                os.replace(src, dst)
                print(f"Saved exchangeability plot: {dst}")
            except Exception:
                print(f"WARNING: Could not rename {src} to {dst}")

        diag_results[method_name] = {
            "ks_mean": float(np.mean(ks_stats)),
            "ks_max": float(np.max(ks_stats)),
        }
        print(f"{method_name} KS mean={diag_results[method_name]['ks_mean']:.4f}, max={diag_results[method_name]['ks_max']:.4f}")

    summary_path = _plot_summary_corrdiff(method_knockoffs)
    print(f"\nSaved summary comparison plot: {summary_path}")

    try:
        visualize_diffusion_process(diff, X_processed, result_dir=save_dir)
        print(f"Saved diffusion process plot: {os.path.join(save_dir, 'diffusion_process.pdf')}")
    except Exception as e:
        print(f"WARNING: visualize_diffusion_process failed: {e}")

    return method_knockoffs, diag_results


if __name__ == "__main__":
    knockoff_quality_check()
