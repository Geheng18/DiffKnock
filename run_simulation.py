import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from models import DeepLINK
from knockoff_generators import DiffKnock, Gaussian, GAN, AutoencoderXKnockoff
from data_generation import generate_tpm_rnaseq_data
from diagnostics import knockoff_quality_check

matplotlib.rcParams['savefig.format'] = 'pdf'

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# Unified hyperparameters for fair comparison across all methods
# ============================================================
GENERATOR_EPOCHS = 300
DEEPLINK_EPOCHS = 300
DEEPLINK_BATCH_SIZE = 64
DEEPLINK_LR = 1e-3
DEEPLINK_WEIGHT_DECAY = 1e-4
DEEPLINK_HIDDEN_DIMS = [50, 20]


def run_multiple_simulations_comparison(n_simulations=50):
    """
    Compare 4 knockoff generators (Diffusion / Gaussian / GAN / Autoencoder)
    with 2 statistics each (Gradient / Filter) across diverse outcome types.

    All methods use identical training budgets and the same DeepLINK pipeline.
    """
    print("\n=== Comparing 4 Knockoff Generators x 2 Statistics Across Outcome Types ===")

    outcome_scenarios = [
        "gaussian",
        "linear_factor",
        "mix_manifold",
        "tail_dependence",
        "spike_slab",
        "nonlinear_dag",
        "switching_ar",
    ]
    signal_amplitudes = np.linspace(0.5, 8, 10)

    metadata_file = os.path.join(RESULT_DIR, 'simulation_metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write(f"n_simulations: {n_simulations}\n")
        f.write(f"n_samples: 1000\n")
        f.write(f"p_genes: 50\n")
        f.write(f"s_causal: 5\n")
        f.write(f"fdr_threshold: 0.2\n")
        f.write(f"generator_epochs: {GENERATOR_EPOCHS}\n")
        f.write(f"deeplink_epochs: {DEEPLINK_EPOCHS}\n")
        f.write(f"signal_amplitudes: {signal_amplitudes.tolist()}\n")
        f.write("methods: diffusion_gradient diffusion_filter gaussian_gradient gaussian_filter "
                "gan_gradient gan_filter autoencoder_gradient autoencoder_filter\n")
        f.write(f"outcome_scenarios: {' '.join(outcome_scenarios)}\n")

    all_results = {}

    n_samples = 1000
    p_genes = 50
    s_causal = 5
    fdr_threshold = 0.2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _fdp_tdp(selected, causal):
        if len(selected) > 0:
            fdp = len(set(selected) - set(causal)) / len(selected)
        else:
            fdp = 0.0
        tdp = len(set(selected) & set(causal)) / s_causal
        return fdp, tdp

    def _train_deeplink(X_proc, Xk, y_vec):
        """Shared DeepLINK training — identical for all methods."""
        model = DeepLINK(input_dim=p_genes, hidden_dims=DEEPLINK_HIDDEN_DIMS).to(device)
        dataset = TensorDataset(
            torch.FloatTensor(X_proc),
            torch.FloatTensor(Xk),
            torch.FloatTensor(y_vec)
        )
        dataloader = DataLoader(dataset, batch_size=DEEPLINK_BATCH_SIZE, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=DEEPLINK_LR, weight_decay=DEEPLINK_WEIGHT_DECAY)

        model.train()
        for _ in range(DEEPLINK_EPOCHS):
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

    for outcome_type in outcome_scenarios:
        print(f"\n{'='*60}")
        print(f"Testing Outcome Type: {outcome_type.upper()}")
        print(f"{'='*60}")

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
                X_tpm, y, beta_true, causal_genes, _ = generate_tpm_rnaseq_data(
                    n_samples=n_samples,
                    p_genes=p_genes,
                    s_causal=s_causal,
                    signal_strength=amp,
                    noise_level=1.0,
                    outcome_type=outcome_type
                )

                X_processed = X_tpm

                # --- (1) Diffusion knockoffs ---
                diff_util = DiffKnock(input_dim=p_genes, device=device)
                diff_util.train_diffusion_for_knockoffs(X_processed, n_epochs=GENERATOR_EPOCHS, batch_size=64)
                Xk_diff = diff_util.generate_knockoffs(X_processed)

                deeplink_diff, crit_diff = _train_deeplink(X_processed, Xk_diff, y)
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

                # --- (2) Gaussian knockoffs ---
                gx = Gaussian(ridge=1e-1)
                gx.fit(X_processed)
                Xk_g = gx.generate_knockoffs(X_processed)

                deeplink_g, crit_g = _train_deeplink(X_processed, Xk_g, y)
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

                # --- (3) GAN knockoffs ---
                gan_gen = GAN(input_dim=p_genes, device=device)
                gan_gen.train_gan_for_knockoffs(X_processed, n_epochs=GENERATOR_EPOCHS, batch_size=64, verbose=False)
                Xk_gan = gan_gen.generate_knockoffs(X_processed)

                deeplink_gan, crit_gan = _train_deeplink(X_processed, Xk_gan, y)
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

                # --- (4) Autoencoder knockoffs ---
                ae = AutoencoderXKnockoff(input_dim=p_genes, r_factors=1, device=device)
                Xk_ae = ae.knockoff_construct_autoencoder(X_processed, activation='relu', epochs=GENERATOR_EPOCHS, verbose=False)

                deeplink_ae, crit_ae = _train_deeplink(X_processed, Xk_ae, y)
                W_ae_grad = diff_util.compute_gradient_based_statistics(X_processed, Xk_ae, y, deeplink_ae, crit_ae)
                W_ae_filt = diff_util.compute_deeplink_statistics(deeplink_ae)

                sel_ae_grad = diff_util.select_features(W_ae_grad, fdr_threshold=fdr_threshold)
                sel_ae_filt = diff_util.select_features(W_ae_filt, fdr_threshold=fdr_threshold)

                fdp, tdp = _fdp_tdp(sel_ae_grad, causal_genes)
                amp_results['autoencoder_gradient']['fdr'].append(fdp)
                amp_results['autoencoder_gradient']['power'].append(tdp)

                fdp, tdp = _fdp_tdp(sel_ae_filt, causal_genes)
                amp_results['autoencoder_filter']['fdr'].append(fdp)
                amp_results['autoencoder_filter']['power'].append(tdp)

            for m in methods:
                results[m]['fdr'].append(float(np.mean(amp_results[m]['fdr'])))
                results[m]['power'].append(float(np.mean(amp_results[m]['power'])))

        all_results[outcome_type] = results

        power_file = os.path.join(RESULT_DIR, f'{outcome_type}_power.txt')
        with open(power_file, 'w') as f:
            f.write("# amp " + " ".join(methods) + "\n")
            for i, amp in enumerate(signal_amplitudes):
                row = [f"{amp:.4f}"] + [f"{results[m]['power'][i]:.6f}" for m in methods]
                f.write(" ".join(row) + "\n")

        fdr_file = os.path.join(RESULT_DIR, f'{outcome_type}_fdr.txt')
        with open(fdr_file, 'w') as f:
            f.write("# amp " + " ".join(methods) + "\n")
            for i, amp in enumerate(signal_amplitudes):
                row = [f"{amp:.4f}"] + [f"{results[m]['fdr'][i]:.6f}" for m in methods]
                f.write(" ".join(row) + "\n")

        print(f"  Results saved to {power_file} and {fdr_file}")

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        for m in methods:
            ax1.plot(signal_amplitudes, results[m]['power'], linewidth=2, marker='o', markersize=3, label=m)
        ax1.set_xlabel('Signal Amplitude', fontsize=12)
        ax1.set_ylabel('Power+', fontsize=12)
        ax1.set_title('Power vs Signal Amplitude', fontsize=14)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

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

        print(f"\n=== Summary for {outcome_type.upper()} ===")
        print(f"At largest amplitude (A={signal_amplitudes[-1]:.1f}):")
        for m in methods:
            print(f"  {m}: FDR={results[m]['fdr'][-1]:.3f}, Power={results[m]['power'][-1]:.3f}")

    return all_results


if __name__ == "__main__":
    # 1) Knockoff quality check (all 4 generators, equal training budgets)
    method_knockoffs, diag_results = knockoff_quality_check(result_dir=RESULT_DIR)

    # 2) Run comparison across outcome scenarios
    print("=" * 70)
    print("KNOCKOFF METHODS COMPARISON ACROSS OUTCOME SCENARIOS")
    print("=" * 70)
    print("\nComparing four knockoff generators with two statistics each:")
    print("Generators: Diffusion, Gaussian, GAN, Autoencoder")
    print("Statistics: Gradient-based, Filter-based")
    print(f"\nUnified training budget: generator_epochs={GENERATOR_EPOCHS}, deeplink_epochs={DEEPLINK_EPOCHS}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nPyTorch device: {device}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: Running on CPU - this will be significantly slower!")

    all_results = run_multiple_simulations_comparison(n_simulations=50)

    # 3) Final summary
    print("\n" + "=" * 70)
    print("FINAL COMPREHENSIVE SUMMARY")
    print("=" * 70)

    outcome_scenarios = [
        "gaussian", "linear_factor",
        "mix_manifold", "tail_dependence", "spike_slab",
        "nonlinear_dag", "switching_ar"
    ]
    methods = [
        'diffusion_gradient', 'diffusion_filter',
        'gaussian_gradient', 'gaussian_filter',
        'gan_gradient', 'gan_filter',
        'autoencoder_gradient', 'autoencoder_filter'
    ]

    print("\n=== Power at Maximum Signal Amplitude ===")
    print(f"{'Outcome Type':<22} {'Best Method':<22} {'Power':<8} {'FDR':<8}")
    print("-" * 70)

    for outcome_type in outcome_scenarios:
        res = all_results[outcome_type]
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

    print("\n" + "=" * 70)
    print("All analyses complete!")
    print(f"Results saved to: {RESULT_DIR}")
    print("=" * 70)
