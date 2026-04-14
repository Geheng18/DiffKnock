import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os

from models import DeepLINK, ImprovedDDPM, DiffusionTransformer
from knockoff_generators import DiffKnock

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

RESULT_DIR = 'murine_results'
os.makedirs(RESULT_DIR, exist_ok=True)

# ============= Data Loading Class =============
class MurineDataLoader:
    """Load murine scRNA-seq data and screening results"""
    
    def __init__(self):
        # Load data
        self.data = pd.read_csv('data/murine_rna.csv')
        self.X = self.data.iloc[:, :-1].values  # Genes
        self.y = self.data.iloc[:, -1].values   # Outcome (0/1)
        self.gene_names = self.data.columns[:-1].tolist()  # Store gene names
        
        # Load screening results (convert to 0-indexed)
        self.top_genes = pd.read_csv('data/top500_p50.csv').values - 1
        self.screening_indices = pd.read_csv('data/indmat_dist_p50.csv').values - 1
        
        # Preprocess
        X_log = np.log1p(self.X)
        self.X_processed = (X_log - np.mean(X_log, axis=0)) / (np.std(X_log, axis=0) + 1e-8)
        

    def get_split_for_rep(self, rep_idx, n_genes):
        """Get train/test split for a repetition with top n genes"""
        # Get gene indices (selected via screening on OTHER 285 cells)
        gene_idx = self.top_genes[:n_genes, rep_idx].astype(int)
        
        # Get the 285 cells NOT used for screening
        held_out_cells = self.screening_indices[rep_idx, :].astype(int)
        
        # Split these 285 cells into train (228) and test (57)
        np.random.seed(rep_idx * 42)  # Reproducible split
        np.random.shuffle(held_out_cells)
        
        n_train = 228  # ~80% of 285
        n_test = 57    # ~20% of 285
        
        train_idx = held_out_cells[:n_train]
        test_idx = held_out_cells[n_train:n_train+n_test]
        
        return {
            'X_train': self.X_processed[train_idx][:, gene_idx],
            'X_test': self.X_processed[test_idx][:, gene_idx],
            'y_train': self.y[train_idx],
            'y_test': self.y[test_idx],
            'gene_indices': gene_idx,
            'train_indices': train_idx,
            'test_indices': test_idx
        }

# ============= Main Analysis Function =============
def run_murine_analysis():
    """Run DeepLINK analysis on murine data"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    data_loader = MurineDataLoader()
    
    # Gene sizes to test
    gene_sizes = [20, 30, 50, 100, 200, 500]
    n_reps = 100
    
    # Storage for results
    all_results = {
        'filter': {},
        'gradient': {}
    }
    
    # Track selected genes across all runs
    gene_frequency_filter = {size: Counter() for size in gene_sizes}
    gene_frequency_gradient = {size: Counter() for size in gene_sizes}
    
    for n_genes in gene_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_genes} genes")
        print(f"{'='*60}")
        
        filter_results = {
            'train_acc_full': [], 'test_acc_full': [],
            'train_acc_selected': [], 'test_acc_selected': [],
            'selected_genes': [], 'n_selected': []
        }
        gradient_results = {
            'train_acc_full': [], 'test_acc_full': [],
            'train_acc_selected': [], 'test_acc_selected': [],
            'selected_genes': [], 'n_selected': []
        }
        
        for rep in range(n_reps):
            if rep % 10 == 0:
                print(f"  Repetition {rep}/{n_reps}")
            
            # Get data split
            split = data_loader.get_split_for_rep(rep, n_genes)
            
            # Initialize DiffKnock (diffusion-based knockoff generator)
            diffknock = DiffKnock(input_dim=n_genes, device=device)
            
            # Train diffusion and generate knockoffs (suppress output)
            _ = diffknock.train_diffusion_for_knockoffs(
                split['X_train'], n_epochs=300, batch_size=32, lr=1e-4
            )
            X_knockoffs = diffknock.generate_knockoffs(split['X_train'])
            
            # Train DeepLINK model (matching original structure)
            deeplink = DeepLINK(input_dim=n_genes, hidden_dims=[n_genes], dropout=0.1)
            deeplink.to(device)
            
            dataset = TensorDataset(
                torch.FloatTensor(split['X_train']),
                torch.FloatTensor(X_knockoffs),
                torch.FloatTensor(split['y_train'])
            )
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            optimizer = optim.Adam(deeplink.parameters(), lr=0.001, weight_decay=0.01)
            criterion = nn.BCEWithLogitsLoss()  # For binary classification
            
            deeplink.train()
            for epoch in range(300):
                for X_batch, X_knock_batch, y_batch in dataloader:
                    X_batch = X_batch.to(device)
                    X_knock_batch = X_knock_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    optimizer.zero_grad()
                    output = deeplink(X_batch, X_knock_batch)
                    loss = criterion(output.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on full feature set (using DeepLINK with knockoffs)
            deeplink.eval()
            with torch.no_grad():
                X_train_t = torch.FloatTensor(split['X_train']).to(device)
                X_test_t = torch.FloatTensor(split['X_test']).to(device)
                y_train_t = torch.FloatTensor(split['y_train']).to(device)
                y_test_t = torch.FloatTensor(split['y_test']).to(device)
                
                # For evaluation, create knockoffs for test set
                X_knockoffs_test = diffknock.generate_knockoffs(split['X_test'])
                X_knockoffs_test_t = torch.FloatTensor(X_knockoffs_test).to(device)
                X_knockoffs_train_t = torch.FloatTensor(X_knockoffs).to(device)
                
                train_pred_full = (torch.sigmoid(deeplink(X_train_t, X_knockoffs_train_t).squeeze()) > 0.5).float()
                test_pred_full = (torch.sigmoid(deeplink(X_test_t, X_knockoffs_test_t).squeeze()) > 0.5).float()
                train_acc_full = (train_pred_full == y_train_t).float().mean().item()
                test_acc_full = (test_pred_full == y_test_t).float().mean().item()
            
            # Compute knockoff statistics
            diffknock.deepLINK = deeplink  # Use the trained model
            W_filter = diffknock.compute_deeplink_statistics(deeplink)
            W_gradient = diffknock.compute_gradient_based_statistics(
                split['X_train'], X_knockoffs, split['y_train'],
                deeplink, criterion, batch_size=32
            )
            
            # Select features
            selected_filter = diffknock.select_features(W_filter, fdr_threshold=0.2)
            selected_gradient = diffknock.select_features(W_gradient, fdr_threshold=0.2)
            
            # Map back to original gene indices
            original_filter = split['gene_indices'][selected_filter] if len(selected_filter) > 0 else []
            original_gradient = split['gene_indices'][selected_gradient] if len(selected_gradient) > 0 else []
            
            # Update frequency counters
            for gene in original_filter:
                gene_frequency_filter[n_genes][gene] += 1
            for gene in original_gradient:
                gene_frequency_gradient[n_genes][gene] += 1
            
            # Evaluate selected features using new DeepLINK models
            # Filter method
            if len(selected_filter) > 0:
                X_train_selected = split['X_train'][:, selected_filter]
                X_test_selected = split['X_test'][:, selected_filter]
                
                # Slice already generated knockoffs for selected features
                X_knockoffs_selected = X_knockoffs[:, selected_filter]
                X_knockoffs_test_selected = X_knockoffs_test[:, selected_filter]

                # Train new model on selected features
                model_filter = DeepLINK(input_dim=len(selected_filter), hidden_dims=[len(selected_filter)], dropout=0.1)
                model_filter.to(device)
                
                dataset_selected = TensorDataset(
                    torch.FloatTensor(X_train_selected),
                    torch.FloatTensor(X_knockoffs_selected),
                    torch.FloatTensor(split['y_train'])
                )
                dataloader_selected = DataLoader(dataset_selected, batch_size=32, shuffle=True)
                
                optimizer_filter = optim.Adam(model_filter.parameters(), lr=0.001, weight_decay=0.01)
                
                model_filter.train()
                for epoch in range(200):
                    for X_batch, X_knock_batch, y_batch in dataloader_selected:
                        X_batch = X_batch.to(device)
                        X_knock_batch = X_knock_batch.to(device)
                        y_batch = y_batch.to(device)
                        
                        optimizer_filter.zero_grad()
                        output = model_filter(X_batch, X_knock_batch)
                        loss = criterion(output.squeeze(), y_batch)
                        loss.backward()
                        optimizer_filter.step()
                
                model_filter.eval()
                with torch.no_grad():
                    X_train_sel_t = torch.FloatTensor(X_train_selected).to(device)
                    X_test_sel_t = torch.FloatTensor(X_test_selected).to(device)
                    X_knock_train_sel_t = torch.FloatTensor(X_knockoffs_selected).to(device)
                    X_knock_test_sel_t = torch.FloatTensor(X_knockoffs_test_selected).to(device)
                    
                    train_pred_sel = (torch.sigmoid(model_filter(X_train_sel_t, X_knock_train_sel_t).squeeze()) > 0.5).float()
                    test_pred_sel = (torch.sigmoid(model_filter(X_test_sel_t, X_knock_test_sel_t).squeeze()) > 0.5).float()
                    train_acc_filter = (train_pred_sel == y_train_t).float().mean().item()
                    test_acc_filter = (test_pred_sel == y_test_t).float().mean().item()
            else:
                # No features selected - use majority class
                train_acc_filter = max(y_train_t.mean().item(), 1 - y_train_t.mean().item())
                test_acc_filter = max(y_test_t.mean().item(), 1 - y_test_t.mean().item())
            
            # Gradient method (similar process)
            if len(selected_gradient) > 0:
                X_train_selected = split['X_train'][:, selected_gradient]
                X_test_selected = split['X_test'][:, selected_gradient]
                
                # Slice already generated knockoffs for selected features
                X_knockoffs_selected = X_knockoffs[:, selected_gradient]
                X_knockoffs_test_selected = X_knockoffs_test[:, selected_gradient]
                
                model_gradient = DeepLINK(input_dim=len(selected_gradient), hidden_dims=[len(selected_gradient)], dropout=0.1)
                model_gradient.to(device)
                
                dataset_selected = TensorDataset(
                    torch.FloatTensor(X_train_selected),
                    torch.FloatTensor(X_knockoffs_selected),
                    torch.FloatTensor(split['y_train'])
                )
                dataloader_selected = DataLoader(dataset_selected, batch_size=32, shuffle=True)
                
                optimizer_gradient = optim.Adam(model_gradient.parameters(), lr=0.001, weight_decay=0.01)
                
                model_gradient.train()
                for epoch in range(200):
                    for X_batch, X_knock_batch, y_batch in dataloader_selected:
                        X_batch = X_batch.to(device)
                        X_knock_batch = X_knock_batch.to(device)
                        y_batch = y_batch.to(device)
                        
                        optimizer_gradient.zero_grad()
                        output = model_gradient(X_batch, X_knock_batch)
                        loss = criterion(output.squeeze(), y_batch)
                        loss.backward()
                        optimizer_gradient.step()
                
                model_gradient.eval()
                with torch.no_grad():
                    X_train_sel_t = torch.FloatTensor(X_train_selected).to(device)
                    X_test_sel_t = torch.FloatTensor(X_test_selected).to(device)
                    X_knock_train_sel_t = torch.FloatTensor(X_knockoffs_selected).to(device)
                    X_knock_test_sel_t = torch.FloatTensor(X_knockoffs_test_selected).to(device)
                    
                    train_pred_sel = (torch.sigmoid(model_gradient(X_train_sel_t, X_knock_train_sel_t).squeeze()) > 0.5).float()
                    test_pred_sel = (torch.sigmoid(model_gradient(X_test_sel_t, X_knock_test_sel_t).squeeze()) > 0.5).float()
                    train_acc_gradient = (train_pred_sel == y_train_t).float().mean().item()
                    test_acc_gradient = (test_pred_sel == y_test_t).float().mean().item()
            else:
                train_acc_gradient = max(y_train_t.mean().item(), 1 - y_train_t.mean().item())
                test_acc_gradient = max(y_test_t.mean().item(), 1 - y_test_t.mean().item())
            
            # Store results
            filter_results['train_acc_full'].append(train_acc_full)
            filter_results['test_acc_full'].append(test_acc_full)
            filter_results['train_acc_selected'].append(train_acc_filter)
            filter_results['test_acc_selected'].append(test_acc_filter)
            filter_results['selected_genes'].append(original_filter)
            filter_results['n_selected'].append(len(selected_filter))
            
            gradient_results['train_acc_full'].append(train_acc_full)
            gradient_results['test_acc_full'].append(test_acc_full)
            gradient_results['train_acc_selected'].append(train_acc_gradient)
            gradient_results['test_acc_selected'].append(test_acc_gradient)
            gradient_results['selected_genes'].append(original_gradient)
            gradient_results['n_selected'].append(len(selected_gradient))
        
        # Store results for this gene size
        all_results['filter'][n_genes] = filter_results
        all_results['gradient'][n_genes] = gradient_results
        
        # Print summary
        print(f"\nSummary for {n_genes} genes:")
        print(f"  Using ALL {n_genes} genes:")
        print(f"    Train acc: {np.mean(filter_results['train_acc_full']):.3f} ± {np.std(filter_results['train_acc_full']):.3f}")
        print(f"    Test acc: {np.mean(filter_results['test_acc_full']):.3f} ± {np.std(filter_results['test_acc_full']):.3f}")
        
        print(f"\n  Filter method (selected features):")
        print(f"    Avg genes selected: {np.mean(filter_results['n_selected']):.1f} ± {np.std(filter_results['n_selected']):.1f}")
        print(f"    Train acc: {np.mean(filter_results['train_acc_selected']):.3f} ± {np.std(filter_results['train_acc_selected']):.3f}")
        print(f"    Test acc: {np.mean(filter_results['test_acc_selected']):.3f} ± {np.std(filter_results['test_acc_selected']):.3f}")
        
        print(f"\n  Gradient method (selected features):")
        print(f"    Avg genes selected: {np.mean(gradient_results['n_selected']):.1f} ± {np.std(gradient_results['n_selected']):.1f}")
        print(f"    Train acc: {np.mean(gradient_results['train_acc_selected']):.3f} ± {np.std(gradient_results['train_acc_selected']):.3f}")
        print(f"    Test acc: {np.mean(gradient_results['test_acc_selected']):.3f} ± {np.std(gradient_results['test_acc_selected']):.3f}")
    
    
    # Report top 20 most frequently selected genes with names
    print("\n" + "="*60)
    print("TOP 20 MOST FREQUENTLY SELECTED GENES")
    print("="*60)
    
    # Print Filter method results
    print("\n" + "-"*40)
    print("FILTER METHOD")
    print("-"*40)
    for n_genes in gene_sizes:
        print(f"\nWith {n_genes} genes in initial set:")
        top_20 = gene_frequency_filter[n_genes].most_common(20)
        if top_20:  # Only print if there are selected genes
            for i, (gene_idx, count) in enumerate(top_20, 1):
                freq = count / n_reps * 100
                gene_name = data_loader.gene_names[gene_idx]
                print(f"  {i:2d}. {gene_name}: {count:3d} times ({freq:5.1f}%)")
        else:
            print("  No genes selected")
    
    # Print Gradient method results
    print("\n" + "-"*40)
    print("GRADIENT METHOD")
    print("-"*40)
    for n_genes in gene_sizes:
        print(f"\nWith {n_genes} genes in initial set:")
        top_20 = gene_frequency_gradient[n_genes].most_common(20)
        if top_20:  # Only print if there are selected genes
            for i, (gene_idx, count) in enumerate(top_20, 1):
                freq = count / n_reps * 100
                gene_name = data_loader.gene_names[gene_idx]
                print(f"  {i:2d}. {gene_name}: {count:3d} times ({freq:5.1f}%)")
        else:
            print("  No genes selected")
    
    # Save detailed results
    results_df = pd.DataFrame()
    for method in ['filter', 'gradient']:
        for n_genes in gene_sizes:
            row = {
                'method': method,
                'n_genes': n_genes,
                'mean_train_acc_full': np.mean(all_results[method][n_genes]['train_acc_full']),
                'std_train_acc_full': np.std(all_results[method][n_genes]['train_acc_full']),
                'mean_test_acc_full': np.mean(all_results[method][n_genes]['test_acc_full']),
                'std_test_acc_full': np.std(all_results[method][n_genes]['test_acc_full']),
                'mean_train_acc_selected': np.mean(all_results[method][n_genes]['train_acc_selected']),
                'std_train_acc_selected': np.std(all_results[method][n_genes]['train_acc_selected']),
                'mean_test_acc_selected': np.mean(all_results[method][n_genes]['test_acc_selected']),
                'std_test_acc_selected': np.std(all_results[method][n_genes]['test_acc_selected']),
                'mean_n_selected': np.mean(all_results[method][n_genes]['n_selected']),
                'std_n_selected': np.std(all_results[method][n_genes]['n_selected'])
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
    results_df.to_csv(os.path.join(RESULT_DIR, 'murine_results_summary.csv'), index=False)
    print(f"\nResults saved to {RESULT_DIR}/murine_results_summary.csv")
    
    return all_results, gene_frequency_filter, gene_frequency_gradient

if __name__ == "__main__":
    results, gene_freq_filter, gene_freq_gradient = run_murine_analysis()
