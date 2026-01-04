# training procedure for DTCR model on UCR datasets

import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ucr_loader import load_dataset, IndexedDataset, DataLoader
from dtcr_model import DTCR  
from sklearn.cluster import KMeans
import time
import pandas as pd
from tqdm.auto import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cluster_accuracy(assign, labels):
    # assignments are cluster indices, labels are true classes
    # we need to match clusters → labels using best permutation
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, assign)
    r, c = linear_sum_assignment(cm.max() - cm)
    return cm[r, c].sum() / len(assign)


def plot_or_save_history(history, plot=True, save=False):
    # On crée une figure plus haute pour accomoder les 4 subplots verticaux
    fig = plt.figure(figsize=(18, 9))
    
    # Grille de 4 lignes x 3 colonnes
    # La colonne 0 servira aux 4 losses séparées
    # Les colonnes 1 et 2 serviront aux plots Log et Métriques (qui prendront toute la hauteur)
    gs = fig.add_gridspec(4, 3)

    # --- COLONNE 1 : Les 4 Losses séparées (Linéaire) ---
    
    # 1. Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['loss'], label='Total Loss', color='black')
    ax1.set_title('Total Loss')
    ax1.grid(True, alpha=0.3)
    # On retire les labels x pour les graphes du haut pour ne pas surcharger
    ax1.set_xticklabels([]) 

    # 2. Reconstruction
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history['recon'], label='Reconstruction', color='blue')
    ax2.set_title('Reconstruction Loss')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])

    # 3. Classification
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(history['classif'], label='Classification', color='green')
    ax3.set_title('Classification Loss')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels([])

    # 4. K-Means
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(history['kmeans'], label='K-Means', color='red')
    ax4.set_title('K-Means Loss')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Epoch') # Seulement sur le dernier
    
    # --- COLONNE 2 : Toutes les losses en Log Scale (Prend toute la hauteur) ---
    ax_log = fig.add_subplot(gs[:, 1]) # ":" signifie "toutes les lignes"
    ax_log.plot(history['loss'], label='Total', color='black', alpha=0.7)
    ax_log.plot(history['recon'], label='Reconstruction', color='blue', alpha=0.7)
    ax_log.plot(history['classif'], label='Classification', color='green', alpha=0.7)
    ax_log.plot(history['kmeans'], label='K-Means', color='red', alpha=0.7)
    ax_log.set_title('All Losses (Log Scale)')
    ax_log.set_xlabel('Epoch')
    ax_log.set_ylabel('Loss (log)')
    ax_log.set_yscale('log')
    ax_log.grid(True, alpha=0.3, which='both')
    ax_log.legend()

    # --- COLONNE 3 : Métriques de Clustering (Prend toute la hauteur) ---
    ax_metrics = fig.add_subplot(gs[:, 2])
    
    # ax_metrics.plot(history['acc'], label='ACC', alpha=0.8)
    # ax_metrics.plot(history['ari'], label='ARI', alpha=0.8)
    # ax_metrics.plot(history['nmi'], label='NMI', alpha=0.8)
    ax_metrics.plot( history['ri'], label='RI', color='red', alpha=0.7)
    
    ax_metrics.set_title('Clustering Metrics')
    ax_metrics.set_xlabel('Epoch')
    ax_metrics.set_ylabel('Score')
    #ax_metrics.set_ylim(0, 1.05) # Fixe l'échelle entre 0 et 1
    ax_metrics.grid(True, alpha=0.3)
    ax_metrics.legend()

    plt.tight_layout()

    if save: plt.savefig(os.path.join('.','figures',f'history_{time.strftime("%Y%m%d-%H%M%S")}.pdf'))
    if plot: plt.show()


def make_fake(x, alpha=0.2): # (cf article)
    B, T, F = x.shape
    x_fake = x.clone()

    num_shuffle = int(T * alpha)
    idx = torch.randperm(T)[:num_shuffle]
    shuffled = x[:, idx[torch.randperm(num_shuffle)], :]
    x_fake[:, idx, :] = shuffled
    return x_fake


def compute_full_embeddings(model, loader):
    model.eval()
    H_all = []
    with torch.no_grad():
        for _, x, _ in loader:
            x = x.to(DEVICE)
            _, z = model(x)
            H_all.append(z)
    H_all = torch.cat(H_all, dim=0)  # (N, embedding_size)
    return H_all


def update_F(H, K):
    # H: (N, d)
    # Compute truncated SVD for first K left singular vectors
    U, S, V = torch.svd(H)  # U: (N, d)
    F = U[:, :K]            # take first K left singular vectors
    return F


def train_dtcr(train_loader, test_loader, model, epochs=100, lr=5e-3, verbose=False, save_history_plot=False):

    optimizer = optim.Adam(model.parameters(), lr=lr) # lr=5e-3 (cf article)

    history = defaultdict(list)

    max_RI = 0
    max_NMI = 0

    progress_bar = tqdm(range(1, epochs+1), desc="Training DTCR")
    # ---- Training Loop ----
    for epoch in progress_bar:

        if model.iteration % model.F_update_freq == 0:
            if verbose:
                print("Updating F...")
            H_all = compute_full_embeddings(model, train_loader)  # full dataset
            model.F = update_F(H_all, model.K)                    # update F
            
        model.iteration += 1
        
        model.train()
        total_loss = 0
        recon_loss = 0
        classif_loss = 0
        kmeans_loss = 0
        for indices, x_real, _ in train_loader:

            x_real = x_real.to(DEVICE)
            x_fake = make_fake(x_real).to(DEVICE)

            out = model.total_loss(x_real, x_fake, indices)  # recon + classif + lambda * kmeans
            loss = out['total']
            loss_r = out['recon_loss']
            loss_c = out['classif_loss']
            loss_km = out['kmeans_loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_real.size(0)
            recon_loss += loss_r.item() * x_real.size(0)
            classif_loss += loss_c.item() * x_real.size(0)
            kmeans_loss += loss_km.item() * x_real.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = recon_loss / len(train_loader.dataset)
        avg_classif = classif_loss / len(train_loader.dataset)
        avg_kmeans = kmeans_loss / len(train_loader.dataset)

        # ---- Evaluate clustering ----
        # The idea is to compute clustering metrics on the test set
        # and consider that the model with the best test metrics is the best (be careful, data leakage here!)
        model.eval()
        all_assign = []
        all_labels = []

        for _, x, y in test_loader:
            x = x.to(DEVICE)

            _, z = model(x)

            H = z.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=model.K)
            assign = kmeans.fit_predict(H)

            all_assign.append(assign)
            all_labels.append(y.numpy())

        all_assign = np.concatenate(all_assign)
        all_labels = np.concatenate(all_labels)

        # Compute clustering metrics
        acc = cluster_accuracy(all_assign, all_labels)
        ari = adjusted_rand_score(all_labels, all_assign)
        nmi = normalized_mutual_info_score(all_labels, all_assign)
        ri  = rand_score(all_labels, all_assign)

        history['acc'].append(acc)
        history['ari'].append(ari)
        history['nmi'].append(nmi)
        history['ri'].append(ri)

        history['loss'].append(avg_loss)
        history['recon'].append(avg_recon)
        history['classif'].append(avg_classif)
        history['kmeans'].append(avg_kmeans)

        # save best metrics
        if ri > max_RI : max_RI = ri
        if nmi > max_NMI : max_NMI = nmi

        progress_bar.set_postfix({
            'Loss': f"{avg_loss:.3f}", 
            'RI': f"{ri:.3f}", 
            'Max_RI': f"{max_RI:.3f}",
            'NMI': f"{ri:.3f}", 
            'Max_NMI': f"{max_RI:.3f}",
        })

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:02d}: loss={avg_loss:.4f} (recon={avg_recon}, classif={avg_classif}, k-means={avg_kmeans}), ACC={acc:.4f}, ARI={ari:.4f}")

        # ---- t-SNE Visualization ----
        if verbose and (epoch == 50 or epoch == 300 or epoch == 1000):
            print(f"Computing t-SNE visualization for Epoch {epoch}...")
            from sklearn.manifold import TSNE
            
            model.eval()
            z_all = []
            y_all = []
            
            with torch.no_grad():
                # Unpack matches IndexedDataset: (index, x, y)
                for _, x, y in test_loader: 
                    x = x.to(DEVICE)
                    _, z = model(x)  # Get embeddings
                    z_all.append(z.cpu().detach().numpy())
                    y_all.append(y.numpy())
            
            Z = np.concatenate(z_all, axis=0)
            Y = np.concatenate(y_all, axis=0)

            # Run t-SNE
            # perplexity=30 is standard. init='pca' usually improves stability.
            tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
            Z_tsne = tsne.fit_transform(Z)

            # Plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=Y, cmap='tab10', s=20, alpha=0.6)
            plt.colorbar(scatter, label='True Class')
            plt.title(f't-SNE Latent Space (Epoch {epoch})')
            plt.grid(True, alpha=0.3)
            plt.show()

        # ---- Reconstruction Visualization (Every 100 epochs) ----
        if verbose and epoch % 100 == 0:
            print(f"Visualizing signal vs reconstruction for Epoch {epoch}...")
            model.eval()
            with torch.no_grad():
                # Get a single batch from the test loader
                # We use next(iter(...)) to grab the first batch
                _, x_vis, _ = next(iter(test_loader)) 
                x_vis = x_vis.to(DEVICE)
                
                # Forward pass to get reconstruction
                # Assuming model returns (reconstruction, latent) based on _, z = model(x) usage
                x_hat, _ = model(x_vis)
                
                # Move to CPU/Numpy for plotting
                x_vis = x_vis.cpu().numpy()
                x_hat = x_hat.cpu().numpy()
                
                # Plot the first 4 samples
                n_plots = 4
                fig, axs = plt.subplots(n_plots, 1, figsize=(10, 10))
                
                for i in range(n_plots):
                    # .squeeze() handles the feature dimension (Time, 1) -> (Time,)
                    axs[i].plot(x_vis[i].squeeze(), label='Original', color='black', linewidth=1.5, alpha=0.7)
                    axs[i].plot(x_hat[i].squeeze(), label='Reconstructed', color='red', linestyle='--', linewidth=1.5)
                    axs[i].set_title(f'Sample {i+1}')
                    axs[i].legend()
                    axs[i].grid(True, alpha=0.3)
                
                plt.suptitle(f'Signal Reconstruction at Epoch {epoch}')
                plt.tight_layout()
                plt.show()

    # Plot loss and metrics over epochs
    if verbose or save_history_plot :
        plot_or_save_history(history, plot=verbose, save=save_history_plot)

    return max_RI, max_NMI

def random_network(train_loader, test_loader, drnn_hidden_sizes, epochs=100, verbose=False):

    history = defaultdict(list)

    max_RI = 0
    max_NMI = 0

    for epoch in range(epochs):

        if epoch % 100 == 0 :
            print(f"{epoch=}")

        # ---- Initialize Model ----
        model = DTCR(
            input_size=seq_dim, # seq dimension (univariate for UCR datasets)
            num_steps=seq_len,
            cell_type="GRU", # (cf article)
            drnn_hidden_sizes=[50, 30, 30], # [100, 50, 50] or [50, 30, 30] (cf article)
            lamb=1e-2, # in [1, 1e-1, 1e-2, 1e-3] (cf article)
            class_num=7, # Plane dataset = 7 classes
            Kmeans_objective=True,
            classification_task=True
        ).to(DEVICE)


        # ---- Evaluate clustering ----
        model.eval()
        all_assign = []
        all_labels = []

        for _, x, y in test_loader:
            x = x.to(DEVICE)

            _, z = model(x)

            H = z.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=model.K)
            assign = kmeans.fit_predict(H)

            all_assign.append(assign)
            all_labels.append(y.numpy())

        all_assign = np.concatenate(all_assign)
        all_labels = np.concatenate(all_labels)

        # Compute clustering metrics
        acc = cluster_accuracy(all_assign, all_labels)
        ari = adjusted_rand_score(all_labels, all_assign)
        nmi = normalized_mutual_info_score(all_labels, all_assign)
        ri  = rand_score(all_labels, all_assign)

        history['acc'].append(acc)
        history['ari'].append(ari)
        history['nmi'].append(nmi)
        history['ri'].append(ri)

        # save best metrics
        if ri > max_RI : max_RI = ri
        if nmi > max_NMI : max_NMI = nmi

    if verbose:
        plt.figure()
        for key in history.keys():
            plt.plot(history[key], label=key, alpha=0.6)
        plt.ylim([0,1])
        plt.legend()
        plt.show()

    return max_RI, max_NMI
        



if __name__ == "__main__":

    one_shot_train = True 

    if one_shot_train:

        # ---- Load Data ----
        X_train, y_train, X_test, y_test = load_dataset(name="Plane")

        plt.figure(figsize=(6,6))
        for i in range(3):
            plt.plot(X_train[i], label=f'class {y_train[i]}')
        plt.title('Sample Time Series from Plane Dataset')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print(np.unique(y_train))

        train_ds = IndexedDataset(X_train, y_train)
        test_ds  = IndexedDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

        seq_len = X_train.shape[1]
        seq_dim = 1  # univariate time series

        epochs=300 

        # ---- Initialize Model ----
        model = DTCR(
            input_size=seq_dim, # dimension of the sequence (univariate for UCR datasets)
            num_steps=seq_len,
            cell_type="GRU", # (cf article)
            drnn_hidden_sizes=[50, 30, 30], # [100, 50, 50] or [50, 30, 30] (cf article)
            lamb=1e-3, # in [1, 1e-1, 1e-2, 1e-3] (cf article)
            class_num=7, # Plane dataset = 7 classes
            Kmeans_objective=True,
            classification_task=True
        ).to(DEVICE)

        max_RI, max_NMI = train_dtcr(train_loader, test_loader, model, epochs=epochs, lr=5e-3, verbose=True)
        print(f"RI: {max_RI} NMI: {max_NMI}")

        #random_network(train_loader, test_loader, drnn_hidden_sizes=[50,30,30],  epochs=epochs)
        
    else: 

        list_dataset_name = ["Plane"]

        # Hyperparameter grids, see 5.2 in the report (long computations)
        list_drnn_hidden_sizes = [[50, 30, 30], [100, 50, 50]]
        list_lambda = [1, 1e-1, 1e-2, 1e-3]
        list_Kmeans_classif = [(True, True)]
        epochs = 300
        nb_runs = 10

        # # Influence of the objectives, see 5.4 in the report
        # list_drnn_hidden_sizes = [[50, 30, 30]]
        # list_lambda = [1e-3]
        # # list of tuples indicating if we consider each of the objectives
        # list_Kmeans_classif = [(True, False), 
        #                        (False, True), 
        #                        (False, False)] 
        # epochs = 300
        # nb_runs = 10

        # Influence of the dilatation schedule, see 5.3 in the report
        # Run the 5.5 experiments with different dilatation schedules by changing line 18 in drnn.py
        # self.dilations = [4**i for i in range(self.n_layers)] # original dilatation schedule (cf article)
        # self.dilations = [2**i for i in range(self.n_layers)] # low dilatation schedule
        # self.dilations = [8**i for i in range(self.n_layers)] # high dilatation schedule
        # self.dilations = [1**i for i in range(self.n_layers)] # no dilatation
        # Please also change epochs=300 (line 469 in train.py)

        # # Learning over 2000 epochs, see 5.5 in the report
        # # Do not forget to change the dilatation schedule in drnn.py back to the original one if you changed it before
        # # and save figure of history by changing save_history_plot=True in line 469
        # list_drnn_hidden_sizes = [[50, 30, 30]]
        # list_lambda = [1e-3]
        # list_Kmeans_classif = [(True, True)]
        # epochs = 2000
        # nb_runs = 10

        results_list = []
        start_time = time.time()
        for name in list_dataset_name:

            # ---- Load Data ----
            X_train, y_train, X_test, y_test = load_dataset(name=name)

            n_classes = len(np.unique(y_train))
            print(f"Classes detected: {n_classes}")

            train_ds = IndexedDataset(X_train, y_train)
            test_ds  = IndexedDataset(X_test, y_test)

            train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
            test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

            seq_len = X_train.shape[1]
            seq_dim = 1  # univariate time series

            for drnn_hidden_sizes in list_drnn_hidden_sizes:
                for lamb in list_lambda:
                    for Kmeans_objective, classification_task in list_Kmeans_classif:

                        print(f"Dataset: {name} | drnn_hidden_sizes: {drnn_hidden_sizes} | lambda: {lamb}")

                        # Temporary lists to store metrics for the 10 runs
                        list_max_RI = []
                        list_max_NMI = []

                        for run_i in range(nb_runs):

                            print(f"   > Run {run_i+1}/{nb_runs}...", end=" ")

                            # ---- Initialize Model ----
                            model = DTCR(
                                input_size=seq_dim, # dimension of the sequence (univariate for UCR datasets)
                                num_steps=seq_len,
                                cell_type="GRU", # (cf article)
                                drnn_hidden_sizes=drnn_hidden_sizes, # [100, 50, 50] or [50, 30, 30] (cf article)
                                lamb=lamb, # in [1, 1e-1, 1e-2, 1e-3] (cf article)
                                class_num=n_classes,
                                Kmeans_objective=Kmeans_objective,
                                classification_task=classification_task
                            ).to(DEVICE)

                            # All experiments except baseline random network
                            max_RI, max_NMI = train_dtcr(train_loader, test_loader, model, epochs=epochs, lr=5e-3, verbose=False, save_history_plot=False)
                            # # Experiment baseline random network
                            # max_RI, max_NMI = random_network(train_loader, test_loader, drnn_hidden_sizes, epochs=epochs)

                            list_max_RI.append(max_RI)
                            list_max_NMI.append(max_NMI)

                            print(f"RI: {max_RI:.4f} | NMI: {max_NMI:.4f}")
                            
                        mean_RI = np.mean(list_max_RI)
                        std_RI = np.std(list_max_RI)

                        mean_NMI = np.mean(list_max_NMI)
                        std_NMI  = np.std(list_max_NMI)

                        print(f"   >>> Result: Mean RI: {mean_RI:.3f} (±{std_RI:.3f}) | Mean NMI: {mean_NMI:.3f} (±{std_NMI:.3f})")

                        results_list.append({
                            "Dataset": name,
                            "Hidden_Sizes": str(drnn_hidden_sizes),
                            "Lambda": lamb,
                            "Kmeans_Objective": Kmeans_objective,
                            "Classification_Task": classification_task,
                            "Mean_RI": mean_RI,
                            "Std_RI": std_RI,
                            "Mean_NMI": mean_NMI,
                            "Std_NMI": std_NMI,
                            "All_RIs": list_max_RI,
                            "All_NMIs": list_max_NMI
                        })
            
        print(f"Time: {time.time() - start_time}")

        df_results = pd.DataFrame(results_list)

        # Display table
        print("\n\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        print(df_results[["Dataset", "Hidden_Sizes", "Lambda", "Kmeans_Objective", "Classification_Task", "Mean_RI", "Std_RI"]])

        # Save to CSV
        df_results.to_csv(f"experiment_results_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
        print(f"\nResults saved to CSV. Total time: {time.time() - start_time:.2f}s")