# train_dtcr_plane.py

import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt

from ucr_loader import load_plane_dataset, IndexedDataset, DataLoader #TimeSeriesDataset
from dtcr_model import DTCR   # Your DTCR implementation

from sklearn.cluster import KMeans


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cluster_accuracy(assign, labels):
    # assignments are cluster indices, labels are true classes
    # we need to match clusters → labels using best permutation
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, assign)
    r, c = linear_sum_assignment(cm.max() - cm)
    return cm[r, c].sum() / len(assign)


def initialize_cluster_centers(model, loader):
    """Run embedding once over data, run KMeans on embeddings."""
    from sklearn.cluster import KMeans

    all_z = []
    for x, _ in loader:
        x = x.to(DEVICE)
        _, z = model(x)
        all_z.append(z.cpu().detach().numpy())
    Z = np.concatenate(all_z, axis=0)

    kmeans = KMeans(n_clusters=model.K).fit(Z)
    model.cluster_centers.data = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float32, device=DEVICE
    )


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


def train_dtcr():
    # ---- Load Data ----
    X_train, y_train, X_test, y_test = load_plane_dataset()

    # print(X_train.shape)

    # plt.figure()
    # for i in range(5) :
    #     plt.plot(X_train[i], label=y_train[i])
    # plt.legend()
    # plt.show()

    # train_ds = TimeSeriesDataset(X_train, y_train)
    # test_ds  = TimeSeriesDataset(X_test, y_test)

    train_ds = IndexedDataset(X_train, y_train)
    test_ds  = IndexedDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    seq_len = X_train.shape[1]
    seq_dim = 1  # univariate time series

    # ---- Initialize Model ----
    model = DTCR(
        input_size=seq_dim, # seq_len
        num_steps=seq_len,
        embedding_size=16,
        cell_type="GRU", # (cf article)
        drnn_layers=3, # (cf article)                  
        drnn_hidden_sizes=[100, 50, 50], # [100, 50, 50] or [50, 30, 30] (cf article)
        lamb=0.01, # in [1, 1e-1, 1e-2, 1e-3] (cf article)
        class_num=7,     # Plane dataset = 7 classes
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=5e-3) # lr=1e-3 (cf article)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10) # (not in the article)

    # # ---- KMeans initialization ----
    # print("Initializing cluster centers...")
    # initialize_cluster_centers(model, train_loader)

    list_loss = []
    list_recon = []
    list_classif = []
    list_kmeans = []
    list_acc = []
    list_ari = []

    # ---- Training Loop ----
    EPOCHS = 300
    for epoch in range(1, EPOCHS+1):

        if model.iteration % model.F_update_freq == 0:
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

            out = model.total_loss(x_real, x_fake, indices)  # recon + classif + λ * kmeans
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

        avg_loss = total_loss / len(train_ds)
        avg_recon = recon_loss / len(train_ds)
        avg_classif = classif_loss / len(train_ds)
        avg_kmeans = kmeans_loss / len(train_ds)

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

        # print(all_assign)
        # print(all_labels)

        acc = cluster_accuracy(all_assign, all_labels)
        ari = adjusted_rand_score(all_labels, all_assign)

        list_loss.append(avg_loss)
        list_recon.append(avg_recon)
        list_classif.append(avg_classif)
        list_kmeans.append(avg_kmeans)
        list_acc.append(acc)
        list_ari.append(ari)

        print(f"Epoch {epoch:02d}: loss={avg_loss:.4f} (recon={avg_recon}, classif={avg_classif}, k-means={avg_kmeans}), ACC={acc:.4f}, ARI={ari:.4f}")

        # scheduler.step(avg_loss)

        # if epoch == 30 or epoch == 50 or epoch == 100:
        #     # Plot embeddings
        #     plt.figure()
        #     z_all = []
        #     for x, _ in test_loader:
        #         x = x.to(DEVICE)
        #         _, z = model(x)
        #         z_all.append(z.cpu().detach().numpy())
        #     Z = np.concatenate(z_all, axis=0)

        #     plt.scatter(Z[:, 0], Z[:, 1], c=all_labels, cmap='tab10', s=15)
        #     plt.title(f'Embeddings at Epoch {epoch}')
        #     plt.show()

    # Plot loss and metrics over epochs
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(list_loss, label='Total Loss')
    plt.plot(list_recon, label='Reconstruction Loss')
    plt.plot(list_classif, label='Classification Loss')
    plt.plot(list_kmeans, label='K-Means Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(list_acc, label='ACC')
    plt.plot(list_ari, label='ARI')
    plt.title('Clustering Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_dtcr()
