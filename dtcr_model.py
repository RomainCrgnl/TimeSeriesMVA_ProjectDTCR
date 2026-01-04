# Time series final project (Master MVA) about DTCR model

import torch
import torch.nn as nn
import torch.nn.functional as F
from drnn import DRNN


class DTCR(nn.Module):

    def __init__(
        self,
        input_size,                      # number of features per timestep
        num_steps,                       # sequence length (T)
        cell_type='GRU',                 # or 'RNN' or 'LSTM' depending on drnn implementation
        drnn_hidden_sizes=[100, 50, 50], # list of hidden sizes per layer
        bidirectional=True,              # whether to use bidirectional DRNN
        lamb=1e-3,                       # weight for kmeans loss
        class_num=10,                    # K clusters
        F_update_freq=10,                # frequency to update F in K-means loss
        d=128,                           # hidden size for the fc layer of fake classifier
        Kmeans_objective=True,
        classification_task=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_steps = num_steps
        self.cell_type = cell_type
        self.drnn_hidden_sizes = drnn_hidden_sizes
        self.drnn_layers = len(drnn_hidden_sizes)
        self.lamb = lamb
        self.K = class_num
        self.d = d

        self.Kmeans_objective = Kmeans_objective # In order to train the model with and without K-means objective
        self.classification_task = classification_task # In order to train the model with and without classification task
        
        factor = 2 if bidirectional else 1
        self.embedding_size = sum(drnn_hidden_sizes) * factor # (cf article), concatenate last hidden state from all encoder layers, *2 for bidirectional

        self.iteration = 0
        self.F_update_freq = F_update_freq

        # Encoder: DRNN 
        self.encoder_drnn = DRNN(
            n_input=self.input_size, 
            hidden_sizes=self.drnn_hidden_sizes, 
            cell_type=self.cell_type,
            bidirectional=bidirectional
        )

        # Decoder: RNN composed of GRU cells
        self.decoder_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.embedding_size)
        self.dec_out = nn.Linear(self.embedding_size, self.input_size)

        if self.classification_task:
            self.fake_classifier = nn.Sequential(
                nn.Linear(self.embedding_size, self.d),
                nn.ReLU(),
                nn.Linear(self.d, 2)   # output = (real_logit, fake_logit)
            )

    def encode(self, x):
        """
        x: tensor shape (batch, seq_len, input_size)
        returns embedding z: (batch, embedding_size)
        """
        _, layer_outputs = self.encoder_drnn(x)             
        final_states = [out[-1] for out in layer_outputs]
        #The latent representation is obtained by concatenating the last hidden state output of each layer of the Dilated RNN
        h = torch.cat(final_states, dim=1) 
        return h

    def decode(self, z):

        batch_size = z.size(0)
        
        # Initialize Hidden State with z
        h = z 
        
        # Initialize Input with Zeros (Start Token)
        curr_input = torch.zeros(batch_size, self.input_size, device=z.device)
        
        recon_seq = []
        for t in range(self.num_steps):
            # Input is (Batch, 1), Hidden is (Batch, 2*(m_1+m_2+m_3))
            h = self.decoder_cell(curr_input, h)
            out = self.dec_out(h) # Output is (Batch, 1)
            recon_seq.append(out)
            
            # Feed output as next input
            curr_input = out 
            
        return torch.stack(recon_seq, dim=1)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        returns: recon (batch, seq_len, input_size), embedding z (batch, embedding_size)
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def recon_loss(self, x, recon):
        """
        Standard reconstruction MSE loss
        """
        loss = F.mse_loss(recon, x, reduction='mean')
        return loss

    def kmeans_loss(self, H, F): # giving self is useless here but avoid errors
        """
        H: (batch, embedding_size)
        F: (batch, K)
        """
        FFt = F @ F.t() # (N, N)
        I = torch.eye(FFt.size(0), device=H.device)

        term1 = torch.trace(H.t() @ H)
        term2 = torch.trace(F.t() @ (H.t() @ H) @ F)
        loss = term1 - term2
        return loss

    def classif_loss(self, z, true_fake_labels):
        """
        Classification of true and fake data.
        labels: (batch, 2) one-hot encoded real/fake labels ([1,0] for real, [0,1] for fake)
        """
        logits = self.fake_classifier(z)  # (batch, 2)
        return F.binary_cross_entropy_with_logits(logits, true_fake_labels)

    def total_loss(self, x_real, x_fake, indices):
        """
        Convenience: pass input real x's (batch,seq_len,input_size), fake x's and indices for F.
        Returns dict of losses + recon and z for diagnostics.
        """
        recon, z_real = self.forward(x_real)
        _, z_fake = self.forward(x_fake)
        z_all = torch.cat([z_real, z_fake], dim=0)

        real_fake_labels = torch.cat([
            torch.tensor([[1., 0.]]).repeat(len(x_real), 1),   # real
            torch.tensor([[0., 1.]]).repeat(len(x_fake), 1)    # fake
        ], dim=0).to(x_real.device)
        
        # Reconstruction loss
        recon_l = self.recon_loss(x_real, recon)

        # Classification loss
        if self.classification_task:
            classif_l = self.classif_loss(z_all, real_fake_labels)
        else :
            classif_l = torch.tensor(0.0, device=x_real.device)

        # K-means loss
        if self.lamb == 0 or not(self.Kmeans_objective): # avoid computing kmeans loss if not used
            kmeans_l = torch.tensor(0.0, device=x_real.device)
        else:
            kmeans_l = self.kmeans_loss(z_real.t(), self.F[indices, :])

        total = recon_l + classif_l + self.lamb * kmeans_l

        return {
            'total': total,
            'recon_loss': recon_l,
            'kmeans_loss': kmeans_l,
            'classif_loss': classif_l,
            'recon': recon,
            'z': z_real
        }


# # Example usage:
# x = torch.randn(8, 128, 1) # batch de taille 8, sequence length 128, input size 1
# enc = MultiLayerDilatedRNN(input_size=1, hidden_size=144, num_layers=4)
# y = enc(x)
# print(y.shape)   # should be (8, 128, 144)