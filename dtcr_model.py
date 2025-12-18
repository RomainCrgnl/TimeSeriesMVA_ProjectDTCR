# Time series final project (Master MVA)

import torch
import torch.nn as nn
import torch.nn.functional as F

from drnn import DRNN


# class DilatedRNNLayer(nn.Module):
#     """
#     One layer of dilated RNN (LSTM/GRU/RNN)
#     """
#     def __init__(self, input_size, hidden_size, dilation=1, cell_type='GRU'):
#         super().__init__()
#         self.dilation = dilation

#         if cell_type == 'GRU':
#             self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
#         elif cell_type == 'LSTM':
#             self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
#         elif cell_type == 'RNN':
#             self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         else:
#             raise ValueError("Unknown cell_type")

#     def forward(self, x):
#         """
#         x: (batch, seq_len, input_size)
#         """
#         B, T, F = x.shape
#         d = self.dilation

#         # 1. Subsample sequence
#         # e.g. for d=2 take t=0,2,4,...
#         x_sub = x[:, ::d, :]   # (B, T_sub, F)

#         # 2. Run RNN on the subsampled sequence
#         out_sub, _ = self.rnn(x_sub)   # (B, T_sub, H)

#         # 3. Expand back to full resolution
#         out = torch.zeros(B, T, out_sub.size(2), device=x.device)

#         # fill in positions that correspond to dilated steps
#         out[:, ::d, :] = out_sub

#         # For positions in between, repeat the last known state
#         # identical to TF code: use nearest previous valid state
#         for t in range(1, T):
#             if t % d != 0:
#                 out[:, t, :] = out[:, t-1, :]

#         return out



"""
class MultiLayerDilatedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell_type='GRU'):
        super().__init__()

        self.layers = nn.ModuleList()
        current_input = input_size

        for _ in range(num_layers):
            self.layers.append(
                DRNN(current_input, hidden_size, n_layers=1, cell_type=cell_type)
            )
            current_input = hidden_size

    def forward(self, x):
        out = x
        for layer in self.layers:
            out, _ = layer(out)
        return out
"""



# class MultiLayerDilatedRNN(nn.Module):
#     def __init__(self,
#                  input_size,
#                  hidden_sizes,
#                  num_layers=3,
#                  cell_type='GRU'):
#         super().__init__()

#         # dilation schedule = 1,2,4,8,... (as in DTCR)
#         # dilations = [2**i for i in range(num_layers)]

#         # dilation schedule = 1,4,16 for 3 layers (cf article)
#         dilations = [4**i for i in range(num_layers)]

#         layers = []
#         for hidden_size, d in zip(hidden_sizes,dilations):
#             layers.append(DilatedRNNLayer(input_size, hidden_size, d, cell_type))
#             input_size = hidden_size
#         self.layers = nn.ModuleList(layers)

#     def forward(self, x):
#         """
#         x: (batch, seq_len, input_size)
#         """
#         out = x
#         for layer in self.layers:
#             out = layer(out)
#         return out  # (batch, seq_len, hidden_size)



# dtcr_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# assume drnn.DRNN is available and matches the repo API:
# model = DRNN(n_input, n_hidden, n_layers, cell_type)
from drnn import DRNN


class DTCR(nn.Module):

    def __init__(
        self,
        input_size,                      # number of features per timestep
        num_steps,                       # sequence length (T)
        cell_type='GRU',                 # or 'RNN' or 'LSTM' depending on drnn impl
        drnn_hidden_sizes=[100, 50, 50], # list of hidden sizes per layer
        bidirectional=True,               # whether to use bidirectional DRNN
        lamb=1e-3,                       # weight for kmeans loss
        class_num=10,                    # K clusters
        F_update_freq=10,                # frequency to update F in K-means loss
        d=128,                           # hidden size for the fc layer of fake classifier
        denoising=False,                 # whether to add denoising (not implemented here)
        sample_loss=False,               # if you compute sample-wise losses specially
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
        self.denoising = denoising
        self.sample_loss = sample_loss
        
        factor = 2 if bidirectional else 1
        self.embedding_size = sum(drnn_hidden_sizes) * factor # (cf article), concatenate last outputs from all encoder layers, *2 for bidirectional

        self.iteration = 0
        self.F_update_freq = F_update_freq

        # Encoder: DRNN -> linear projection to embedding
        # drnn expects input shape (seq_len, batch, input_size) per its README examples
        # self.encoder_drnn = DRNN(self.input_size, self.hidden_size, self.drnn_layers, self.cell_type)
        self.encoder_drnn = DRNN(
            n_input=self.input_size, 
            hidden_sizes=self.drnn_hidden_sizes, 
            cell_type=self.cell_type,
            bidirectional=bidirectional
        )

        # after DRNN, we'll map final hidden -> embedding (if GRU, hidden shape depends on implementation)
        # We'll map from hidden_size to embedding_size
        # self.DTCR_hidden_size = sum(self.drnn_hidden_sizes)  # concatenate last outputs from all layers (cf article)
        
        # self.enc_proj = nn.Sequential(
        #     nn.Linear(concat_size, self.embedding_size), 
        #     nn.BatchNorm1d(self.embedding_size)
        # )

        # Decoder: simple GRU decoder that decodes embedding -> sequence
        # We'll decode by repeating embedding for each time step as initial input or initialize hidden from embedding
        # self.decoder_gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.decoder_hidden_size, num_layers=1)
        # self.dec_out = nn.Linear(self.decoder_hidden_size, self.input_size)
        # self.decoder_gru = nn.GRU(
        #     input_size=self.input_size,
        #     hidden_size=self.embedding_size,
        #     num_layers=1,
        # )
        # self.dec_out = nn.Linear(self.embedding_size, self.input_size)

        self.decoder_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.embedding_size)
        self.dec_out = nn.Linear(self.embedding_size, self.input_size)

        self.fake_classifier = nn.Sequential(
            nn.Linear(self.embedding_size, self.d),
            nn.ReLU(),
            nn.Linear(self.d, 2)   # output = (real_logit, fake_logit)
        )

        # # optional classifier head (if you want supervised classification)
        # self.classifier = nn.Linear(self.embedding_size, self.K)

    def encode(self, x):
        """
        x: tensor shape (batch, seq_len, input_size)
        drnn expects (seq_len, batch, input_size)
        returns embedding z: (batch, embedding_size)
        """
        _, layer_outputs = self.encoder_drnn(x)             
        final_states = [out[-1] for out in layer_outputs]
        #The latent representation is obtained by concatenating the last hidden state output of each layer of the Dilated RNN
        h = torch.cat(final_states, dim=1) 
        #z = self.enc_proj(h)
        return h

    # def decode(self, z):
    #     """
    #     z: (batch, embedding_size)
    #     returns: recon: (batch, seq_len, input_size)
    #     """
    #     batch = z.size(0)
    #     # use z to initialize GRU hidden: need shape (num_layers, batch, hidden_size)
    #     h0 = z.unsqueeze(0).repeat(1, 1, 1)  # (1, batch, embedding_size)
    #     # map embedding to hidden_size if different
    #     if self.embedding_size != self.decoder_hidden_size:
    #         h0 = F.linear(h0, torch.eye(self.embedding_size, self.decoder_hidden_size).to(h0.device)) if False else \
    #              h0.new_zeros(1, batch, self.decoder_hidden_size)  # simpler: zero-init and let decoder learn
    #     # feed a repeated embedding (or zeros) as inputs at each timestep
    #     dec_inputs = z.unsqueeze(0).repeat(self.num_steps, 1, 1)  # (seq_len, batch, embedding_size)
    #     out_seq, _ = self.decoder_gru(dec_inputs, h0)  # (seq_len, batch, hidden_size)
    #     out_seq = out_seq.permute(1, 0, 2)  # (batch, seq_len, hidden_size)
    #     recon = self.dec_out(out_seq)  # (batch, seq_len, input_size)
    #     return recon

    def decode(self, z):

        batch_size = z.size(0)
        
        # Initialize Hidden State with z
        h = z 
        
        # Initialize Input with Zeros (Start Token)
        curr_input = torch.zeros(batch_size, self.input_size, device=z.device)
        
        recon_seq = []
        for t in range(self.num_steps):
            # Input is (Batch, 1), Hidden is (Batch, 200)
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
        Standard reconstruction MSE loss (sum or mean per sample)
        """
        # element-wise MSE
        loss = F.mse_loss(recon, x, reduction='mean')
        return loss

    # def k_means_loss(self, z):
    #     """
    #     z: (batch, embedding_size)
    #     cluster_centers: (K, embedding_size)
    #     returns scalar: mean min squared distance per sample
    #     Also returns cluster assignments if needed.
    #     """
    #     # compute squared distances between each z and each center:
    #     # z: (B, E), centers: (K, E) -> dists: (B, K)
    #     # use (a-b)^2 = a^2 + b^2 - 2ab
    #     z2 = (z**2).sum(dim=1, keepdim=True)  # (B,1)
    #     c2 = (self.cluster_centers**2).sum(dim=1).unsqueeze(0)  # (1,K)
    #     cross = z @ self.cluster_centers.t()  # (B,K)
    #     dists = z2 + c2 - 2*cross  # (B,K)
    #     # nearest cluster squared distance
    #     min_dists, assignments = torch.min(dists, dim=1)  # (B,)
    #     loss = min_dists.mean()
    #     return loss, assignments

    def kmeans_loss(self, H, F): #giving self is useless here but avoid errors
        """
        H: (batch, embedding_size)
        F: (batch, K)
        """
        # print(H.shape, F.shape)
        # compute FF^T
        FFt = F @ F.t()        # (N, N)
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
        Convenience: pass input x (batch,seq_len,input_size) and optional labels.
        Returns dict of losses + recon and z for diagnostics.
        """
        recon, z_real = self.forward(x_real)
        _, z_fake = self.forward(x_fake)
        z_all = torch.cat([z_real, z_fake], dim=0)

        real_fake_labels = torch.cat([
            torch.tensor([[1., 0.]]).repeat(len(x_real), 1),   # real
            torch.tensor([[0., 1.]]).repeat(len(x_fake), 1)    # fake
        ], dim=0).to(x_real.device)

        recon_l = self.recon_loss(x_real, recon)
        classif_l = self.classif_loss(z_all, real_fake_labels)
        if self.lamb == 0: # avoid computing kmeans loss if not used
            kmeans_l = torch.tensor(0.0, device=x_real.device)
        else:
            kmeans_l = self.kmeans_loss(z_real.t(), self.F[indices, :])

        total = recon_l + classif_l + self.lamb/2 * kmeans_l

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