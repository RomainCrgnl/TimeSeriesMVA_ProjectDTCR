# drnn implementation using pytorch
# from https://github.com/zalandoresearch/pytorch-dilated-rnn/blob/master/drnn.py and modified as needed

import torch
import torch.nn as nn


use_cuda = torch.cuda.is_available()

class DRNN(nn.Module):

    def __init__(self, n_input, hidden_sizes, dropout=0, cell_type='GRU', bidirectional=True, batch_first=True):
        super(DRNN, self).__init__()

        self.n_layers = len(hidden_sizes)

        # dilation schedule = 1, 4, 16 for 3 layers (cf article)
        self.dilations = [4**i for i in range(self.n_layers)] 

        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        # Iterate through the provided hidden sizes to build the stack
        current_input_dim = n_input
        for h_size in hidden_sizes:
            layers.append(cell(current_input_dim, h_size, dropout=dropout, bidirectional=self.bidirectional))
            # The output dimension of this layer is the input for the next
            current_input_dim = h_size * (2 if self.bidirectional else 1)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs # return last layer outputs and all layers outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            # FIX: Handle bidirectional initialization
            num_directions = 2 if self.bidirectional else 1
            
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                # Repeat for directions: (1, B, H) -> (num_directions, B, H)
                hidden = (c.unsqueeze(0).repeat(num_directions, 1, 1), 
                          m.unsqueeze(0).repeat(num_directions, 1, 1))
            else:
                # Repeat for directions: (1, B, H) -> (num_directions, B, H)
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0).repeat(num_directions, 1, 1)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        if use_cuda:
            hidden = hidden.cuda()
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            if use_cuda:
                memory = memory.cuda()
            return (hidden, memory)
        else:
            return hidden
        
