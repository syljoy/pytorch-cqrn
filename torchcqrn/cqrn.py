# -*- coding: utf-8 -*-
#
# @Time    : 2022-04-15 10:42:10
# @Author  : Yunlong Shi
# @Email   : syljoy@163.com
# @FileName: cqrn.py
# @Software: PyCharm
# @Github  : https://github.com/syljoy
# @Desc    : CQRN


import torch
from torch import nn
import torch.nn.functional as function
from torch.autograd import Variable


class CQRNLayer(nn.Module):
    r"""Applies a single layer Convolutional Quasi-Recurrent Network (CQRN)
    to an input sequence.

    .. math::

        Z_t = \mathrm{ELU}(W_^1_Z*x_{t-k+1}+W_^2_Z*x_{t-k+2}+...+W_^k_Z*x_t)

        F_t = \sigma(W_^1_F*x_{t-k+1}+W_^2_F*x_{t-k+2}+...+W_^k_F*x_t)

        O_t = \sigma(W_^1_O*x_{t-k+1}+W_^2_O*x_{t-k+2}+...+W_^k_O*x_t)

        C_t = F_t\bigodot C_{t-1} + (1-F_t) \bigodot Z_t

        H_t = O_t \bigodot C_t

    where :math:`\mathrm{ELU}` is the ELU function, :math:`\sigma` is the
    sigmoid function, and :math:`\bigodot` is the Hadamard product.

    Args:
        input_dim: The number of channels in the input x.
        hidden_size: The number of channels produced by the convolution.
        save_prev_x: Whether to store previous inputs for use in future
            masked convolutional windows (i.e. for a continuing sequence such
            as in language modeling). If true, you must call reset to remove
            cached previous values of x. Default: False.
        window: Defines the size of the  masked convolutional window (how many
            previous tokens to look when computing the CQRN values). Supports
            1, 2 and 3. Default: 1.
        kernel_size: The size of the convolving kernel. Default: [window, 9].

    Inputs: input_data, hidden
        - **input_data** of shape `(seq_len, batch, channels, bins)`: tensor
          containing the features of the input sequence.
        - **hidden** of shape `(hidden_size, batch, bins)`: tensor containing
          the initial hidden state for the CQRN.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, hidden_size, bins)`: tensor
          containing the output of the CQRN for each timestep.
        - **h_n** of shape `(1, batch, hidden_size, bins)`: tensor containing the
          hidden state for t=seq_len
    """

    def __init__(self, input_dim, hidden_size, save_prev_x=False, window=1,
                 kernel_size=None, ):
        super(CQRNLayer, self).__init__()
        assert window in [1, 2, 3], "This CQRN implementation currently only " \
                                    "handles masked convolutional window of " \
                                    "size 1, size 2 or size 3 "

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.save_prev_x = save_prev_x
        self.window = window
        self.kernel_size = kernel_size if kernel_size else [window, 9]
        self.padding = (0, self.kernel_size[1] // 2)

        self.prevX = None
        self.conv = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=3 * self.hidden_size,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def reset(self):
        # If you are saving the previous value of x, you should call this
        # when starting with a new state
        self.prevX = None

    def set_save_prev_x(self, save_prev):
        # If you are saving the previous value of x, you should call this
        # when starting with a new state
        self.save_prev_x = save_prev

    def forward(self, input_data, hidden=None):
        seq_len, batch_size, channels, bins = input_data.size()
        source = None
        if self.window == 1:
            source = input_data.view(seq_len * batch_size, channels,
                                     self.window, bins)
        elif self.window == 2:
            temp0 = [self.prevX if self.prevX is not None else
                     input_data[:1, :, :] * 0]
            if len(input_data) > 1:
                temp0.append(input_data[:-1, :, :])
            temp0 = torch.cat(temp0, 0).view(seq_len, batch_size, channels, 1,
                                             bins)
            input_data_view = input_data.view(seq_len, batch_size, channels, 1,
                                              bins)
            source = torch.cat([temp0, input_data_view], 3)
            source = source.view(seq_len * batch_size, channels, self.window,
                                 bins)
        elif self.window == 3:
            temp00 = [self.prevX if self.prevX is not None else
                      input_data[:2, :, :] * 0]
            if len(input_data) > 2:
                temp00.append(input_data[:-2, :, :])
            temp00 = torch.cat(temp00, 0).view(seq_len, batch_size, channels,
                                               1, bins)
            temp0 = [self.prevX[1:] if self.prevX is not None else
                     input_data[:1, :, :] * 0]
            if len(input_data) > 1:
                temp0.append(input_data[:-1, :, :])
            temp0 = torch.cat(temp0, 0).view(seq_len, batch_size, channels, 1,
                                             bins)
            input_data_view = input_data.view(seq_len, batch_size, channels, 1,
                                              bins)
            source = torch.cat([temp00, temp0, input_data_view], 3)
            source = source.view(seq_len * batch_size, channels, self.window,
                                 bins)
        Y = self.conv(source)
        Zt, Ft, Ot = Y.chunk(3, dim=1)
        Zt = function.elu(Zt)
        Ft = torch.sigmoid(Ft)
        Ot = torch.sigmoid(Ot)
        Zt = Zt.squeeze().view(seq_len, batch_size, -1, bins)
        Ft = Ft.squeeze().view(seq_len, batch_size, -1, bins)
        Ot = Ot.squeeze().view(seq_len, batch_size, -1, bins)

        Zt = Zt.contiguous()
        Ft = Ft.contiguous()

        # fo-pooling
        C = CPUForgetMult()(Ft, Zt, hidden)

        H = Ot * C
        if self.window == 2 and self.save_prev_x:
            self.prevX = Variable(input_data[-1:, :, :].data,
                                  requires_grad=False)
        elif self.window == 3 and self.save_prev_x:
            if len(input_data) > 1:
                self.prevX = Variable(input_data[-2:, :, :].data,
                                      requires_grad=False)
            elif len(input_data) == 1 and self.prevX is not None:
                self.prevX = torch.cat((self.prevX[-1:, :, :],
                                        Variable(input_data.data,
                                                 requires_grad=False)), 0)
            else:
                self.prevX = torch.cat((torch.zeros(input_data.size()),
                                        Variable(input_data.data,
                                                 requires_grad=False)), 0)

        return H, C[-1:, :, :, :]


class CQRN(nn.Module):
    r"""Applies a multiple layer Convolutional Quasi-Recurrent Network (CQRN)
    to an input sequence.

    Args:
        input_dim: The number of channels in the input x.
        hidden_size: The number of channels produced by the convolution.
        num_layers: The number of CQRN layers to produce.
        dropout: Whether to apply zoneout (i.e. failing to update elements in
            the hidden state) to the hidden state updates. Default: 0.0
        layers: List of preconstructed CQRN layers to use for the CQRN module.

    Inputs: input, hidden
        - **input** of shape `(seq_len, batch, channels, bins)`: tensor
          containing the features of the input sequence.
        - **hidden** of shape `(hidden_size, batch, bins)`: tensor containing
          the initial hidden state for the CQRN.

    Outputs: output, next_hidden
        - **output** of shape `(seq_len, batch, hidden_size, bins)`: tensor
          containing the output of the CQRN for each timestep.
        - **next_hidden** of shape `(layers, batch, hidden_size, bins)`:
          tensor containing the hidden state for t=seq_len.
    """

    def __init__(self, input_dim, hidden_size, num_layers=1, dropout=0.0,
                 layers=None, **kwargs):
        super(CQRN, self).__init__()

        self.layers = nn.ModuleList(
            layers if layers else [
                CQRNLayer(input_dim if l == 0 else hidden_size, hidden_size,
                          **kwargs) for l in
                range(num_layers)])

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.dropout = dropout

    def reset(self):
        r"""If your masked convolutional window is greater than 1, you must reset
        at the beginning of each new sequence """
        [layer.reset() for layer in self.layers]

    def set_save_prev_x(self, save_prev_x=True):
        [layer.set_save_prev_x(save_prev_x) for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []
        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i])
            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout,
                                                    training=self.training,
                                                    inplace=False)
        next_hidden = torch.cat(next_hidden, 0)
        return input, next_hidden


class CPUForgetMult(torch.nn.Module):
    r"""ForgetMult computes a simple recurrent equation on the CPU
    :math:`C_t = F_t * Z_t + (1 - F_t) * C_{t-1}`
    This equation is equivalent to dynamic weighted averaging.

    Inputs: F, X, hidden
        - **X** of shape `(seq_len, batch_size, channels, bins)`: tensor
          containing the features of the input sequence.
        - **F** of shape `(seq_len, batch_size, channels, bins)`: tensor
          containing the forget gate values, assumed in range [0, 1].
        - **hidden_init** of shape `(hidden_size, batch, bins)`: tensor
          containing the initial hidden state for the recurrence
          (:math:`h_{t-1}`).
    """
    def __init__(self):
        super(CPUForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None:
                h = h + (1 - forgets[i]) * prev_h
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        return torch.stack(result)


if __name__ == '__main__':
    seq_len, batch_size, channels, hidden_size, bins, layers = \
        7, 3, 1, 512, 257, 5
    size = (seq_len, batch_size, channels, bins)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # (seq_len, batch_size, channels, bins)
    input_tensor = Variable(torch.rand(size), requires_grad=True).to(device)
    cqrn = CQRN(input_dim=channels, hidden_size=hidden_size, num_layers=layers,
                dropout=0.4, window=2).to(device)
    output, hidden = cqrn(input_tensor)
    assert list(output.size()) == [seq_len, batch_size, hidden_size, bins]
    assert list(hidden.size()) == [layers, batch_size, hidden_size, bins]













