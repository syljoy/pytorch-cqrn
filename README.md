# Convolutional Quasi-Recurrent Network (CQRN) for PyTorch

This repository contains the PyTorch implementation of the Convolutional Quasi-Recurrent Network(CQRN) proposed in paper [A Convolutional Quasi-Recurrent Network for Real-time Speech Enhancement](https://oversea.cnki.net/kcms/detail/61.1076.TN.20211207.1214.006.html).

## Requirements
Requirements are provided in `requirements.txt`.

- PyTorch 1.8 from a nightly release. Installation instructions can be found in [Pytorch](https://pytorch.org/get-started/previous-versions/#v180)


## Installation

#### From source:
1. Clone the source:

  ```git clone https://github.com/syljoy/pytorch-cqrn.git```
2. Install the Pytorch-CQRN package into python virtual environment:

  `python setup.py install` or `pip install`.

#### From PyPi:
  ```pip install Pytorch-CQRN```


## Usage

```python
import torch
from torchcqrn import CQRN

seq_len, batch_size, channels, hidden_size, bins, layers = \
        7, 3, 1, 512, 257, 5
size = (seq_len, batch_size, channels, bins)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_tensor = torch.autograd.Variable(torch.rand(size), 
                                       requires_grad=True).to(device)

cqrn = CQRN(input_dim=channels, hidden_size=hidden_size, num_layers=layers,
            dropout=0.4, window=2).to(device)
output, hidden = cqrn(input_tensor)

print(output.size(), hidden.size())
```

The full documentation for the `CQRN` is listed below:

```
CQRN(input_dim, hidden_size, num_layers, dropout, layers, save_prev_x, window, kernel_size)
    Applies a multiple layer Convolutional Quasi-Recurrent Network (CQRN) to an input sequence.
    
    Args:
        input_dim: The number of channels in the input x.
        hidden_size: The number of channels produced by the convolution.
        num_layers: The number of CQRN layers to produce.
        dropout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.0
        layers: List of preconstructed CQRN layers to use for the CQRN module.
        save_prev_x: Whether to store previous inputs for use in future masked convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the  masked convolutional window (how many previous tokens to look when computing the CQRN values). Supports 1, 2 and 3. Default: 1.
        kernel_size: The size of the convolving kernel. Default: [window, 9].
    
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
```
If you are using convolutional windows of size 2 or 3 (i.e. looking at the inputs from two previous timesteps to compute the input) and want to run over a long sequence in batches, you can set save_prev_x=True and call reset when you wish to reset the cached previous inputs.

If you want flexibility in the definition of each CQRN layer, you can construct individual `CQRNLayer` modules and pass them to the `CQRN` module using the `layers` argument.

## Maintainers

[@syljoy](https://github.com/syljoy).

## Release History
* 0.1.0
  * The first release
  * Implement simple CQRNLayer and CQRN Model.


