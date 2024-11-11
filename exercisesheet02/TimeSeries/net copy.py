import einops
import torch


class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        pass

    def forward(self, x, h):
        """
        Performs the forward pass of the RNN cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h (Tensor): Hidden state of shape (batch_size, hidden_size).

        Returns:
            Tensor: Hidden state of shape (batch_size, hidden_size).
        """
        pass


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        pass

    def forward(self, x):
        """
        Performs the forward pass of the RNN. Computes a whole sequence.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        pass


class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        pass

    def forward(self, x, h):
        """
        Performs the forward pass of the LSTM cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).

        Returns:
            Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).
        """
        pass


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        pass

    def forward(self, x):
        """
        Performs the forward pass of the LSTM. Computes a whole sequence.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        pass


class Conv1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super().__init__()
        pass

    def forward(self, x):
        """
        Performs a one-dimensional convolution.

        Args:
            x (Tensor): Input of shape (batch_size, input_size, sequence_length).

        Returns:
            Tensor: Output of shape (batch_size, hidden_size, sequence_length).
        """
        pass


class TCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        pass

    def forward(self, x):
        """
        Performs the forward pass of the TCN.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        pass
