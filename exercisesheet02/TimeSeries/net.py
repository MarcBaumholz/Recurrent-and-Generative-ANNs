import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x, h):
        """
        Performs the forward pass of the RNN cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h (Tensor): Hidden state of shape (batch_size, hidden_size).

        Returns:
            Tensor: Hidden state of shape (batch_size, hidden_size).
        """
        # Compute new hidden state
        h_new = self.activation(self.i2h(x) + self.h2h(h))
        return h_new


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the RNN. Computes a whole sequence.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        batch_size, sequence_length, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(sequence_length):
            h = self.rnn_cell(x[:, t, :], h)  # Update hidden state for each time step

        # Pass the final hidden state to the output layer
        output = self.fc(h)
        return output



class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size * 4)
        self.h2h = nn.Linear(hidden_size, hidden_size * 4)

    def forward(self, x, h):
        """
        Performs the forward pass of the LSTM cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).

        Returns:
            Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).
        """
        h_state, c = h
        gates = self.i2h(x) + self.h2h(h_state)
        i, f, o, g = gates.chunk(4, dim=1)  # Split the gates

        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell gate

        c = f * c + i * g  # New cell state
        h_state = o * torch.tanh(c)  # New hidden state
        return h_state, c


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the LSTM. Computes a whole sequence.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        batch_size, sequence_length, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(sequence_length):
            h, c = self.lstm_cell(x[:, t, :], (h, c))

        # Pass the final hidden state to the output layer
        output = self.fc(h)
        return output


class Conv1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Linear(input_size * kernel_size, hidden_size)

    def forward(self, x):
        """
        Performs a one-dimensional convolution.

        Args:
            x (Tensor): Input of shape (batch_size, input_size, sequence_length).

        Returns:
            Tensor: Output of shape (batch_size, hidden_size, sequence_length).
        """
        batch_size, input_size, sequence_length = x.size()
        padding = (self.kernel_size - 1)  # Equivalent to "same" padding
        x = nn.functional.pad(x, (padding, 0))

        outputs = []
        for i in range(0, sequence_length, self.stride):
            if i + self.kernel_size > x.size(2):
                break
            x_chunk = x[:, :, i:i+self.kernel_size]
            x_chunk = x_chunk.contiguous().view(batch_size, -1)
            outputs.append(self.conv(x_chunk))

        return torch.stack(outputs, dim=2)  # Shape: (batch_size, hidden_size, new_sequence_length)



class TCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = Conv1d(input_size, hidden_size, 3, 1)
        self.conv2 = Conv1d(hidden_size, hidden_size, 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the TCN.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        # Transpose x to (batch_size, input_size, sequence_length) for convolution
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)

        # Take the last time step output and pass through the fully connected layer
        x = x[:, :, -1]
        output = self.fc(x)
        return output
