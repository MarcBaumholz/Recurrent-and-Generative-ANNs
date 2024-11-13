import einops
import torch


class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_to_hidden = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_to_hidden = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h):
        """
        Performs the forward pass of the RNN cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h (Tensor): Hidden state of shape (batch_size, hidden_size).

        Returns:
            Tensor: Hidden state of shape (batch_size, hidden_size).
        """
        return torch.tanh(self.input_to_hidden(x) + self.hidden_to_hidden(h))


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #self.rnn_cell = torch.nn.RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.rnn_cell = RNNCell(input_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

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

        # Process each time step
        for t in range(sequence_length):
            h = self.rnn_cell(x[:, t, :], h)

        return self.output_layer(h)
    
class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Single linear layer to compute all gates simultaneously
        self.linear = torch.nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h):
        """
        Performs the forward pass of the LSTM cell. Computes a single time step.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).
            h Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).

        Returns:
            Tuple(Tensor): Hidden and cell state of shape (batch_size, hidden_size).
        """
        h_prev, c_prev = h

        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)

        # Compute all gate activations in one go
        gates = self.linear(combined)

        # Split the gates into their respective parts
        i_t, f_t, g_t, o_t = torch.chunk(gates, 4, dim=1)

        # Apply activations
        i_t = torch.sigmoid(i_t) # Input
        f_t = torch.sigmoid(f_t) # Forget
        g_t = torch.tanh(g_t) # Cell
        o_t = torch.sigmoid(o_t) # Output
        
        # Update cell state and hidden state
        c_next = f_t * c_prev + i_t * g_t
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        #self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the LSTM. Computes a whole sequence.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        batch_size, sequence_length, _ = x.size()
        
        # Initialize hidden and cell states to zeros
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Process each time step in the sequence
        for t in range(sequence_length):
            h, c = self.lstm_cell(x[:, t, :], (h, c))

        return self.output_layer(h)


class Conv1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize the convolution weights and biases
        self.conv = torch.nn.Linear(input_size * kernel_size, hidden_size)

    def forward(self, x):
        """
        Performs a one-dimensional convolution.

        Args:
            x (Tensor): Input of shape (batch_size, input_size, sequence_length).

        Returns:
            Tensor: Output of shape (batch_size, hidden_size, sequence_length).
        """
        batch_size, input_size, sequence_length = x.size()

        # Apply padding
        padding = (self.kernel_size - 1)
        x = torch.nn.functional.pad(x, (padding, 0))

        # Extract patches, new shape: (batch_size, sequence_length, input_size * kernel_size)
        x = x.unfold(dimension=2, size=self.kernel_size, step=self.stride)

        # Reshape to (batch_size, sequence_length, input_size * kernel_size)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, -1, input_size * self.kernel_size)

        # Apply linear weights and biases to each patch
        output = self.conv(x)

        # Transpose to get output in shape (batch_size, hidden_size, sequence_length)
        return output.transpose(1, 2)

class TCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.num_layers = 3
        self.convs = torch.nn.ModuleList()

        for i in range(self.num_layers):
            #self.convs.append(torch.nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size=3, stride=3))
            self.convs.append(Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size=3, stride=3))

        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the TCN.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).
        """
        # Reorder dimensions: (batch, channels, sequence)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = torch.relu(x)
        # Take the last output in the sequence
        x = x[:, :, -1]

        return self.output_layer(x)
