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
    def __init__(self, input_size, hidden_size, kernel_size, stride, dilation):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # Initialize the convolution weights and biases
        self.linear = torch.nn.Linear(input_size * kernel_size, hidden_size)

    def forward(self, x):
        """
        Performs a one-dimensional convolution.

        Args:
            x (Tensor): Input of shape (batch_size, input_size, sequence_length).

        Returns:
            Tensor: Output of shape (batch_size, hidden_size, sequence_length).
        """
        batch_size, input_size, sequence_length = x.shape

        # Apply padding
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        if(input_size % effective_kernel_size != 0):
            padding = effective_kernel_size -1
            x = torch.nn.functional.pad(x, (0, padding))

        # Unfold input
        x_unfolded = x.unfold(dimension=2, size=self.kernel_size, step=self.stride)

        # Select the appropriate elements in the output sequence with dilation
        x_unfolded = x_unfolded[:, :, ::self.dilation,:]  

        # Reshape from [batch_size, input_size, output_length, kernel_size] to [batch_size * output_length, input_size * kernel_size]
        x_unfolded = einops.rearrange(x_unfolded, "b i s k -> (b s) (i k)")

        # Apply linear weights and biases to each patch
        output = self.linear(x_unfolded)

        # Rearrange to (batch_size, hidden_size, sequence_length)
        return einops.rearrange(output, "(b s) h -> b h s", b=batch_size)

class TCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        dilation_factor = 1
        self.num_layers = 4 if dilation_factor == 1 else 3
        self.convs = torch.nn.ModuleList()

        for i in range(self.num_layers):
            dilation = dilation_factor ** i
            #self.convs.append(torch.nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size=3, stride=3, dilation=dilation))
            self.convs.append(Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size=3, stride=3, dilation=dilation))
          
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
            #x = torch.relu(x)
            #print(x.shape)
        # Take the last output in the sequence (should only have one element if num_layers is correct)
        x = x[:, :, -1]

        return self.output_layer(x)
