import torch
import torch.nn as nn

# Define custom LinearFunction
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # Save input and weight for backward
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # TODO Implement backward pass here TODO

        # Compute gradients
        grad_input = grad_output.mm(weight)  # Gradient wrt input
        grad_weight = grad_output.t().mm(input)  # Gradient wrt weight
        grad_bias = grad_output.sum(0)  # Gradient wrt bias

        return grad_input, grad_weight, grad_bias

# Define custom ReLUFunction
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None

        # TODO Implement backward pass here TODO
        
        # Compute gradient wrt input
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0  # Zero out gradient where input was negative or zero

        return grad_input


# Define the FeedForwardNetwork using the custom functions
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=32):
        super(FeedForwardNetwork, self).__init__()
        # Initialize weights and biases as parameters
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1_bias = nn.Parameter(torch.randn(hidden_size))

        self.fc2_weight = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.randn(hidden_size))

        self.fc3_weight = nn.Parameter(torch.randn(1, hidden_size))
        self.fc3_bias = nn.Parameter(torch.randn(1))

        # Apply Xavier initialization to the weights
        nn.init.xavier_uniform_(self.fc1_weight)
        nn.init.xavier_uniform_(self.fc2_weight)
        nn.init.xavier_uniform_(self.fc3_weight)

        # For biases, we usually initialize to zero
        nn.init.constant_(self.fc1_bias, 0)
        nn.init.constant_(self.fc2_bias, 0)
        nn.init.constant_(self.fc3_bias, 0)

    def forward(self, x):
        out = LinearFunction.apply(x, self.fc1_weight, self.fc1_bias)
        out = ReLUFunction.apply(out)
        out = LinearFunction.apply(out, self.fc2_weight, self.fc2_bias)
        out = ReLUFunction.apply(out)
        out = LinearFunction.apply(out, self.fc3_weight, self.fc3_bias)
        return out