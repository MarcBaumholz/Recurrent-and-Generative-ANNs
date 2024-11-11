import torch
import torch.nn as nn
from torch.autograd import gradcheck
from net import LinearFunction, ReLUFunction

# Test for LinearFunction
def test_linear_function():
    """
    Test the custom LinearFunction by comparing its gradients with those computed by PyTorch's autograd.
    """

    # Input dimensions
    batch_size = 2
    input_features = 3
    output_features = 4

    # Create random input, weight, and bias tensors with requires_grad=True
    # TODO

    # Create random input, weight, and bias tensors with requires_grad=True
    input = torch.randn(batch_size, input_features, dtype=torch.double, requires_grad=True)
    weight = torch.randn(output_features, input_features, dtype=torch.double, requires_grad=True)
    bias = torch.randn(output_features, dtype=torch.double, requires_grad=True)


    # Wrap tensors in tuples for gradcheck
    func = LinearFunction.apply
    test = gradcheck(func, (input, weight, bias), eps=1e-6, atol=1e-4)
    print('LinearFunction gradcheck passed:', test)

# Edge case test for LinearFunction with zero inputs
def test_linear_function_zero_input():
    """
    Test LinearFunction with zero inputs to ensure gradients are computed correctly.
    """
    batch_size = 2
    input_features = 3
    output_features = 4

    # Create zero input tensor with requires_grad=True and random weight and bias tensors
    # TODO

    # Create zero input tensor with requires_grad=True and random weight and bias tensors
    input = torch.zeros(batch_size, input_features, dtype=torch.double, requires_grad=True)
    weight = torch.randn(output_features, input_features, dtype=torch.double, requires_grad=True)
    bias = torch.randn(output_features, dtype=torch.double, requires_grad=True)


    func = LinearFunction.apply
    test = gradcheck(func, (input, weight, bias), eps=1e-6, atol=1e-4)
    print('LinearFunction zero input gradcheck passed:', test)

# Edge case test for LinearFunction with zero weights
def test_linear_function_zero_weight():
    """
    Test LinearFunction with zero weights to ensure gradients are computed correctly.
    """
    batch_size = 2
    input_features = 3
    output_features = 4

    # Create random input tensor with requires_grad=True and zero weight tensor and random bias tensor
    # TODO

    # Create random input tensor with requires_grad=True and zero weight tensor and random bias tensor
    input = torch.randn(batch_size, input_features, dtype=torch.double, requires_grad=True)
    weight = torch.zeros(output_features, input_features, dtype=torch.double, requires_grad=True)
    bias = torch.randn(output_features, dtype=torch.double, requires_grad=True)


    func = LinearFunction.apply
    test = gradcheck(func, (input, weight, bias), eps=1e-6, atol=1e-4)
    print('LinearFunction zero weight gradcheck passed:', test)

# Test for ReLUFunction
def test_relu_function():
    """
    Test the custom ReLUFunction by comparing its gradients with those computed by PyTorch's autograd.
    """
    # Create random input tensor with requires_grad=True
    # TODO

    # Inputs slightly above and below zero (add requires_grad=True)
    input = torch.tensor([[-1e-6, 0, 1e-6], [-0.1, 0.0, 0.1]], dtype=torch.double, requires_grad=True)
    

    # Use gradcheck to test gradients
    func = ReLUFunction.apply
    test = gradcheck(func, (input,), eps=1e-6, atol=1e-4)
    print('ReLUFunction gradcheck passed:', test)

def test_relu_function_near_zero():
    """
    Test ReLUFunction with inputs close to zero but not exactly zero to avoid non-differentiable points.
    """
    # Inputs slightly above and below zero (add requires_grad=True)
    # TODO

    # Inputs slightly above and below zero (add requires_grad=True)
    input = torch.tensor([[-1e-6, 0, 1e-6], [-0.1, 0.0, 0.1]], dtype=torch.double, requires_grad=True)
    

    
    func = ReLUFunction.apply
    test = gradcheck(func, (input,), eps=1e-6, atol=1e-4)
    print('ReLUFunction near zero input gradcheck passed:', test)

# Edge case test for ReLUFunction with negative input
def test_relu_function_negative_input():
    """
    Test ReLUFunction with negative inputs to ensure gradients are zero as expected.
    """
    # Negative input tensor with requires_grad=True
    # TODO

    # Negative input tensor with requires_grad=True
    input = torch.randn(10, 5, dtype=torch.double, requires_grad=True) * -1  # Ensure negative values


    func = ReLUFunction.apply
    test = gradcheck(func, (input,), eps=1e-6, atol=1e-4)
    print('ReLUFunction negative input gradcheck passed:', test)

# Run all tests
if __name__ == '__main__':
    print("Testing LinearFunction...")
    test_linear_function()
    test_linear_function_zero_input()
    test_linear_function_zero_weight()

    print("\nTesting ReLUFunction...")
    test_relu_function()
    test_relu_function_near_zero()
    test_relu_function_negative_input()


