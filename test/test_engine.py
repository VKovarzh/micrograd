import torch
from micrograd.engine import Value

def test_sanity_check():

    """
    Performs a sanity check on the Value class by comparing its behavior with PyTorch.

    This function verifies that the Value class, which is designed to mimic the behavior of PyTorch tensors, produces the same results as PyTorch for a given set of operations. The operations performed include basic arithmetic, ReLU activation, and backpropagation. The function asserts that the results and gradients produced by the Value class match those produced by PyTorch.

    Parameters
    ----------

    None

    Returns
    -------

    None

    Raises
    ------

    AssertionError
        If the results or gradients produced by the Value class do not match those produced by PyTorch.
    """
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    """
    Tests various arithmetic operations on Value objects, comparing the results with PyTorch tensors.

    Performs a series of operations, including addition, multiplication, exponentiation, and division, on Value objects and PyTorch tensors. Then, compares the results and gradients of the Value objects with the corresponding PyTorch tensors.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The test uses a tolerance of 1e-06 to compare the results and gradients. If the absolute difference between the two is less than the tolerance, the test passes. Otherwise, it fails.
    """
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
