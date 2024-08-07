
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        """
        Initializes an instance of the class.

        Parameters
        ----------
        data : object
            The data stored in the instance.
        _children : tuple, optional
            The child instances (default is an empty tuple).
        _op : str, optional
            The operation associated with the instance (default is an empty string).

        Attributes
        ----------
        grad : int
            The gradient of the instance, initialized to 0.
        _backward : callable
            A function that performs the backward operation, initialized to a no-op function.
        _prev : set
            A set of previous instances, initialized with the provided child instances.
        _op : str
            The operation associated with the instance.
        """
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        """
        Adds two values element-wise.

        Args:
            other (Value or Any): The value to add to this value. If not a Value instance, it will be converted to a Value.

        Returns:
            Value: A new Value instance containing the sum of this value and the other value.

        Notes:
            The gradients of this value and the other value will be updated accordingly in the backward pass.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            """
            Accumulates gradients of the output tensor to the input tensors.

            .. note::
               This method is intended for internal use and should not be called directly.

            Parameters
            ----------
            None

            Returns
            -------
            None

            Notes
            -----
            This method modifies the :attr:`grad` attribute of the instance and another object in place. It is a part of the backpropagation process in a neural network.
            """
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Multiplies the current instance with another value.

        Parameters
        ----------
        other : Value or any number type
            The value to be multiplied with the current instance.

        Returns
        -------
        Value
            A new instance representing the result of the multiplication.

        Notes
        -----
        The multiplication operation is recorded and can be used for automatic differentiation.
        The gradients of the current instance and the other value are updated accordingly.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            """
            Computes the gradients of the operation with respect to both operands.

            Updates the :attr:`grad` attribute of both the current object and the :attr:`other` object by accumulating the product of the :attr:`data` attribute of one object and the :attr:`grad` attribute of the :attr:`out` object.

            This method is intended to be used as part of the backpropagation algorithm.
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Element-wise power operation.

        Raises the current value to the power of ``other``.

        Args:
            other (int or float): The exponent.

        Returns:
            Value: The result of the power operation.

        Raises:
            AssertionError: If ``other`` is not an integer or float.

        Notes:
            Currently only supports integer and float exponents.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """
        Applies the ReLU activation function to the input value.

        The ReLU function returns 0 for negative input values and the input value itself for non-negative inputs.

        Returns:
            Value: The output of the ReLU function applied to the input value.

        Note:
            This function also sets up the backward pass for the ReLU operation by defining the _backward function.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        """
        Computes the gradients of the output with respect to the inputs in reverse topological order.

        Performs a depth-first search to build the topological ordering, then traverses the graph in reverse order to compute the gradients.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This function modifies the :attr:`grad` attribute of the current instance and its predecessors in the computation graph. After execution, the :attr:`grad` attribute contains the gradient of the output with respect to each node.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
