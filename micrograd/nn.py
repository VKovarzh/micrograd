import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        """
        Initializes the object.

        Parameters
        ----------
        nin : int
            Number of input values.
        nonlin : bool, optional
            Whether to apply non-linearity (default is True).

        Attributes
        ----------
        w : list of Value
            Weights for the input values.
        b : Value
            Bias value.
        nonlin : bool
            Whether non-linearity is applied.
        """
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """
        Calls the layer on input data.

        Parameters
        ----------
        x : iterable
            Input data to the layer.

        Returns
        -------
        output
            Activated output if non-linearity is applied, otherwise the linear combination of weights, input, and bias.

        Notes
        -----
        If ``nonlin`` attribute is set, the output is passed through a ReLU activation function. Otherwise, the output is the direct result of the linear combination.
        """
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        Calls the neurons in this layer on the input data.

        Parameters
        ----------
        x : object
           Input data to be processed by the neurons.

        Returns
        -------
        object
           Output of the neuron(s). If this layer consists of a single neuron, 
           the output is returned directly; otherwise, a list of outputs is returned.
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        """
        Initializes the object with input and output layer sizes.

        Parameters
        ----------
        nin : int
            Number of inputs to the network.
        nouts : list of int
            List of output layer sizes.

        Notes
        -----
         Initializes a series of layers with the specified input and output sizes, 
         applying non-linearity to all but the last layer.
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Calls the neural network on input data.

        Parameters
        ----------
        x : input data
            Data to pass through the network.

        Returns
        -------
        output : processed data
            Result of passing the input data through all layers of the network.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
