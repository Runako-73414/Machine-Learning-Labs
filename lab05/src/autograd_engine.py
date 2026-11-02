class Value:

    def __init__(
        self,
        data,
        children=(),
        op=""
    ):

        self.data = data
        self.grad = 0.0

        self._prev = set(children)

        self._op = op

        self._backward = lambda: None

    def __repr__(self):

        return (
            f"Value(data={self.data}, "
            f"grad={self.grad})"
        )

    def __add__(self, other):

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():

            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():

            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, power):

        out = Value(
            self.data ** power,
            (self,),
            f"**{power}"
        )

        def _backward():

            self.grad += (
                power
                * (self.data ** (power - 1))
                * out.grad
            )

        out._backward = _backward

        return out

    def relu(self):

        out = Value(
            max(0, self.data),
            (self,),
            "ReLU"
        )

        def _backward():

            self.grad += (
                1.0 if out.data > 0 else 0.0
            ) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        topo = []

        visited = set()

        def build(v):

            if v not in visited:

                visited.add(v)

                for child in v._prev:

                    build(child)

                topo.append(v)

        build(self)

        self.grad = 1.0

        for node in reversed(topo):

            node._backward()


a = Value(2.0)

b = Value(3.0)

c = Value(1.0)

out = ((a * b + c) ** 2).relu()

out.backward()

print("Output =", out.data)

print("a.grad =", a.grad)

print("b.grad =", b.grad)

print("c.grad =", c.grad)

import torch

ta = torch.tensor(
    2.0,
    requires_grad=True
)

tb = torch.tensor(
    3.0,
    requires_grad=True
)

tc = torch.tensor(
    1.0,
    requires_grad=True
)

tout = torch.relu(
    ((ta * tb + tc) ** 2)
)

tout.backward()

print("\nPyTorch")

print("a.grad =", ta.grad.item())

print("b.grad =", tb.grad.item())

print("c.grad =", tc.grad.item())

print(
    "\nDifference a:",
    abs(a.grad - ta.grad.item())
)

print(
    "Difference b:",
    abs(b.grad - tb.grad.item())
)

print(
    "Difference c:",
    abs(c.grad - tc.grad.item())
)

epsilon = 1e-6

a_value = 2.0

def f(x):

    return max(
        0,
        ((x * 3.0 + 1.0) ** 2)
    )

numerical_grad = (
    f(a_value + epsilon)
    - f(a_value - epsilon)
) / (2 * epsilon)

print(
    "\nNumerical Gradient:",
    numerical_grad
)

print(
    "Autograd Gradient:",
    a.grad
)

print(
    "Difference:",
    abs(
        numerical_grad
        - a.grad
    )
)