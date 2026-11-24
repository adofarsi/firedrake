"""Check ten SGD steps on an interval and a square that share MPI ranks."""

import firedrake as fd
import numpy as np
import torch
from firedrake.ml.pytorch import MLParameters, ml_operator


model = torch.nn.Linear(1, 1, dtype=torch.float64)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()
theta = MLParameters(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.25)

V_interval = fd.FunctionSpace(fd.UnitIntervalMesh(4), "CG", 1)
V_square = fd.FunctionSpace(fd.UnitSquareMesh(2, 2), "DG", 0)
losses = []
for V, x, target in ((V_interval, -1.0, -1.0), (V_square, 1.0, 3.0)):
    u = fd.Function(V).assign(x)
    N = ml_operator(model, V, inputs_format="pointwise")
    error = N(u, theta) - target
    losses.append(0.5 * fd.inner(error, error) * fd.dx)
dJ_dtheta = sum(fd.derivative(J, theta) for J in losses)

for epoch in range(10):
    optimizer.zero_grad()
    loss = sum(fd.assemble(J).real for J in losses)
    gradient = fd.assemble(dJ_dtheta)
    theta.backward(gradient)
    optimizer.step()

    factor = 0.5 ** epoch
    np.testing.assert_allclose(loss, 5 * factor**2)
    np.testing.assert_allclose((model.weight.grad.item(), model.bias.grad.item()), (-4 * factor, -2 * factor))
    np.testing.assert_allclose((model.weight.item(), model.bias.item()), (2 - factor, 1 - 0.5 * factor))
    if V_interval.comm.rank == 0:
        print(f"Epoch {epoch}: loss = {loss:.9f}")
