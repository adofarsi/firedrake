"""Train one model on two independent mesh communicators.

Run with a real Firedrake build and CPU PyTorch using
mpiexec -n 3 python train_two_meshes.py. The interval uses one rank
and the square uses two. The objective is the sum of the two losses.
"""

import firedrake as fd
from firedrake.ml.pytorch import MLParameters, ml_operator
from mpi4py import MPI
import numpy as np
import torch


def main() -> None:
    training = MPI.COMM_WORLD
    if training.size < 2:
        raise ValueError("Run with at least two MPI ranks.")
    interval = training.rank == 0
    group = training.Split(int(not interval), training.rank)
    try:
        mesh = fd.UnitIntervalMesh(4, comm=group) if interval else fd.UnitSquareMesh(2, 2, comm=group)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.Function(V).assign(1 if interval else 2)
        model = torch.nn.Linear(1, 1, dtype=torch.float64)
        with torch.no_grad():
            model.weight.fill_(1)
            model.bias.zero_()

        # Both handles reference the same live tensors on this process.
        theta = MLParameters(model.parameters(), comm=training)
        local_theta = MLParameters(model.parameters(), comm=group)
        transfer = fd.ScalarSpaceTransfer(theta.function_space(), local_theta.function_space())
        local_theta.assign(transfer.replicate(theta.copy()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        prediction = ml_operator(model, V, inputs_format="pointwise")(u, local_theta)
        J = 0.5 * fd.inner(prediction, prediction) * fd.dx
        mesh_loss = fd.assemble(J).real
        loss = training.allreduce(mesh_loss if group.rank == 0 else 0)
        gradient = transfer.adjoint(fd.assemble(fd.derivative(J, local_theta)))

        optimizer.zero_grad()
        theta.backward(gradient)
        optimizer.step()

        np.testing.assert_allclose((loss, model.weight.grad.item(), model.bias.grad.item()), (2.5, 5, 3))
        np.testing.assert_allclose((model.weight.item(), model.bias.item()), (0.5, -0.3))
        if training.rank == 0:
            print(f"Total loss {loss:g}, weight {model.weight.item():g}, bias {model.bias.item():g}")
    finally:
        group.Free()


if __name__ == "__main__":
    main()
