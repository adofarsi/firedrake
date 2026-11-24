"""One SGD step across two mesh communicators using Firedrake PR #5669.

Use a real, double-precision Firedrake build and CPU PyTorch. Run with
`mpiexec -n 3 python train_two_meshes.py` to give the interval one rank and
the square two ranks.
"""

import firedrake as fd
import numpy as np
import torch
from mpi4py import MPI


def sum_losses_across_rank_groups(
    group_loss: float,
    group_gradient: fd.Cofunction,
    group_comm: MPI.Comm,
    training_comm: MPI.Comm,
) -> tuple[float, fd.Cofunction]:
    """Sum each MPI rank group's loss and weight gradient exactly once.

    Call on every rank of ``training_comm``. The groups must partition that
    communicator. Each rank supplies its group's assembled totals. All groups
    use replicas of the same model with matching parameter order, shapes and dtypes.
    """

    representative = group_comm.rank == 0
    loss = training_comm.allreduce(group_loss if representative else 0.0, op=MPI.SUM)
    gradient = fd.Cofunction(group_gradient.function_space())
    buffer = gradient.dat.data_wo
    buffer[:] = group_gradient.dat.data_ro if representative else 0
    training_comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
    return loss, gradient


def main() -> None:
    from firedrake.ml.pytorch import MLParameters, ml_operator

    training_comm = MPI.COMM_WORLD
    if training_comm.size < 2:
        raise ValueError("Run with at least two MPI ranks.")

    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()
    theta = MLParameters(model.parameters())
    layout = tuple((name, tuple(p.shape), str(p.dtype)) for name, p in model.named_parameters())
    if len(set(training_comm.allgather(layout))) != 1:
        raise ValueError("Training ranks must agree on parameter names, order, shapes and dtypes.")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    interval = training_comm.rank == 0
    mesh_comm = training_comm.Split(color=0 if interval else 1, key=training_comm.rank)
    if interval:
        mesh = fd.UnitIntervalMesh(4, comm=mesh_comm)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.Function(V).assign(1.0)
    else:
        mesh = fd.UnitSquareMesh(2, 2, comm=mesh_comm)
        V = fd.FunctionSpace(mesh, "DG", 0)
        u = fd.Function(V).assign(2.0)

    N = ml_operator(model, V, inputs_format="pointwise")
    prediction = N(u, theta)
    J = 0.5 * fd.inner(prediction, prediction) * fd.dx

    mesh_loss = fd.assemble(J).real
    gradient = fd.assemble(fd.derivative(J, theta))
    loss, gradient = sum_losses_across_rank_groups(mesh_loss, gradient, mesh_comm, training_comm)

    optimizer.zero_grad(set_to_none=True)
    theta.backward(gradient)
    optimizer.step()

    dw, db = model.weight.grad, model.bias.grad
    np.testing.assert_allclose((loss, dw.item(), db.item()), (2.5, 5.0, 3.0))
    np.testing.assert_allclose((model.weight.item(), model.bias.item()), (0.5, -0.3))
    if training_comm.rank == 0:
        print(f"Total loss {loss:g}, gradient ({dw.item():g}, {db.item():g})")
        print(f"Updated weight {model.weight.item():g}, bias {model.bias.item():g}")


if __name__ == "__main__":
    main()
