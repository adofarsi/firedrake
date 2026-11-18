"""Check ML-dependent solves and their field gradients on the main branches."""

import pytest
import torch

from firedrake import *
from firedrake.adjoint import *
from firedrake.ml.pytorch import ml_operator


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.mark.skipcomplex
@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.usefixtures("garbage_cleanup")
@pytest.mark.parametrize("placement", ["plain_rhs", "rhs", "state"])
def test_ml_solve_gradient(placement: str) -> None:
    """Compare the solution, repeated adjoints and replay with an exact solution."""
    class Scale(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.weight * value

    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)
    source = Function(V).interpolate(1 + x)
    u, v = Function(V), TestFunction(V)
    model = Scale()
    N = ml_operator(model, function_space=V)
    if placement in ("plain_rhs", "rhs"):
        operand = source if placement == "plain_rhs" else interpolate(source, V)
        residual = inner(u - N(operand), v) * dx
        solution_factor, jacobian_factor = 2.0, 1.0
    else:
        residual = inner(u + N(interpolate(u, V)) - source, v) * dx
        solution_factor, jacobian_factor = 1.0 / 3.0, 3.0

    jacobian = jacobian_factor * inner(TrialFunction(V), v) * dx
    problem = NonlinearVariationalProblem(residual, u, J=jacobian)
    solver = NonlinearVariationalSolver(problem, solver_parameters={
        "ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12,
        "snes_rtol": 1e-12, "snes_atol": 1e-12,
    })
    solver.solve()
    with stop_annotating():
        error = assemble(u - solution_factor * source)
        with error.dat.vec_ro as vector:
            assert vector.norm() < 1e-10

    objective = assemble(0.5 * u**2 * dx)
    assert abs(float(objective) - solution_factor**2 * 7.0 / 6.0) < 1e-10
    functional = ReducedFunctional(objective, Control(source))
    for seed in (1.0, -2.0, 0.0, 3.0):
        gradient = functional.derivative(adj_input=seed)
        with stop_annotating():
            expected = assemble(seed * solution_factor**2 * source * v * dx)
            error = assemble(gradient - expected)
            with error.dat.vec_ro as vector:
                assert vector.norm() < 1e-10
    assert model.weight.grad is None
    direction = Function(V).assign(0.1)
    assert taylor_test(functional, source, direction) > 1.95


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.usefixtures("garbage_cleanup")
@pytest.mark.parametrize("constant_jacobian", [False, True])
def test_cached_adjoint_rhs(constant_jacobian: bool) -> None:
    """A cached adjoint solver uses the current RHS for every derivative.

    Interpolation is the identity on V, so the solution is exactly f/2.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)
    f = Function(V).interpolate(1 + x)
    u, v = Function(V), TestFunction(V)
    F = inner(interpolate(u, V), v)*dx + inner(u - f, v)*dx
    problem = NonlinearVariationalProblem(F, u)
    problem._constant_jacobian = constant_jacobian
    solver = NonlinearVariationalSolver(problem, solver_parameters={
        "ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12,
    })
    solver.solve()
    linear = ReducedFunctional(assemble(u*dx), Control(f))
    quadratic = ReducedFunctional(assemble(0.5*u**2*dx), Control(f))
    adjoint_solver = solver._ad_solvers["adjoint_lvs"]

    for functional, seed, density in (
        (linear, 1.0, 0.5),
        (quadratic, -2.0, f/4),
        (quadratic, 0.0, f/4),
        (linear, 3.0, 0.5),
    ):
        gradient = functional.derivative(adj_input=seed)
        with stop_annotating():
            expected = assemble(Constant(seed)*density*v*dx)
            error = assemble(gradient - expected)
        with error.dat.vec_ro as vec:
            assert vec.norm() < 1e-11
        assert solver._ad_solvers["adjoint_lvs"] is adjoint_solver
