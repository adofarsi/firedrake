"""Check cached adjoint gradients on the current FEML integration stack."""

import pytest
import ufl

from firedrake import *
from firedrake.adjoint import *


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.usefixtures("garbage_cleanup")
@pytest.mark.parametrize("case, constant_jacobian", [
    (case, constant_jacobian)
    for case in [
        "plain", "interpolate", "nested", "two_interpolations", "different_space",
        "nonlinear", "vector", "external", "external_of_interpolate",
        "interpolate_of_external", "preassembled",
    ]
    for constant_jacobian in [False, True]
    if case != "nonlinear" or not constant_jacobian
])
def test_rhs_shape_gradients(case: str, constant_jacobian: bool) -> None:
    """Check residual structure and exact gradients through supported operators."""
    mesh = UnitIntervalMesh(4)
    scalar = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2) if case == "vector" else scalar
    x, = SpatialCoordinate(mesh)
    c = as_vector([1, 2]) if case == "vector" else Constant(1)
    value = as_vector([1 + x, 2 - x]) if case == "vector" else 1 + x
    f = Function(V).interpolate(value)
    u, v = Function(V), TestFunction(V)
    coefficient = 2
    N = interpolate(u, V)
    if case == "plain":
        N = u
        coefficient = 1
        F = inner(u - f, v)*dx
    else:
        if case == "nested":
            N = interpolate(interpolate(u, V), V)
        elif case == "two_interpolations":
            N = interpolate(u, V) + interpolate(2*u, V)
            coefficient = 4
        elif case == "different_space":
            W = FunctionSpace(mesh, "CG", 2)
            N = interpolate(interpolate(u, W), V)
        elif case == "nonlinear":
            f.assign(2)
            N = interpolate(u**2, V)
        elif case in ("external", "external_of_interpolate", "interpolate_of_external"):
            identity = point_expr(lambda z: z, function_space=V)
            N = identity(interpolate(u, V)) if case == "external_of_interpolate" else identity(u)
            if case == "interpolate_of_external":
                N = interpolate(N, V)
        if case == "preassembled":
            with stop_annotating():
                mass = assemble(inner(TrialFunction(V), v)*dx)
            F = action(mass, interpolate(u, V)) + inner(u - f, v)*dx
        else:
            F = inner(N, v)*dx + inner(u - f, v)*dx

    problem = NonlinearVariationalProblem(F, u)
    problem._constant_jacobian = constant_jacobian
    solver = NonlinearVariationalSolver(problem, solver_parameters={
        "ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12,
        "snes_rtol": 1e-12, "snes_atol": 1e-12,
    })
    solver.solve()
    cached = solver._ad_solvers["adjoint_lvs"]
    residual = cached._problem.F
    components = residual.components() if isinstance(residual, ufl.FormSum) else ()
    print(f"{case}: components={[type(term).__name__ for term in components]}", flush=True)
    assert len(components) == 2
    assert isinstance(components[1], Cofunction)
    rhs = components[1]
    linear = ReducedFunctional(assemble(inner(c, u)*dx), Control(f))
    quadratic = ReducedFunctional(assemble(0.5*inner(u, u)*dx), Control(f))
    linear_density = c/coefficient
    quadratic_density = f/coefficient**2
    if case == "nonlinear":
        linear_density = c/3
        quadratic_density = Constant(1/3)
    for functional, seed, density in (
        (linear, 1.0, linear_density),
        (quadratic, -2.0, quadratic_density),
        (quadratic, 0.0, quadratic_density),
        (linear, 3.0, linear_density),
    ):
        gradient = functional.derivative(adj_input=seed)
        with stop_annotating():
            expected = assemble(inner(Constant(seed)*density, v)*dx)
            error = assemble(gradient - expected)
        with error.dat.vec_ro as vec:
            assert vec.norm() < 1e-10
        assert solver._ad_solvers["adjoint_lvs"] is cached
        assert cached._problem.F.components()[1] is rhs
