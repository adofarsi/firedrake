"""Numerical comparison cases shared by the baseline and candidate."""
import pytest
import torch
from torch.nn import Module
from firedrake import *
from firedrake.ml.pytorch import ml_operator


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.usefixtures("garbage_cleanup")
@pytest.mark.skiptorch
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 6])
def test_nested_operator_adjoint(depth: int) -> None:
    """Nested interpolation and ML operators preserve the adjoint action."""
    class Scale(Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return 2 * x

    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)
    f = Function(V).interpolate(1 + x)
    w = Function(V).interpolate(2 - x)
    N = ml_operator(Scale(), function_space=V)
    value = f
    for _ in range(depth):
        value = N(interpolate(value, V))
    form = inner(value, TestFunction(V)) * dx
    actual = assemble(action(adjoint(derivative(form, f)), w))
    expected = assemble(2 ** depth * inner(w, TestFunction(V)) * dx)
    with assemble(actual - expected).dat.vec_ro as difference:
        assert difference.norm() < 1e-12


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.usefixtures("garbage_cleanup")
@pytest.mark.skiptorch
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 6])
def test_nested_operator_expanded_adjoint(depth: int) -> None:
    """Assemble the adjoint of an explicitly expanded composition Jacobian."""
    class Scale(Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return 2 * x

    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)
    value = Function(V).interpolate(1 + x)
    w = Function(V).interpolate(2 - x)
    N = ml_operator(Scale(), function_space=V)
    jacobian = None
    for _ in range(depth):
        operand = interpolate(value, V)
        value = N(operand)
        operator_jacobian = value._ufl_expr_reconstruct_(
            operand, derivatives=(1,), argument_slots=(Argument(V.dual(), 0), TrialFunction(V)))
        layer = action(operator_jacobian, interpolate(TrialFunction(V), V))
        jacobian = layer if jacobian is None else action(layer, jacobian)
    mass = inner(TrialFunction(V), TestFunction(V)) * dx
    actual = assemble(action(adjoint(action(mass, jacobian)), w))
    expected = assemble(2 ** depth * inner(w, TestFunction(V)) * dx)
    with assemble(actual - expected).dat.vec_ro as difference:
        assert difference.norm() < 1e-12


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.usefixtures("garbage_cleanup")
def test_vector_to_scalar_direction() -> None:
    """Normalize the vector-to-scalar derivative slot used by paper example 01."""
    from firedrake.assemble import BaseFormAssembler
    from ufl.core.external_operator import ExternalOperator

    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    W = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    direction = interpolate(TrialFunction(W)[0], V)
    dN = ExternalOperator(Function(V), function_space=V, derivatives=(1,),
                          argument_slots=(Argument(V.dual(), 0), direction))
    result = BaseFormAssembler.preprocess_base_form(dN)
    assert result.arguments()[-1].ufl_function_space() == W


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.usefixtures("garbage_cleanup")
def test_interpolate_action_value() -> None:
    """An interpolation can consume the value of a composed linear map."""
    import ufl

    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    I = interpolate(TrialFunction(V), V)
    composed = interpolate(ufl.Action(I, I), V)
    x, = SpatialCoordinate(mesh)
    f = Function(V).interpolate(1 + x)
    result = assemble(action(composed, f))
    with assemble(result - f).dat.vec_ro as error:
        assert error.norm() < 1e-12


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.usefixtures("garbage_cleanup")
@pytest.mark.parametrize("preallocated", [False, True])
def test_transpose_output(preallocated: bool) -> None:
    """Assemble a nonsymmetric matrix adjoint into fresh and existing outputs."""
    import ufl

    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    form = (inner(u, v) + inner(u.dx(0), v))*dx
    A = assemble(form)
    target = assemble(inner(u, v)*dx) if preallocated else None
    actual = assemble(ufl.Adjoint(A), tensor=target)
    if preallocated:
        assert actual is target
    expected = assemble(adjoint(form))
    x, = SpatialCoordinate(mesh)
    f = Function(V).interpolate(1 + x)
    error = assemble(action(actual, f)) - assemble(action(expected, f))
    with assemble(error).dat.vec_ro as difference:
        assert difference.norm() < 1e-12
    original_error = assemble(action(A, f)) - assemble(action(form, f))
    with assemble(original_error).dat.vec_ro as difference:
        assert difference.norm() < 1e-12
