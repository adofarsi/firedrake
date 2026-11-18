"""Run in a Firedrake #5551 environment with the pinned UFL #451 revision."""

from firedrake import (
    Argument,
    Function,
    FunctionSpace,
    TrialFunction,
    UnitIntervalMesh,
    VectorFunctionSpace,
    interpolate,
)
from firedrake.assemble import BaseFormAssembler
from ufl.core.external_operator import ExternalOperator

mesh = UnitIntervalMesh(4)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1, dim=2)
direction = interpolate(TrialFunction(W)[0], V)
expression = ExternalOperator(
    Function(V),
    function_space=V,
    derivatives=(1,),
    argument_slots=(Argument(V.dual(), 0), direction),
)

print("Expected normalization to retain the free vector argument in W.", flush=True)
normalized = BaseFormAssembler.preprocess_base_form(expression)
assert normalized.arguments()[-1].ufl_function_space() == W
print("PASS", flush=True)
