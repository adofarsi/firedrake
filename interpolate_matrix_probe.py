"""Trace matrix-valued interpolation operands after numerical replacement."""

import importlib
import json
from pathlib import Path
import traceback

import ufl
from ufl.algorithms import extract_arguments
from ufl.core.base_form_operator import BaseFormOperator
from ufl.duals import is_dual

from firedrake import (Function, FunctionSpace, TestFunction, UnitIntervalMesh,
                       assemble, derivative, dx, inner, interpolate, point_expr)
from firedrake.assemble import BaseFormAssembler
from firedrake.matrix import MatrixBase
from firedrake.petsc import PETSc


def describe(expression, depth: int = 2) -> dict:
    arguments = expression.arguments() if isinstance(expression, ufl.BaseForm) else extract_arguments(expression)
    result = {"type": type(expression).__name__,
              "arguments": [{"number": arg.number(), "dual": is_dual(arg), "shape": arg.ufl_shape}
                            for arg in arguments]}
    if depth > 0:
        if isinstance(expression, BaseFormOperator):
            result["slots"] = [describe(slot, depth - 1) for slot in expression.argument_slots()]
        elif isinstance(expression, ufl.Form):
            result["operators"] = [describe(op, depth - 1) for op in expression.base_form_operators()]
        else:
            result["operands"] = [describe(op, depth - 1) for op in expression.ufl_operands]
    return result


interpolation_module = importlib.import_module("firedrake.interpolation")
original_visitor = BaseFormAssembler.base_form_assembly_visitor
original_compile = interpolation_module.compile_expression
reports = []
events = []
mode = "baseline"


def visitor(self, expression, tensor, bcs, *args):
    if isinstance(expression, ufl.Interpolate):
        events.append({"event": "interpolation_visit", "expression": describe(expression, 3),
                       "assembled": [describe(arg, 1) for arg in args]})
        if mode == "normalize_replacement" and len(args) > 1 and isinstance(args[1], MatrixBase):
            reconstructed = expression._ufl_expr_reconstruct_(args[1], v=args[0])
            normalized = self.preprocess_base_form(reconstructed)
            events.append({"event": "normalized_replacement", "expression": describe(normalized, 3)})
            if not isinstance(normalized, ufl.Interpolate):
                return self._assemble_base_form(normalized, tensor=tensor)
    return original_visitor(self, expression, tensor, bcs, *args)


def compile_expression(comm, expression, *args, **kwargs):
    events.append({"event": "compile", "expression": describe(expression, 4)})
    return original_compile(comm, expression, *args, **kwargs)


BaseFormAssembler.base_form_assembly_visitor = visitor
interpolation_module.compile_expression = compile_expression
try:
    for mode in ("baseline", "extra_preprocess", "normalize_replacement"):
        for case in ("direct", "weak", "residual"):
            events = []
            mesh = UnitIntervalMesh(4)
            V = FunctionSpace(mesh, "CG", 1)
            u, v = Function(V), TestFunction(V)
            N = point_expr(lambda z: z, function_space=V)
            outer = interpolate(N(u), V)
            expression = {"direct": outer, "weak": inner(outer, v)*dx,
                          "residual": inner(outer, v)*dx + inner(u, v)*dx}[case]
            jacobian = derivative(expression, u)
            report = {"mode": mode, "case": case}
            try:
                normalized = BaseFormAssembler.preprocess_base_form(jacobian)
                report["normalized"] = describe(normalized, 5)
                value = assemble(normalized if mode == "extra_preprocess" else jacobian)
                report["status"] = "passed"
                report["result"] = describe(value)
            except Exception as error:
                report["status"] = "failed"
                report["error"] = str(error)
                report["traceback"] = [{"file": item.filename, "line": item.lineno, "function": item.name}
                                       for item in traceback.extract_tb(error.__traceback__)]
            report["events"] = events
            reports.append(report)
finally:
    BaseFormAssembler.base_form_assembly_visitor = original_visitor
    interpolation_module.compile_expression = original_compile

if PETSc.COMM_WORLD.rank == 0:
    Path("../results/matrix-probe.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps([{key: value for key, value in report.items() if key not in ("events", "normalized", "traceback")}
                      for report in reports], indent=2), flush=True)
