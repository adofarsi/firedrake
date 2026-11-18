"""Measure transpose-buffer reuse and source lifetimes without changing Firedrake."""

import ctypes
import importlib
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI
import ufl

from firedrake import (Constant, Function, FunctionSpace, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, SpatialCoordinate, TestFunction,
                       TrialFunction, UnitIntervalMesh, assemble, dx, inner,
                       interpolate)
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                              stop_annotating)
from firedrake.matrix import MatrixBase
from firedrake.petsc import PETSc


# This diagnostic calls the public C API because petsc4py has no binding for it.
# PetscObjectState is a PetscInt64 in this pinned PETSc build.
_petsc = ctypes.CDLL(PETSc.__file__)
_get_nonzero_state = _petsc.MatGetNonzeroState
_get_nonzero_state.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64))
_get_nonzero_state.restype = ctypes.c_int


def nonzero_state(matrix: PETSc.Mat) -> int:
    state = ctypes.c_int64()
    error = _get_nonzero_state(matrix.handle, ctypes.byref(state))
    if error:
        raise RuntimeError(f"MatGetNonzeroState returned {error}")
    return state.value


class TransposeBuffer:
    """Keep one transpose until its source identity or structure changes."""

    def __init__(self) -> None:
        self.key = None
        self.buffer = None
        self.allocations = 0

    def update(self, source: PETSc.Mat) -> PETSc.Mat:
        key = (source.getId(), source.getType(), source.getSizes(), nonzero_state(source))
        rebuild = source.comm.tompi4py().allreduce(key != self.key, op=MPI.LOR)
        if rebuild:
            result = source.hermitianTranspose(out=PETSc.Mat())
            self.buffer = result
            self.key = key
            self.allocations += 1
        else:
            source.hermitianTranspose(out=self.buffer)
        return self.buffer


def check_buffer_reuse() -> dict:
    source = PETSc.Mat().createAIJ(size=(6, 6), nnz=3, comm=MPI.COMM_WORLD)
    first, last = source.getOwnershipRange()
    rows = np.arange(first, last, dtype=PETSc.IntType)
    source.setPreallocationCOO(rows, rows)
    source.setValuesCOO(np.full(len(rows), 2, dtype=PETSc.ScalarType))
    source.assemble()
    output = source.hermitianTranspose(out=PETSc.Mat())
    output_handle = output.handle
    cache = TransposeBuffer()
    stages = []

    def check(label: str) -> None:
        result = cache.update(source)
        result.copy(output)
        reference = source.hermitianTranspose(out=PETSc.Mat())
        difference = output.copy()
        difference.axpy(-1, reference)
        assert difference.norm() < 1e-12
        assert output.handle == output_handle
        stages.append({"label": label, "allocations": cache.allocations,
                       "work_id": result.getId(), "source_id": source.getId(),
                       "nonzero_state": nonzero_state(source)})

    check("initial")
    for scale in (2, -1, 0):
        source.scale(scale)
        check(f"scale_{scale}")
    assert cache.allocations == 1

    source.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    source.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, False)
    source.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, True)
    if first <= 0 < last:
        source.setValue(0, 1, 7)
    source.assemble()
    assert abs(source.norm() - 7) < 1e-12
    check("one_rank_inserts_entry")
    assert cache.allocations == 2, stages

    previous_source = source
    source = previous_source.copy()
    assert source.getId() != previous_source.getId()
    check("replacement_source_same_graph")
    assert cache.allocations == 3
    source.scale(2)
    check("replacement_source_values_only")
    assert cache.allocations == 3
    return {"stages": stages, "transpose_allocations": cache.allocations,
            "updates": len(stages)}


def trace_firedrake_sources() -> list[dict]:
    module = importlib.import_module("firedrake.assemble")
    original = module.BaseFormAssembler.base_form_assembly_visitor
    records = []
    phase = "explicit_matrix"

    def traced(self, expression, tensor, bcs, *args):
        if isinstance(expression, ufl.Adjoint) and len(args) == 1 and isinstance(args[0], MatrixBase):
            source = args[0].petscmat
            records.append({"phase": phase, "source_id": source.getId(),
                            "nonzero_state": nonzero_state(source),
                            "output_id": None if tensor is None else tensor.petscmat.getId()})
        return original(self, expression, tensor, bcs, *args)

    module.BaseFormAssembler.base_form_assembly_visitor = traced
    try:
        mesh = UnitIntervalMesh(4)
        V = FunctionSpace(mesh, "CG", 1)
        coefficient = Function(V).assign(1)
        form = inner(coefficient * TrialFunction(V), TestFunction(V)) * dx
        source = assemble(form)
        output = assemble(inner(TrialFunction(V), TestFunction(V)) * dx)
        for value in (1, 2, -1, 0):
            coefficient.assign(value)
            assemble(form, tensor=source)
            assemble(ufl.Adjoint(source), tensor=output)

        phase = "cached_adjoint_setup"
        continue_annotation()
        x, = SpatialCoordinate(mesh)
        f = Function(V).interpolate(1 + x)
        u, v = Function(V), TestFunction(V)
        residual = inner(interpolate(u, V), v)*dx + inner(u - f, v)*dx
        problem = NonlinearVariationalProblem(residual, u)
        solver = NonlinearVariationalSolver(problem, solver_parameters={
            "ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12,
        })
        solver.solve()
        functional = ReducedFunctional(assemble(0.5 * u**2 * dx), Control(f))
        for index, seed in enumerate((1.0, -2.0, 0.0, 3.0)):
            phase = f"cached_adjoint_derivative_{index}"
            gradient = functional.derivative(adj_input=seed)
            with stop_annotating():
                expected = assemble(Constant(seed) * f / 4 * v * dx)
                error = assemble(gradient - expected)
                with error.dat.vec_ro as vector:
                    assert vector.norm() < 1e-11
        return records
    finally:
        module.BaseFormAssembler.base_form_assembly_visitor = original


report = {"petsc": PETSc.Sys.getVersion(), "ranks": MPI.COMM_WORLD.size,
          "native_buffer_reuse": check_buffer_reuse(),
          "firedrake_source_trace": trace_firedrake_sources()}
reports = MPI.COMM_WORLD.gather(report, root=0)
if MPI.COMM_WORLD.rank == 0:
    path = Path(f"../results/reuse-n{MPI.COMM_WORLD.size}.json")
    path.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2), flush=True)
