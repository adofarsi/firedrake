"""Exercise the paper's first solver with the explicit parameter interface."""
import importlib
import json
from pathlib import Path
import sys
import traceback

import firedrake as fd
from firedrake.adjoint import Control, ReducedFunctional, continue_annotation, set_working_tape, stop_annotating
from firedrake.ml.pytorch import MLParameters, ml_operator
import numpy as np
import torch


def quadratic(stage):
    mesh = fd.UnitIntervalMesh(2)
    V = fd.FunctionSpace(mesh, "CG", 1)
    field = fd.Function(V).assign(1)
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(2)
    theta = MLParameters(model.parameters(), comm=mesh.comm)
    op = ml_operator(model, V, inputs_format="pointwise")
    form = 0.5 * fd.inner(op(field, theta), op(field, theta)) * fd.dx
    if stage == "explicit":
        with stop_annotating():
            loss = float(fd.assemble(form))
            gradient = fd.assemble(fd.derivative(form, theta))
            theta.backward(gradient)
            derivative = model.weight.grad.item()
        np.testing.assert_allclose([loss, derivative], [2, 2])
        return {"loss": loss, "gradient": derivative}
    if stage == "control":
        Control(theta)
        return
    with set_working_tape():
        continue_annotation()
        loss = fd.assemble(form)
        rf = ReducedFunctional(loss, Control(theta))
        return {"gradient": rf.derivative().dat.data_ro.tolist()}


def paper_level(annotate):
    paper = importlib.import_module("01_displacement_controlled")
    mesh = fd.RectangleMesh(6, 6, 0.05, 0.1)
    D = fd.FunctionSpace(mesh, "DG", paper.DEGREE - 1)
    model = paper.TorchModelForE(lower_bound_stiffness=20.0).double()
    with torch.no_grad():
        model.mlp[4].weight.normal_(0.0, 0.001)
        model.mlp[4].bias.fill_(-3.0)
    theta = MLParameters(model.parameters(), comm=mesh.comm)
    op = ml_operator(model, D, inputs_format="pointwise")
    with set_working_tape():
        continue_annotation()
        with stop_annotating():
            _, reference, _ = paper.uniaxial_solver(paper.TRAIN_LEVELS[:1], mesh)
        if annotate:
            _, forces, _ = paper.uniaxial_solver(paper.TRAIN_LEVELS[:1], mesh, op, theta)
            loss = fd.norm(forces[0] - reference[0]) / fd.norm(reference[0])
            rf = ReducedFunctional(loss, Control(theta))
            theta.backward(rf.derivative())
        else:
            with stop_annotating():
                _, forces, _ = paper.uniaxial_solver(paper.TRAIN_LEVELS[:1], mesh, op, theta)
        value = float(forces[0].dat.data_ro.item())
        assert np.isfinite(value) and value != 0
        return {"first_force": value}


def scalar_solve():
    with stop_annotating():
        mesh = fd.UnitIntervalMesh(4)
        V = fd.FunctionSpace(mesh, "CG", 1)
        p = fd.Function(fd.ScalarSpace(fd.NonSpatialDomain(mesh.comm)), val=2)
        u = fd.Function(V)
        F = fd.inner(u - p, fd.TestFunction(V)) * fd.dx
        solver = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F, u))
        means = []
        for value in (2, 3):
            p.assign(value)
            solver.solve()
            means.append(float(fd.assemble(u * fd.dx).real))
        np.testing.assert_allclose(means, [2, 3], atol=1.e-12)
        return {"means": means}


if __name__ == "__main__":
    sys.path.insert(0, str(Path("paper/scripts").resolve()))
    probes = {"explicit": lambda: quadratic("explicit"), "control": lambda: quadratic("control"),
              "taped_loss": lambda: quadratic("taped"), "paper_forward": lambda: paper_level(False),
              "paper_taped": lambda: paper_level(True), "scalar_solve": scalar_solve}
    results = {}
    for name in sys.argv[1:] or probes:
        try:
            result = probes[name]()
            results[name] = {"passed": True, "result": result}
        except Exception as exc:
            traceback.print_exc()
            results[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        print(name, results[name], flush=True)
        Path("results/controls.json").write_text(json.dumps(results, indent=2))
    sys.exit(int(not all(result["passed"] for result in results.values())))
