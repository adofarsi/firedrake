"""Evaluate complete forward objectives at the paper's initial weights."""
import importlib
import json
from pathlib import Path
import sys
import traceback

import firedrake as fd
from firedrake.adjoint import stop_annotating
from firedrake.ml.pytorch import MLParameters, ml_operator
import numpy as np
import torch


def displacement():
    ex = importlib.import_module("01_displacement_controlled")
    mesh = fd.RectangleMesh(6, 6, 0.05, 0.1)
    space = fd.FunctionSpace(mesh, "DG", ex.DEGREE - 1)
    model = ex.TorchModelForE(lower_bound_stiffness=20.0).double()
    with torch.no_grad():
        model.mlp[4].weight.normal_(0.0, 0.001)
        model.mlp[4].bias.fill_(-3.0)
    theta = MLParameters(model.parameters(), comm=mesh.comm)
    op = ml_operator(model, space, inputs_format="pointwise")
    records = {}
    for levels, name in [(ex.TRAIN_LEVELS, "train"), (ex.TEST_LEVELS, "test")]:
        _, reference, _ = ex.uniaxial_solver(levels, mesh)
        if name == "train":
            noise = np.random.normal(0., 0.01, len(reference))
            reference = [f * (1 + e) for f, e in zip(reference, noise)]
        _, forces, _ = ex.uniaxial_solver(levels, mesh, op, theta)
        relative = [fd.norm(f - r) / fd.norm(r) for f, r in zip(forces, reference)]
        records[name] = {"levels": len(levels), "loss": float(np.mean(relative))}
    return records


def shear():
    ex = importlib.import_module("04_shear_coupon")
    Path(ex.OUTPUT).mkdir(parents=True, exist_ok=True)
    modulus = ex.calibrate_and_correct()
    targets = ex.T.hardening_targets(ex.CORRECTED_CSV)
    levels = targets["displacement_levels"]
    mesh = fd.Mesh(str(ex.SHEAR_CONFIG["mesh_path"]))
    Q = fd.FunctionSpace(mesh, "DG", 0)
    sy = ex.IntegralMonotoneSigmaY(
        sigma_y0_init=ex.SIGMA_Y0_INIT, hidden_size=ex.HARD_HIDDEN_SIZE, p_scale=ex.HARD_P_SCALE,
        n_quad=ex.HARD_N_QUAD, stress_scale=ex.HARD_STRESS_SCALE,
    ).double()
    Hp = ex.IntegralHTangent(sy).double()
    theta = MLParameters(sy.parameters(), comm=mesh.comm)
    ops = {k: ml_operator(model, Q, inputs_format="pointwise") for k, model in (("sy", sy), ("Hp", Hp))}
    _, loads, _ = ex.plane_stress_svk_solver(
        levels, mesh, ops, {"sy": theta, "Hp": theta}, ex.SHEAR_CONFIG, modulus, ex.NU, ex.WEAK_BC_PENALTY,
    )
    branches = [(targets["is_elastic"], ex.T.HARD_W_ELASTIC),
                (targets["is_transition"], ex.T.HARD_W_TRANSITION), (targets["is_outer"], ex.T.HARD_W_OUTER)]
    loss = ex.T.force_curve_loss(loads, targets["mean_force"], targets["sigma_eff"], branches)
    return {"hardening": {"levels": len(levels), "loss": float(loss)}}


def thermal():
    ex = importlib.import_module("05_thermal_conductivity")
    mesh = ex.heat_mesh()
    V = fd.FunctionSpace(mesh, "CG", 1)
    model = ex.TorchModelForK().double()
    theta = MLParameters(model.parameters(), comm=mesh.comm)
    op = ml_operator(model, V, inputs_format="pointwise")
    records = {}
    for source, name in [(ex.SOURCE_TRAIN, "train"), (ex.SOURCE_TEST, "test")]:
        obs = ex.heat_solver(source, mesh)
        for field in obs:
            field.dat.data[:] += field.dat.data * np.random.normal(0, 0.02, size=field.dat.data.shape)
        pred = ex.heat_solver(source, mesh, op, theta)
        relative = [ex.sample_norm(f - r) / ex.sample_norm(r) for f, r in zip(pred, obs)]
        records[name] = {"timesteps": len(pred), "loss": float(np.mean(relative))}
    return records


if __name__ == "__main__":
    sys.path.insert(0, str(Path("paper/scripts").resolve()))
    name, = sys.argv[1:]
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        with stop_annotating():
            result = {"values": {"01": displacement, "04": shear, "05": thermal}[name]()}
        assert all(np.isfinite(v["loss"]) for v in result["values"].values())
    except Exception as error:
        traceback.print_exc()
        result = {"error": f"{type(error).__name__}: {error}"}
    Path(f"results/forward-{name}.json").write_text(json.dumps(result, indent=2))
    print(name, result)
    sys.exit(int("error" in result))
