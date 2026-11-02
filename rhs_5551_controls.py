"""Optional diagnostic controls; the cached RHS assignment stays unchanged."""
import importlib
import os

import pytest
import ufl


@pytest.fixture(autouse=True)
def diagnostic_assembly_controls(monkeypatch):
    assembly = importlib.import_module("firedrake.assemble")
    original_init = assembly.OneFormAssembler.__init__

    def init(self, form, *args, **kwargs):
        arguments = form.arguments()
        if isinstance(form, ufl.Form) and len(arguments) == 1 and arguments[0].number() != 0:
            print(f"ONE_FORM_ARGUMENT: {arguments[0].number()}; renumber={os.getenv('RHS_RENUMBER', '0')}", flush=True)
            if os.getenv("RHS_RENUMBER") == "1":
                argument, = arguments
                form = ufl.replace(form, {argument: argument.reconstruct(number=0)})
        original_init(self, form, *args, **kwargs)

    monkeypatch.setattr(assembly.OneFormAssembler, "__init__", init)

    if os.getenv("RHS_RENUMBER") == "1":
        original_preprocess = assembly.BaseFormAssembler.preprocess_base_form

        def preprocess(expr, *args, **kwargs):
            form = original_preprocess(expr, *args, **kwargs)
            arguments = form.arguments() if isinstance(form, ufl.Form) else ()
            if len(arguments) == 1 and arguments[0].number() != 0:
                argument, = arguments
                form = ufl.replace(form, {argument: argument.reconstruct(number=0)})
            return form

        monkeypatch.setattr(assembly.BaseFormAssembler, "preprocess_base_form", staticmethod(preprocess))

    if os.getenv("RHS_TRANSPOSE") == "1":
        original_visitor = assembly.BaseFormAssembler.base_form_assembly_visitor

        def visitor(self, expr, tensor, bcs, *args):
            if isinstance(expr, ufl.Adjoint) and tensor is not None:
                fresh = original_visitor(self, expr, None, bcs, *args)
                fresh.petscmat.copy(result=tensor.petscmat)
                return tensor
            return original_visitor(self, expr, tensor, bcs, *args)

        monkeypatch.setattr(assembly.BaseFormAssembler, "base_form_assembly_visitor", visitor)
