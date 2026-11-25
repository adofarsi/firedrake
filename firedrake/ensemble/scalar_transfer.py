"""Transfer nonspatial values between a parent communicator and its rank groups."""

from __future__ import annotations

from mpi4py import MPI
from pyadjoint import Block
from pyadjoint.overloaded_function import overloaded_function
import numpy as np
from pyop3.dtypes import get_mpi_dtype

from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.scalarspace import ScalarSpace


class _TransferBlock(Block):
    """Record a linear map whose reverse operation is its adjoint."""

    def __init__(self, transfer, value, transpose):
        super().__init__()
        self.add_dependency(value)
        self.forward = transfer.adjoint if transpose else transfer.replicate
        self.reverse = transfer.replicate if transpose else transfer.adjoint

    def recompute_component(self, inputs, block_variable, idx, prepared=None):
        return self.forward(inputs[0])

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return self.forward(tlm_inputs[0])

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return self.reverse(adj_inputs[0])

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable,
                                   idx, relevant_dependencies, prepared=None):
        return self.reverse(hessian_inputs[0])


class ScalarSpaceTransfer:
    """Replicate a value on rank groups and sum the dual contributions of the groups.

    Parameters
    ----------
    source : ScalarSpace
        Space on the parent communicator.
    target : ScalarSpace
        Space on the local group's communicator. The groups must partition
        the source communicator. Their sizes and rank order can differ.

    Notes
    -----
    Construction and both operations are collective on the source communicator.
    Each group owns one value even when its mesh is distributed over many ranks.
    """

    def __init__(self, source: ScalarSpace, target: ScalarSpace) -> None:
        if not isinstance(source, ScalarSpace) or not isinstance(target, ScalarSpace):
            raise TypeError("A scalar transfer requires two nonspatial primal spaces.")
        comm = source.comm
        if len(set(comm.allgather((source.shape, target.shape)))) != 1 or source.shape != target.shape:
            raise ValueError("Source and target must have the same value shape on every rank.")
        self.source = source
        self.target = target
        # A group has one leaf on its owner. Every leaf reads the same parent root.
        self._sf = PETSc.SF().create(comm=comm)
        remote = np.zeros((int(target.comm.rank == 0), 2), dtype=PETSc.IntType)
        self._sf.setGraph(int(comm.rank == 0), None, remote)

    def replicate(self, value: Function) -> Function:
        """Broadcast the parent value into a function on each group.

        Parameters
        ----------
        value : Function
            Function in the source space.

        Returns
        -------
        Function
            The local group's copy.
        """
        if not isinstance(value, Function) or value.function_space() != self.source:
            raise ValueError("Expected a function in the source space.")
        return self._apply(value, False)

    def adjoint(self, value: Cofunction) -> Cofunction:
        """Sum one dual contribution per group in the parent dual space.

        Parameters
        ----------
        value : Cofunction
            Cofunction in the target dual space.

        Returns
        -------
        Cofunction
            Sum of the group contributions.
        """
        if not isinstance(value, Cofunction) or value.function_space() != self.target.dual():
            raise ValueError("Expected a cofunction in the target dual space.")
        return self._apply(value, True)

    @overloaded_function(_TransferBlock)
    def _apply(self, value: Function | Cofunction, transpose: bool) -> Function | Cofunction:
        unit, _ = get_mpi_dtype(value.dat.dtype, self.source.dim())
        if transpose:
            result = Cofunction(self.source.dual(), dtype=value.dat.dtype)
            args = (unit, value.dat.data_ro, result.dat.data_rw, MPI.SUM)
            self._sf.reduceBegin(*args)
            self._sf.reduceEnd(*args)
        else:
            result = Function(self.target, dtype=value.dat.dtype)
            args = (unit, value.dat.data_ro, result.dat.data_wo, MPI.REPLACE)
            self._sf.bcastBegin(*args)
            self._sf.bcastEnd(*args)
        return result
