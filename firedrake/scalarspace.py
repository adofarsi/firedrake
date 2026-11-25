"""Finite-dimensional Real spaces with no mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from operator import index

from mpi4py import MPI
import numpy as np
import pyop3 as op3
import ufl

from firedrake.utils import ScalarType


def real_axes(comm: MPI.Comm, shape: tuple[int, ...]) -> op3.AxisTree:
    """Build axes with one global owner for each component."""
    size, *rest = shape or (1,)
    root = op3.AxisComponent(size, sf=op3.single_star_sf(comm, size))
    if not shape:
        return op3.AxisTree(op3.Axis(root))
    components = [root, *(op3.AxisComponent(n) for n in rest)]
    return op3.AxisTree.from_iterable(
        op3.Axis(c, label=f"dim{i}") for i, c in enumerate(components)
    )


@dataclass(frozen=True, slots=True)
class _RealElement(ufl.AbstractFiniteElement):
    """Describe the value shape without constructing an element per component."""

    reference_value_shape: tuple[int, ...] = field()
    cell = ufl.vertex
    sobolev_space = ufl.HInf
    pullback = ufl.identity_pullback
    embedded_subdegree = embedded_superdegree = 0
    sub_elements = ()

    def family(self) -> str:
        return "Real"

    def __str__(self) -> str:
        return repr(self)


@dataclass(eq=False)
class _Layout:
    domain: NonSpatialDomain
    shape: tuple[int, ...]

    @cached_property
    def axes(self) -> op3.AxisTree:
        return real_axes(self.domain.comm, self.shape)


class NonSpatialDomain(ufl.NonSpatialDomain):
    """Declare coefficient spaces that share a communicator and have no geometry.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Ranks that share the algebraic values.

    Notes
    -----
    Reusing the same domain and value shape gives compatible spaces.
    """

    def __init__(self, comm: MPI.Comm) -> None:
        super().__init__()
        self.comm = comm
        self._layouts: dict[tuple[int, ...], _Layout] = {}

    def _layout(self, shape: tuple[int, ...]) -> _Layout:
        if shape not in self._layouts:
            self._layouts[shape] = _Layout(self, shape)
        return self._layouts[shape]


class ScalarSpaceBase:
    parent = None

    def __init__(self, layout: _Layout, name: str | None = None) -> None:
        self._layout = layout
        self.name = name
        super().__init__(layout.domain, _RealElement(layout.shape))

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield self

    def __getitem__(self, i: int) -> ScalarSpaceBase:
        if i != 0:
            raise IndexError(i)
        return self

    @property
    def comm(self) -> MPI.Comm:
        return self._layout.domain.comm

    @property
    def shape(self) -> tuple[int, ...]:
        return self._layout.shape

    @property
    def topological(self) -> ScalarSpaceBase:
        return self

    def mesh(self) -> None:
        return None

    def ufl_function_space(self) -> ScalarSpaceBase:
        return self

    def dim(self) -> int:
        """Return the global number of degrees of freedom."""
        return self.value_size

    def make_dat(self, val=None, valuetype=ScalarType, name: str | None = None) -> op3.Dat:
        axes = self._layout.axes
        if val is None:
            return op3.Dat.zeros(axes, dtype=valuetype, name=name)
        data = np.asarray(val, dtype=valuetype)
        if data.size != self.dim():
            raise ValueError(f"Expected {self.dim()} values, got {data.size}.")
        return op3.Dat(axes, data=data.ravel(), name=name)


class ScalarSpace(ScalarSpaceBase, ufl.FunctionSpace):
    """Declare a finite-dimensional space whose values are shared on a communicator.

    Parameters
    ----------
    domain : NonSpatialDomain
        Domain on whose communicator each component has one global value.
    shape : tuple of int, optional
        Shape of the value. The empty tuple gives a scalar.

    Notes
    -----
    Spaces with the same domain and shape are compatible. Their dual
    contains the corresponding linear functionals. A mesh-independent space
    has no spatial gradient or integration measure of its own.

    Examples
    --------
    >>> P = ScalarSpace(NonSpatialDomain(COMM_WORLD), (2,))
    >>> h = Function(P, val=[1, 2])
    >>> g = Cofunction(P.dual(), val=[3, 4])
    >>> assemble(action(g, h)).real
    11.0
    """

    def __init__(self, domain: NonSpatialDomain, shape: tuple[int, ...] = (),
                 name: str | None = None) -> None:
        shape = tuple(index(n) for n in shape)
        if any(n < 1 for n in shape):
            raise ValueError("Real space extents must be positive.")
        super().__init__(domain._layout(shape), name)

    @cached_property
    def _dual_space(self) -> _DualScalarSpace:
        return _DualScalarSpace(self)

    def dual(self) -> _DualScalarSpace:
        """Return the space of linear functionals on this space."""
        return self._dual_space


class _DualScalarSpace(ScalarSpaceBase, ufl.functionspace.DualSpace):
    def __init__(self, primal: ScalarSpace) -> None:
        self._primal_space = primal
        super().__init__(primal._layout, primal.name)

    def dual(self) -> ScalarSpace:
        return self._primal_space
