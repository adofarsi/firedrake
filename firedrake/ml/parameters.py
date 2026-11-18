"""Symbolic parameter spaces for machine learning models."""

import ufl
from ufl.cell import AbstractCell
from ufl.finiteelement import AbstractFiniteElement
from ufl.pullback import AbstractPullback, identity_pullback
from ufl.sobolevspace import SobolevSpace


class _MLParameterElement(AbstractFiniteElement):
    """Describe a finite-dimensional parameter vector without a spatial mesh."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __repr__(self) -> str:
        return f"_MLParameterElement({self._size})"

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((type(self), self._size))

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self._size == other._size

    @property
    def sobolev_space(self) -> SobolevSpace:
        return ufl.HInf

    @property
    def pullback(self) -> AbstractPullback:
        return identity_pullback

    @property
    def embedded_superdegree(self) -> int:
        return 0

    @property
    def embedded_subdegree(self) -> int:
        return 0

    @property
    def cell(self) -> AbstractCell:
        return ufl.vertex

    @property
    def reference_value_shape(self) -> tuple[int, ...]:
        return (self._size,)

    @property
    def sub_elements(self) -> tuple[AbstractFiniteElement, ...]:
        return ()


class AbstractMLParameters(ufl.Coefficient):
    """Identify a vector of model parameters in UFL.

    Parameters
    ----------
    size : int
        Number of scalar parameters.

    Notes
    -----
    Subclasses provide numerical storage in their machine learning framework.
    The UFL space describes a global parameter vector. Its metadata has no
    mesh, degree-of-freedom map, or numerical allocation.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("A parameter vector must contain at least one scalar.")
        super().__init__(ufl.FunctionSpace(None, _MLParameterElement(size)))
