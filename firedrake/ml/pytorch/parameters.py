"""Model parameters that retain their native PyTorch storage."""

from __future__ import annotations

from collections.abc import Iterable
import os

try:
    import torch
except ImportError:
    if "FIREDRAKE_BUILDING_DOCS" not in os.environ:
        raise

from firedrake.ml.parameters import AbstractMLParameters


class MLParameters(AbstractMLParameters):
    """Represent PyTorch tensors as a single symbolic parameter vector.

    Parameters
    ----------
    parameters : Iterable[torch.Tensor]
        Tensors to reference, normally ``model.parameters()``. Their order and
        shapes define the parameter vector and must remain fixed. All tensors
        must have floating-point or complex values.

    Notes
    -----
    Construction retains references to the supplied tensors. It does not copy
    their values, change their dtype or device, or allocate a Firedrake field.
    Each MPI rank retains its own model parameters; this class performs no
    communication or reduction.

    Examples
    --------
    >>> parameters = MLParameters(model.parameters())
    >>> saved_parameters = parameters.copy()
    >>> parameters.assign(saved_parameters)
    """

    def __init__(self, parameters: Iterable[torch.Tensor]) -> None:
        self._tensors = tuple(parameters)
        if any(not isinstance(t, torch.Tensor) for t in self._tensors):
            raise TypeError("Model parameters must be PyTorch tensors.")
        if any(not (t.is_floating_point() or t.is_complex()) for t in self._tensors):
            raise TypeError("Model parameters must have floating-point or complex values.")
        super().__init__(sum(t.numel() for t in self._tensors))

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the native parameter tensors.

        Returns
        -------
        tuple[torch.Tensor]
            References to the tensors in parameter order.
        """
        return self._tensors

    def copy(self) -> "MLParameters":
        """Copy parameter values without retaining their autograd history.

        Returns
        -------
        MLParameters
            Independent tensors with the same shapes, dtypes and devices.
        """
        return type(self)(t.detach().clone() for t in self._tensors)

    def assign(self, other: "MLParameters") -> "MLParameters":
        """Copy values while preserving the original tensors and their storage.

        Parameters
        ----------
        other : MLParameters
            Values with the same tensor shapes, dtypes and devices.

        Returns
        -------
        MLParameters
            This parameter vector.

        Raises
        ------
        ValueError
            If the tensor shapes, dtypes or devices differ.
        """
        source = other.tensors
        target = self._tensors
        if len(source) != len(target) or any(
            (s.shape, s.dtype, s.device) != (t.shape, t.dtype, t.device)
            for s, t in zip(source, target)
        ):
            raise ValueError("Parameter tensors must have matching shapes, dtypes and devices.")
        with torch.no_grad():
            for t, s in zip(target, source):
                t.copy_(s)
        return self
