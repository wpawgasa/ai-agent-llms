"""Cl(3,0) geometric algebra primitives for RotorQuant.

Implements the Clifford algebra Cl(3,0) with 8-dimensional multivectors
for rotor-based rotation of KV cache vectors. The rotor sandwich product
R x R† requires ~100 FMAs for d=128, compared to 16,384 FMAs for dense
matrix-vector rotation.

Algebra structure:
  Cl(3,0) basis: {1, e1, e2, e3, e12, e13, e23, e123}
  Grades: scalar(0), vector(1), bivector(2), trivector(3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Cl(3,0) basis element indices
SCALAR = 0  # 1
E1 = 1  # e1
E2 = 2  # e2
E3 = 3  # e3
E12 = 4  # e1e2
E13 = 5  # e1e3
E23 = 6  # e2e3
E123 = 7  # e1e2e3 (pseudoscalar)

N_COMPONENTS = 8


@dataclass
class Rotor:
    """A Cl(3,0) rotor represented as an 8-component multivector.

    A rotor R satisfies R R† = 1 and is composed of even-grade elements
    (scalar + bivector): R = a + b*e12 + c*e13 + d*e23.
    """

    components: "torch.Tensor"  # Shape: (..., 8)

    @property
    def scalar_part(self) -> "torch.Tensor":
        return self.components[..., SCALAR]

    @property
    def bivector_part(self) -> "torch.Tensor":
        return self.components[..., E12:E23 + 1]


class CliffordAlgebra:
    """Cl(3,0) algebra implementation for rotor-based rotation.

    Key operations:
      - embed(v: R^d) → Cl(3,0) multivector (grade-1)
      - rotor_sandwich(R, x) → R x R† (~100 FMAs for d=128)
      - extract(mv) → R^d
    """

    def embed(self, vectors: "torch.Tensor") -> "torch.Tensor":
        """Embed R^3 vectors into Cl(3,0) as grade-1 elements.

        Args:
            vectors: Input vectors of shape (..., 3).

        Returns:
            Multivectors of shape (..., 8) with only grade-1 components.
        """
        import torch

        shape = vectors.shape[:-1]
        mv = torch.zeros(*shape, N_COMPONENTS, dtype=vectors.dtype, device=vectors.device)
        mv[..., E1] = vectors[..., 0]
        mv[..., E2] = vectors[..., 1]
        mv[..., E3] = vectors[..., 2]
        return mv

    def extract(self, multivectors: "torch.Tensor") -> "torch.Tensor":
        """Extract R^3 vector from grade-1 components of a multivector.

        Args:
            multivectors: Multivectors of shape (..., 8).

        Returns:
            Vectors of shape (..., 3).
        """
        import torch

        return torch.stack([
            multivectors[..., E1],
            multivectors[..., E2],
            multivectors[..., E3],
        ], dim=-1)

    def rotor_from_params(self, params: "torch.Tensor") -> Rotor:
        """Construct a normalized Cl(3,0) rotor from 4 parameters.

        A rotor is an even-grade element: R = a + b*e12 + c*e13 + d*e23.
        The rotor is normalized so that R R† = 1.

        Args:
            params: Parameters of shape (..., 4) → [scalar, e12, e13, e23].

        Returns:
            Normalized Rotor.
        """
        import torch

        # Normalize to unit rotor
        norm = torch.norm(params, dim=-1, keepdim=True)
        normalized = params / (norm + 1e-8)

        shape = params.shape[:-1]
        components = torch.zeros(*shape, N_COMPONENTS, dtype=params.dtype, device=params.device)
        components[..., SCALAR] = normalized[..., 0]
        components[..., E12] = normalized[..., 1]
        components[..., E13] = normalized[..., 2]
        components[..., E23] = normalized[..., 3]

        return Rotor(components=components)

    def reverse(self, rotor: Rotor) -> Rotor:
        """Compute the reverse (†) of a rotor: reverses the order of basis vectors.

        For even elements: scalar unchanged, bivectors negated.
        R† = a - b*e12 - c*e13 - d*e23
        """
        import torch

        rev = rotor.components.clone()
        # Negate bivector components (grade 2)
        rev[..., E12] = -rev[..., E12]
        rev[..., E13] = -rev[..., E13]
        rev[..., E23] = -rev[..., E23]
        return Rotor(components=rev)

    def geometric_product(
        self, a: "torch.Tensor", b: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute the geometric product of two Cl(3,0) multivectors.

        Uses the multiplication table for Cl(3,0) with signature (+,+,+).
        Supports broadcasting between a and b.

        Args:
            a, b: Multivectors of shape (..., 8).

        Returns:
            Product multivector of shape (broadcast(...), 8).
        """
        import torch

        # Broadcast to common shape
        output_shape = torch.broadcast_shapes(a.shape, b.shape)
        a = a.expand(output_shape)
        b = b.expand(output_shape)
        result = torch.zeros(output_shape, dtype=a.dtype, device=a.device)

        # Scalar component
        result[..., SCALAR] = (
            a[..., SCALAR] * b[..., SCALAR]
            + a[..., E1] * b[..., E1]
            + a[..., E2] * b[..., E2]
            + a[..., E3] * b[..., E3]
            - a[..., E12] * b[..., E12]
            - a[..., E13] * b[..., E13]
            - a[..., E23] * b[..., E23]
            - a[..., E123] * b[..., E123]
        )

        # e1 component
        result[..., E1] = (
            a[..., SCALAR] * b[..., E1]
            + a[..., E1] * b[..., SCALAR]
            - a[..., E2] * b[..., E12]
            - a[..., E3] * b[..., E13]
            + a[..., E12] * b[..., E2]
            + a[..., E13] * b[..., E3]
            - a[..., E23] * b[..., E123]
            - a[..., E123] * b[..., E23]
        )

        # e2 component
        result[..., E2] = (
            a[..., SCALAR] * b[..., E2]
            + a[..., E1] * b[..., E12]
            + a[..., E2] * b[..., SCALAR]
            - a[..., E3] * b[..., E23]
            - a[..., E12] * b[..., E1]
            + a[..., E23] * b[..., E3]
            + a[..., E13] * b[..., E123]
            + a[..., E123] * b[..., E13]
        )

        # e3 component
        result[..., E3] = (
            a[..., SCALAR] * b[..., E3]
            + a[..., E1] * b[..., E13]
            + a[..., E2] * b[..., E23]
            + a[..., E3] * b[..., SCALAR]
            - a[..., E13] * b[..., E1]
            - a[..., E23] * b[..., E2]
            - a[..., E12] * b[..., E123]
            - a[..., E123] * b[..., E12]
        )

        # e12 component
        result[..., E12] = (
            a[..., SCALAR] * b[..., E12]
            + a[..., E1] * b[..., E2]
            - a[..., E2] * b[..., E1]
            + a[..., E3] * b[..., E123]
            + a[..., E12] * b[..., SCALAR]
            - a[..., E13] * b[..., E23]
            + a[..., E23] * b[..., E13]
            + a[..., E123] * b[..., E3]
        )

        # e13 component
        result[..., E13] = (
            a[..., SCALAR] * b[..., E13]
            + a[..., E1] * b[..., E3]
            - a[..., E2] * b[..., E123]
            - a[..., E3] * b[..., E1]
            + a[..., E12] * b[..., E23]
            + a[..., E13] * b[..., SCALAR]
            - a[..., E23] * b[..., E12]
            - a[..., E123] * b[..., E2]
        )

        # e23 component
        result[..., E23] = (
            a[..., SCALAR] * b[..., E23]
            + a[..., E1] * b[..., E123]
            + a[..., E2] * b[..., E3]
            - a[..., E3] * b[..., E2]
            - a[..., E12] * b[..., E13]
            + a[..., E13] * b[..., E12]
            + a[..., E23] * b[..., SCALAR]
            + a[..., E123] * b[..., E1]
        )

        # e123 component (pseudoscalar)
        result[..., E123] = (
            a[..., SCALAR] * b[..., E123]
            + a[..., E1] * b[..., E23]
            - a[..., E2] * b[..., E13]
            + a[..., E3] * b[..., E12]
            + a[..., E12] * b[..., E3]
            - a[..., E13] * b[..., E2]
            + a[..., E23] * b[..., E1]
            + a[..., E123] * b[..., SCALAR]
        )

        return result

    def sandwich_product(self, rotor: Rotor, x: "torch.Tensor") -> "torch.Tensor":
        """Apply rotor sandwich product: R x R†.

        This is the core rotation operation. For d=128, this requires
        ~100 FMAs compared to 16,384 for dense matrix rotation.

        Args:
            rotor: Normalized rotor.
            x: Grade-1 multivector of shape (..., 8).

        Returns:
            Rotated multivector of shape (..., 8).
        """
        rev = self.reverse(rotor)
        rx = self.geometric_product(rotor.components, x)
        return self.geometric_product(rx, rev.components)
