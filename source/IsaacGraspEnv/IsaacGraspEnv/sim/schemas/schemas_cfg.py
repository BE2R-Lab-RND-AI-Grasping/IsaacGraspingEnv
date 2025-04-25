# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal, Tuple

from isaaclab.utils import configclass

from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg

@configclass
class SDFCollisionPropertiesCfg:
    """Properties to apply to SDF colliders in a rigid body.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    sdf_bits_per_subgrid_pixel: Literal[ "BitsPerPixel8", "BitsPerPixel16", "BitsPerPixel32"] | None = None
    """Values of 8, 16 and 32 bits per subgrid pixel are supported.

    Dense SDFs always use 32 bits per pixel. The less bits per pixel, the smaller the resulting SDF
    but also the less precise. The SDF's memory consumption scales proportionally with the number of bits
    per subgrid pixel.
    """
    
    sdf_enable_remeshing: bool | None = None
    """Enables optional remeshing as a preprocessing step before the SDF is computed.

    Remeshing can help generate valid SDF data even if the input mesh has bad properties like inconsistent winding
    or self-intersections. The SDF distances (and therefore the collisions) will be slightly less accurate when remeshing
    is enabled.
    """
    
    sdf_margin: float | None = None
    """Margin to increase the size of the SDF relative to the bounding box diagonal length of the mesh.

    A sdf margin value of 0.01 means the sdf boundary will be enlarged in any direction by 1% of 
    the mesh's bounding box diagonal length. Representing the margin relative to the bounding box diagonal 
    length ensures that it is scale independent. Margins allow for precise distance queries in a region slightly 
    outside of the mesh's bounding box. Range: [0, inf) Units: dimensionless
    """

    sdf_narrow_band_thickness: float | None = None
    """Size of the narrow band around the mesh surface where high resolution SDF samples are available.
    
    Outside of the narrow band, only low resolution samples are stored. Representing the narrow band
    thickness as a fraction of the mesh's bounding box diagonal length ensures that it is scale independent.
    A value of 0.01 is usually large enough. The smaller the narrow band thickness, the smaller the memory 
    consumption of the sparse SDF. Range: [0, 1] Units: dimensionless
    """

    sdf_resolution: int | None = None
    """The spacing of the uniformly sampled SDF is equal to the largest AABB extent of the mesh, divided by the resolution.
    
    Choose the lowest possible resolution that provides acceptable performance; very high resolution results in large 
    memory consumption, and slower cooking and simulation performance. Range: (1, inf)"""

    sdf_subgrid_resolution: int | None = None
    """A positive subgrid resolution enables sparsity on signed-distance-fields (SDF) while a value of 0 leads to the usage of a dense SDF.
    
    A value in the range of 4 to 8 is a reasonable compromise between block size and the overhead introduced by block addressing. 
    The smaller a block, the more memory is spent on the address table. The bigger a block, the less precisely the sparse SDF
    can adapt to the mesh's surface. In most cases sparsity reduces the memory consumption of a SDF significantly. Range: [0, inf)
    """

    sdf_triangle_count_reduction_factor: float | None = None
    """Factor that quantifies the percentage of the input triangles to keep. 
    
    1 means the input triangle mesh does not get modified. 0.5 would mean that the triangle count gets reduced to half
    the amount of the original mesh such that the collider needs to process less data. This helps to speed up collision
    detection at the cost of a small geometric error. Range: [0, 1] Units: dimensionless
    """


@configclass
class ExtendedCollisionPropertiesCfg(CollisionPropertiesCfg):
    """Properties to apply to colliders in a rigid body.

    See :meth:`modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    collision_enabled: bool | None = None
    """Whether to enable or disable collisions."""

    contact_offset: float | None = None
    """Contact offset for the collision shape (in m).

    The collision detector generates contact points as soon as two shapes get closer than the sum of their
    contact offsets. This quantity should be non-negative which means that contact generation can potentially start
    before the shapes actually penetrate.
    """

    rest_offset: float | None = None
    """Rest offset for the collision shape (in m).

    The rest offset quantifies how close a shape gets to others at rest, At rest, the distance between two
    vertically stacked objects is the sum of their rest offsets. If a pair of shapes have a positive rest
    offset, the shapes will be separated at rest by an air gap.
    """

    torsional_patch_radius: float | None = None
    """Radius of the contact patch for applying torsional friction (in m).

    It is used to approximate rotational friction introduced by the compression of contacting surfaces.
    If the radius is zero, no torsional friction is applied.
    """

    min_torsional_patch_radius: float | None = None
    """Minimum radius of the contact patch for applying torsional friction (in m)."""

    sdf_collision_properties_cfg: SDFCollisionPropertiesCfg | None = None


@configclass
class ExtendedMassPropertiesCfg(MassPropertiesCfg):
    """Properties to define explicit mass properties of a rigid body.
    It is extened version of IsaacLab's class

    See :meth:`modify_mass_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    center_of_mass: Tuple[float] | None = None
    """Center of mass in the prim's local space.
    Units: distance.
    """
    
    principal_axes: Tuple[float] | None = None
    """Orientation of the inertia tensor's principal axes in the prim's local space.
    Orientation is setted in quaternion notation
    """
    
    diagonal_inertia: Tuple[float] | None = None
    """If non-zero, specifies diagonalized inertia tensor along the principal axes.

    Note: if diagonalInertial is (0.0, 0.0, 0.0) it is ignored. Units: mass*distance*distance.
    """