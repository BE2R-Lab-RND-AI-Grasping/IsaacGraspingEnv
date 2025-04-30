# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: Usd.Stage | None
from __future__ import annotations


import isaacsim.core.utils.stage as stage_utils
from pxr import PhysxSchema, Usd, UsdPhysics

from isaaclab.sim.utils import (
    apply_nested,
    safe_set_attribute_on_usd_schema,
)
from isaaclab.sim import schemas as lab_schemas_cfg
from . import schemas_cfg


"""
Collision properties.
"""


def define_collision_properties(
    prim_path: str,
    cfg: (
        schemas_cfg.ExtendedCollisionPropertiesCfg
        | lab_schemas_cfg.CollisionPropertiesCfg
    ),
    stage: Usd.Stage | None = None,
):
    """Apply the collision schema on the input prim and set its properties.

    See :func:`modify_collision_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the collider.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
    # set collision properties
    modify_collision_properties(prim_path, cfg, stage)


@apply_nested
def modify_collision_properties(
    prim_path: str,
    cfg: (
        schemas_cfg.ExtendedCollisionPropertiesCfg
        | lab_schemas_cfg.CollisionPropertiesCfg
    ),
    stage: Usd.Stage | None = None,
) -> bool:
    """Modify PhysX properties of collider prim.

    These properties are based on the `UsdPhysics.CollisionAPI`_ and `PhysxSchema.PhysxCollisionAPI`_ schemas.
    For more information on the properties, please refer to the official documentation.

    Tuning these parameters influence the contact behavior of the rigid body. For more information on
    tune them and their effect on the simulation, please refer to the
    `PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html>`__.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _UsdPhysics.CollisionAPI: https://openusd.org/dev/api/class_usd_physics_collision_a_p_i.html
    .. _PhysxSchema.PhysxCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_collision_a_p_i.html

    Args:
        prim_path: The prim path of parent.
        cfg: The configuration for the collider.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    collider_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(collider_prim):
        return False
    # retrieve the USD collision api
    usd_collision_api = UsdPhysics.CollisionAPI(collider_prim)
    # retrieve the collision api

    physx_collision_api = PhysxSchema.PhysxCollisionAPI(collider_prim)
    if not physx_collision_api:
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)

    physx_sdf_collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI(collider_prim)
    if not physx_sdf_collision_api:
        physx_sdf_collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(
            collider_prim
        )

    # convert to dict
    cfg = cfg.to_dict()

    # set into USD API
    for attr_name in ["collision_enabled"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(
            usd_collision_api, attr_name, value, camel_case=True
        )
    # set into PhysX SDFMeshCollisionAPI
    for attr_name in ["sdf_collision_properties_cfg"]:
        sdf_cfg = cfg.pop(attr_name, None)
        for attr_name, value in sdf_cfg.items():
            safe_set_attribute_on_usd_schema(
                physx_sdf_collision_api, attr_name, value, camel_case=True
            )
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(
            physx_collision_api, attr_name, value, camel_case=True
        )
    # success
    return True


"""
Mass properties.
"""


def define_mass_properties(
    prim_path: str,
    cfg: schemas_cfg.ExtendedMassPropertiesCfg | lab_schemas_cfg.MassPropertiesCfg,
    stage: Usd.Stage | None = None,
):
    """Apply the mass schema on the input prim and set its properties.

    See :func:`modify_mass_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the mass properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has mass applied on it
    if not UsdPhysics.MassAPI(prim):
        UsdPhysics.MassAPI.Apply(prim)
    # set mass properties
    modify_mass_properties(prim_path, cfg, stage)


@apply_nested
def modify_mass_properties(
    prim_path: str,
    cfg: schemas_cfg.ExtendedMassPropertiesCfg | lab_schemas_cfg.MassPropertiesCfg,
    stage: Usd.Stage | None = None,
) -> bool:
    """Set properties for the mass of a rigid body prim.

    These properties are based on the `UsdPhysics.MassAPI` schema. If the mass is not defined, the density is used
    to compute the mass. However, in that case, a collision approximation of the rigid body is used to
    compute the density. For more information on the properties, please refer to the
    `documentation <https://openusd.org/release/wp_rigid_body_physics.html#body-mass-properties>`__.

    .. caution::

        The mass of an object can be specified in multiple ways and have several conflicting settings
        that are resolved based on precedence. Please make sure to understand the precedence rules
        before using this property.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. UsdPhysics.MassAPI: https://openusd.org/dev/api/class_usd_physics_mass_a_p_i.html

    Args:
        prim_path: The prim path of the rigid body.
        cfg: The configuration for the mass properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    rigid_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has mass API applied on it
    if not UsdPhysics.MassAPI(rigid_prim):
        return False
    # retrieve the USD mass api
    usd_physics_mass_api = UsdPhysics.MassAPI(rigid_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # set into USD API
    for attr_name in ["mass", "density", "center_of_mass", "diagonal_inertia", "principal_axes"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(
            usd_physics_mass_api, attr_name, value, camel_case=True
        )
    # success
    return True
