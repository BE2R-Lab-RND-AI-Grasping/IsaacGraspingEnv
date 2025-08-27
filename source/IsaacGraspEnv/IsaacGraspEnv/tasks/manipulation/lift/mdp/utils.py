from isaaclab.managers import SceneEntityCfg


def generate_contact_sensor_params():
    d_contact_cfg = {"thumb_rot_cfgs": SceneEntityCfg("contact_forces_thumb_rot"),
            "thumb_flex_cfgs": SceneEntityCfg("contact_forces_thumb_flex"),
            "thumb_finray_cfgs": SceneEntityCfg("contact_forces_thumb_finray"),
            "right_flex_cfgs": SceneEntityCfg("contact_forces_right_flex"),
            "right_finray_cfgs": SceneEntityCfg("contact_forces_right_finray"),
            "left_flex_cfgs": SceneEntityCfg("contact_forces_left_flex"),
            "left_finray_cfgs": SceneEntityCfg("contact_forces_left_finray"), "threshold": 30.0}

    add_is_contact_param = lambda b, d_contact_cfg = d_contact_cfg: b.update(d_contact_cfg) or b
    
    return add_is_contact_param