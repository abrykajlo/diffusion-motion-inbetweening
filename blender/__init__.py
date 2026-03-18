bl_info = {
    "name": "Diffusion Motion Inbetweening",
    "author": "DMI",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Motion Inbetween",
    "description": (
        "Create SMPL armatures, mark joint constraints per frame, run diffusion "
        "motion inbetweening inference, and import the generated animation."
    ),
    "category": "Animation",
}

import bpy
from bpy.props import PointerProperty

from .ui import DMI_AddonPreferences, DMI_Properties, DMI_PT_Panel
from .operators import classes as operator_classes


_classes = (
    DMI_AddonPreferences,
    DMI_Properties,
    *operator_classes,
    DMI_PT_Panel,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dmi_props = PointerProperty(type=DMI_Properties)


def unregister():
    del bpy.types.Scene.dmi_props
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
