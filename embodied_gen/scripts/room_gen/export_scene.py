# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# Some code comes from: https://github.com/princeton-vl/infinigen/blob/main/infinigen/tools/export.py

import argparse
import logging
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
import coacd
import gin
import numpy as np
import trimesh
from infinigen.core.util import blender as butil
from mathutils import Vector

logger = logging.getLogger(__name__)

FORMAT_CHOICES = ["fbx", "obj", "usdc", "usda", "stl", "ply"]
BAKE_TYPES = {
    "DIFFUSE": "Base Color",
    "ROUGHNESS": "Roughness",
    "NORMAL": "Normal",
}  # "EMIT":"Emission Color" #  "GLOSSY": "Specular IOR Level", "TRANSMISSION":"Transmission Weight" don"t export
SPECIAL_BAKE = {"METAL": "Metallic", "TRANSMISSION": "Transmission Weight"}
ALL_BAKE = BAKE_TYPES | SPECIAL_BAKE


def apply_all_modifiers(obj):
    for mod in obj.modifiers:
        if mod is None:
            continue
        try:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=mod.name)
            logger.info(f"Applied modifier {mod} on {obj}")
            obj.select_set(False)
        except RuntimeError:
            logger.info(f"Can't apply {mod} on {obj}")
            obj.select_set(False)
            return


def realizeInstances(obj):
    for mod in obj.modifiers:
        if mod is None or mod.type != "NODES":
            continue
        geo_group = mod.node_group
        outputNode = geo_group.nodes["Group Output"]

        logger.info(f"Realizing instances on {mod}")
        link = outputNode.inputs[0].links[0]
        from_socket = link.from_socket
        geo_group.links.remove(link)
        realizeNode = geo_group.nodes.new(type="GeometryNodeRealizeInstances")
        geo_group.links.new(realizeNode.inputs[0], from_socket)
        geo_group.links.new(outputNode.inputs[0], realizeNode.outputs[0])


def remove_shade_smooth(obj):
    for mod in obj.modifiers:
        if mod is None or mod.type != "NODES":
            continue
        geo_group = mod.node_group
        outputNode = geo_group.nodes["Group Output"]
        if geo_group.nodes.get("Set Shade Smooth"):
            logger.info("Removing shade smooth on " + obj.name)
            smooth_node = geo_group.nodes["Set Shade Smooth"]
        else:
            continue

        link = smooth_node.inputs[0].links[0]
        from_socket = link.from_socket
        geo_group.links.remove(link)
        geo_group.links.new(outputNode.inputs[0], from_socket)


def check_material_geonode(node_tree):
    if node_tree.nodes.get("Set Material"):
        logger.info("Found set material!")
        return True

    for node in node_tree.nodes:
        if node.type == "GROUP" and check_material_geonode(node.node_tree):
            return True

    return False


def handle_geo_modifiers(obj, export_usd):
    has_geo_nodes = False
    for mod in obj.modifiers:
        if mod is None or mod.type != "NODES":
            continue
        has_geo_nodes = True

    if has_geo_nodes and not obj.data.materials:
        mat = bpy.data.materials.new(name=f"{mod.name} shader")
        obj.data.materials.append(mat)
        mat.use_nodes = True
        mat.node_tree.nodes.remove(mat.node_tree.nodes["Principled BSDF"])

    if not export_usd:
        realizeInstances(obj)


def split_glass_mats():
    split_objs = []
    for obj in bpy.data.objects:
        if obj.hide_render or obj.hide_viewport:
            continue
        if any(
            exclude in obj.name
            for exclude in [
                "BowlFactory",
                "CupFactory",
                "OvenFactory",
                "BottleFactory",
            ]
        ):
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            if (
                "shader_glass" in mat.name or "shader_lamp_bulb" in mat.name
            ) and len(obj.material_slots) >= 2:
                logger.info(f"Splitting {obj}")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.separate(type="MATERIAL")
                bpy.ops.object.mode_set(mode="OBJECT")
                obj.select_set(False)
                split_objs.append(obj.name)
                break

    matches = [
        obj
        for split_obj in split_objs
        for obj in bpy.data.objects
        if split_obj in obj.name
    ]
    for match in matches:
        if len(match.material_slots) == 0 or match.material_slots[0] is None:
            continue
        mat = match.material_slots[0].material
        if mat is None:
            continue
        if "shader_glass" in mat.name or "shader_lamp_bulb" in mat.name:
            match.name = f"{match.name}_SPLIT_GLASS"


def clean_names(obj=None):
    if obj is not None:
        obj.name = (obj.name).replace(" ", "_")
        obj.name = (obj.name).replace(".", "_")

        if obj.type == "MESH":
            for uv_map in obj.data.uv_layers:
                uv_map.name = uv_map.name.replace(".", "_")

        for mat in bpy.data.materials:
            if mat is None:
                continue
            mat.name = (mat.name).replace(" ", "_")
            mat.name = (mat.name).replace(".", "_")

        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            mat.name = (mat.name).replace(" ", "_")
            mat.name = (mat.name).replace(".", "_")
        return

    for obj in bpy.data.objects:
        obj.name = (obj.name).replace(" ", "_")
        obj.name = (obj.name).replace(".", "_")

        if obj.type == "MESH":
            for uv_map in obj.data.uv_layers:
                uv_map.name = uv_map.name.replace(
                    ".", "_"
                )  # if uv has "." in name the node will export wrong in USD

    for mat in bpy.data.materials:
        if mat is None:
            continue
        mat.name = (mat.name).replace(" ", "_")
        mat.name = (mat.name).replace(".", "_")


def remove_obj_parents(obj=None):
    if obj is not None:
        world_matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world_matrix
        return

    for obj in bpy.data.objects:
        world_matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world_matrix


def remove_placeholder_area_lights() -> int:
    removed_count = 0
    for obj in list(bpy.data.objects):
        if obj.type != "LIGHT" or obj.data.type != "AREA":
            continue

        parent_name = obj.parent.name if obj.parent is not None else ""
        if "WindowFactory" not in parent_name:
            continue
        if not obj.name.startswith("Area"):
            continue
        if not math.isclose(float(obj.data.energy), 10.0, abs_tol=1e-4):
            continue

        world_loc = np.array(obj.matrix_world.translation)
        if not np.allclose(world_loc, 0.0, atol=1e-4):
            continue

        bpy.data.objects.remove(obj, do_unlink=True)
        removed_count += 1

    if removed_count > 0:
        logger.info(
            "Removed placeholder window area lights before export: "
            f"{removed_count}"
        )
    return removed_count


def _get_export_scene_bounds() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    positions = []
    view_objs = set(bpy.context.view_layer.objects)
    for obj in bpy.data.objects:
        if (
            obj.type != "MESH"
            or obj.data is None
            or not obj.data.vertices
            or obj.hide_render
            or obj not in view_objs
        ):
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            positions.append(np.array(world_corner))

    if not positions:
        return None

    points = np.stack(positions)
    return points.min(axis=0), points.max(axis=0)


def _get_world_background_strength() -> float:
    world = bpy.context.scene.world
    if world is None or not world.use_nodes:
        return 0.25

    strengths = []
    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            strengths.append(float(node.inputs["Strength"].default_value))

    if not strengths:
        return 0.25
    return max(strengths)


def _get_world_sky_rotation() -> tuple[float, float]:
    world = bpy.context.scene.world
    if world is None or not world.use_nodes:
        return (math.radians(55.0), 0.0)

    for node in world.node_tree.nodes:
        if node.type != "TEX_SKY":
            continue
        elevation = float(getattr(node, "sun_elevation", math.radians(35.0)))
        rotation = float(getattr(node, "sun_rotation", 0.0))
        return (math.pi * 0.5 - elevation, rotation)

    return (math.radians(55.0), 0.0)


def add_world_export_lights(
    world_strength: float = 8.0,
) -> list[bpy.types.Object]:
    world = bpy.context.scene.world
    if world is None:
        return []

    bounds = _get_export_scene_bounds()
    if bounds is None:
        return []

    min_corner, max_corner = bounds
    center = (min_corner + max_corner) * 0.5
    diagonal = float(np.linalg.norm(max_corner - min_corner))
    strength = max(_get_world_background_strength(), world_strength)
    sun_pitch, sun_yaw = _get_world_sky_rotation()

    created_lights = []

    bpy.ops.object.light_add(
        type="SUN",
        location=(
            float(center[0]),
            float(center[1]),
            float(max_corner[2] + diagonal),
        ),
        rotation=(sun_pitch, 0.0, sun_yaw),
    )
    sun = bpy.context.object
    sun.name = "__EXPORT_WORLD_SUN__"
    sun.data.energy = max(strength * 2.0, 0.5)
    created_lights.append(sun)

    bpy.ops.object.light_add(
        type="AREA",
        location=(
            float(center[0]),
            float(center[1]),
            float(max_corner[2] + 0.5 * diagonal),
        ),
        rotation=(0.0, 0.0, 0.0),
    )
    area = bpy.context.object
    area.name = "__EXPORT_WORLD_AREA__"
    area.data.shape = "DISK"
    area.data.size = max(diagonal, 2.0)
    area.data.energy = max(strength * 2500.0, 500.0)
    created_lights.append(area)

    logger.info(
        "Added temporary world export lights: "
        f"{[obj.name for obj in created_lights]}"
    )
    return created_lights


def remove_temp_export_objects(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        if obj is None:
            continue
        if obj.name not in bpy.data.objects:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)


def delete_objects():
    logger.info("Deleting placeholders collection")
    collection_name = "placeholders"
    collection = bpy.data.collections.get(collection_name)

    if collection:
        for scene in bpy.data.scenes:
            if collection.name in scene.collection.children:
                scene.collection.children.unlink(collection)

        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        def delete_child_collections(parent_collection):
            for child_collection in parent_collection.children:
                delete_child_collections(child_collection)
                bpy.data.collections.remove(child_collection)

        delete_child_collections(collection)
        bpy.data.collections.remove(collection)

    if bpy.data.objects.get("Grid"):
        bpy.data.objects.remove(bpy.data.objects["Grid"], do_unlink=True)

    if bpy.data.objects.get("atmosphere"):
        bpy.data.objects.remove(bpy.data.objects["atmosphere"], do_unlink=True)

    if bpy.data.objects.get("KoleClouds"):
        bpy.data.objects.remove(bpy.data.objects["KoleClouds"], do_unlink=True)


def rename_all_meshes(obj=None):
    if obj is not None:
        if obj.data and obj.data.users == 1:
            obj.data.name = obj.name
        return

    for obj in bpy.data.objects:
        if obj.data and obj.data.users == 1:
            obj.data.name = obj.name


def update_visibility():
    outliner_area = next(
        a for a in bpy.context.screen.areas if a.type == "OUTLINER"
    )
    space = outliner_area.spaces[0]
    space.show_restrict_column_viewport = (
        True  # Global visibility (Monitor icon)
    )
    collection_view = {}
    obj_view = {}
    for collection in bpy.data.collections:
        collection_view[collection] = collection.hide_render
        collection.hide_viewport = False  # reenables viewports for all
        collection.hide_render = False  # enables renders for all collections

    # disables viewports and renders for all objs
    for obj in bpy.data.objects:
        obj_view[obj] = obj.hide_render
        obj.hide_viewport = True
        obj.hide_render = True
        obj.hide_set(0)

    return collection_view, obj_view


def uv_unwrap(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.data.uv_layers.new(name="ExportUV")
    bpy.context.object.data.uv_layers["ExportUV"].active = True

    logger.info("UV Unwrapping")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.smart_project(angle_limit=0.7)
    except RuntimeError:
        logger.info("UV Unwrap failed, skipping mesh")
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
        return False
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return True


def bakeVertexColors(obj):
    logger.info(f"Baking vertex color on {obj}")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    vertColor = bpy.context.object.data.color_attributes.new(
        name="VertColor", domain="CORNER", type="BYTE_COLOR"
    )
    bpy.context.object.data.attributes.active_color = vertColor
    bpy.ops.object.bake(
        type="DIFFUSE", pass_filter={"COLOR"}, target="VERTEX_COLORS"
    )
    obj.select_set(False)


def apply_baked_tex(obj, paramDict={}):
    bpy.context.view_layer.objects.active = obj
    bpy.context.object.data.uv_layers["ExportUV"].active_render = True
    for uv_layer in reversed(obj.data.uv_layers):
        if "ExportUV" not in uv_layer.name:
            logger.info(f"Removed extraneous UV Layer {uv_layer}")
            obj.data.uv_layers.remove(uv_layer)

    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        logger.info("Reapplying baked texs on " + mat.name)

        # delete all nodes except baked nodes and bsdf
        excludedNodes = [type + "_node" for type in ALL_BAKE]
        excludedNodes.extend(["Material Output", "Principled BSDF"])
        for n in nodes:
            if n.name not in excludedNodes:
                nodes.remove(
                    n
                )  # deletes an arbitrary principled BSDF in the case of a mix, which is handled below

        output = nodes["Material Output"]

        # stick baked texture in material
        if nodes.get("Principled BSDF") is None:  # no bsdf
            logger.info("No BSDF, creating new one")
            principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
        elif (
            len(output.inputs[0].links) != 0
            and output.inputs[0].links[0].from_node.bl_idname
            == "ShaderNodeBsdfPrincipled"
        ):  # trivial bsdf graph
            logger.info("Trivial shader graph, using old BSDF")
            principled_bsdf_node = nodes["Principled BSDF"]
        else:
            logger.info("Non-trivial shader graph, creating new BSDF")
            nodes.remove(
                nodes["Principled BSDF"]
            )  # shader graph was a mix of bsdfs
            principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")

        links = mat.node_tree.links

        # create the new shader node links
        links.new(output.inputs[0], principled_bsdf_node.outputs[0])
        for type in ALL_BAKE:
            if not nodes.get(type + "_node"):
                continue
            tex_node = nodes[type + "_node"]
            if type == "NORMAL":
                normal_node = nodes.new("ShaderNodeNormalMap")
                links.new(normal_node.inputs["Color"], tex_node.outputs[0])
                links.new(
                    principled_bsdf_node.inputs[ALL_BAKE[type]],
                    normal_node.outputs[0],
                )
                continue
            links.new(
                principled_bsdf_node.inputs[ALL_BAKE[type]],
                tex_node.outputs[0],
            )

        # bring back cleared param values
        if mat.name in paramDict:
            principled_bsdf_node.inputs["Metallic"].default_value = paramDict[
                mat.name
            ]["Metallic"]
            principled_bsdf_node.inputs["Sheen Weight"].default_value = (
                paramDict[mat.name]["Sheen Weight"]
            )
            principled_bsdf_node.inputs["Coat Weight"].default_value = (
                paramDict[mat.name]["Coat Weight"]
            )


def create_glass_shader(node_tree, export_usd):
    nodes = node_tree.nodes
    if nodes.get("Glass BSDF"):
        color = nodes["Glass BSDF"].inputs[0].default_value
        roughness = nodes["Glass BSDF"].inputs[1].default_value
        ior = nodes["Glass BSDF"].inputs[2].default_value

    if nodes.get("Principled BSDF"):
        nodes.remove(nodes["Principled BSDF"])

    principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")

    if nodes.get("Glass BSDF"):
        principled_bsdf_node.inputs["Base Color"].default_value = color
        principled_bsdf_node.inputs["Roughness"].default_value = roughness
        principled_bsdf_node.inputs["IOR"].default_value = ior
    else:
        principled_bsdf_node.inputs["Roughness"].default_value = 0

    principled_bsdf_node.inputs["Transmission Weight"].default_value = 1
    if export_usd:
        principled_bsdf_node.inputs["Alpha"].default_value = 0.6
    node_tree.links.new(
        principled_bsdf_node.outputs[0], nodes["Material Output"].inputs[0]
    )


def process_glass_materials(obj, export_usd):
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        outputNode = nodes["Material Output"]
        if nodes.get("Glass BSDF"):
            if (
                outputNode.inputs[0].links[0].from_node.bl_idname
                == "ShaderNodeBsdfGlass"
            ):
                logger.info(f"Creating glass material on {obj.name}")
            else:
                logger.info(
                    f"Non-trivial glass material on {obj.name}, material export will be inaccurate"
                )
            create_glass_shader(mat.node_tree, export_usd)
        elif "glass" in mat.name or "shader_lamp_bulb" in mat.name:
            logger.info(f"Creating glass material on {obj.name}")
            create_glass_shader(mat.node_tree, export_usd)


def bake_pass(
    obj, dest: Path, img_size, bake_type, export_usd, export_name=None
):
    if export_name is None:
        img = bpy.data.images.new(
            f"{obj.name}_{bake_type}", img_size, img_size
        )
        clean_name = (
            (obj.name).replace(" ", "_").replace(".", "_").replace("/", "_")
        )
        clean_name = (
            clean_name.replace("(", "_").replace(")", "").replace("-", "_")
        )
        file_path = dest / f"{clean_name}_{bake_type}.png"
    else:
        img = bpy.data.images.new(
            f"{export_name}_{bake_type}", img_size, img_size
        )
        file_path = dest / f"{export_name}_{bake_type}.png"
    dest = dest / "textures"

    bake_obj = False
    bake_exclude_mats = {}

    # materials are stored as stack so when removing traverse the reversed list
    for index, slot in reversed(list(enumerate(obj.material_slots))):
        mat = slot.material
        if mat is None:
            bpy.context.object.active_material_index = index
            bpy.ops.object.material_slot_remove()
            continue

        logger.info(mat.name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        output = nodes["Material Output"]

        img_node = nodes.new("ShaderNodeTexImage")
        img_node.name = f"{bake_type}_node"
        img_node.image = img
        img_node.select = True
        nodes.active = img_node
        img_node.select = True

        if len(output.inputs["Displacement"].links) != 0:
            bake_obj = True

        if len(output.inputs[0].links) == 0:
            logger.info(
                f"{mat.name} has no surface output, not using baked textures"
            )
            bake_exclude_mats[mat] = img_node
            continue

        # surface_node = output.inputs[0].links[0].from_node
        # if (
        #     bake_type in ALL_BAKE
        #     and surface_node.bl_idname == "ShaderNodeBsdfPrincipled"
        #     and len(surface_node.inputs[ALL_BAKE[bake_type]].links) == 0
        # ):  # trivial bsdf graph
        #     logger.info(
        #         f"{mat.name} has no procedural input for {bake_type}, not using baked textures"
        #     )
        #     bake_exclude_mats[mat] = img_node
        #     continue

        bake_obj = True

    if bake_type in SPECIAL_BAKE:
        internal_bake_type = "EMIT"
    else:
        internal_bake_type = bake_type

    if bake_obj:
        logger.info(f"Baking {bake_type} pass")
        bpy.ops.object.bake(
            type=internal_bake_type,
            pass_filter={"COLOR"},
            save_mode="EXTERNAL",
        )
        img.filepath_raw = str(file_path)
        img.save()
        logger.info(f"Saving to {file_path}")
    else:
        logger.info(
            f"No necessary materials to bake on {obj.name}, skipping bake"
        )

    for mat, img_node in bake_exclude_mats.items():
        mat.node_tree.nodes.remove(img_node)


def bake_special_emit(
    obj, dest, img_size, export_usd, bake_type, export_name=None
):
    # If at least one material has both a BSDF and non-zero bake type value, then bake
    should_bake = False

    # (Root node, From Socket, To Socket)
    links_removed = []
    links_added = []

    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            logger.warn("No material on mesh, skipping...")
            continue
        if not mat.use_nodes:
            logger.warn("Material has no nodes, skipping...")
            continue

        nodes = mat.node_tree.nodes
        principled_bsdf_node = None
        root_node = None
        logger.info(f"{mat.name} has {len(nodes)} nodes: {nodes}")
        for node in nodes:
            if node.type != "GROUP":
                continue

            for subnode in node.node_tree.nodes:
                logger.info(
                    f" [{subnode.type}] {subnode.name} {subnode.bl_idname}"
                )
                if subnode.type == "BSDF_PRINCIPLED":
                    logger.debug(f" BSDF_PRINCIPLED: {subnode.inputs}")
                    principled_bsdf_node = subnode
                    root_node = node

        if nodes.get("Principled BSDF"):
            principled_bsdf_node = nodes["Principled BSDF"]
            root_node = mat
        elif not principled_bsdf_node:
            logger.warn("No Principled BSDF, skipping...")
            continue
        elif ALL_BAKE[bake_type] not in principled_bsdf_node.inputs:
            logger.warn(f"No {bake_type} input, skipping...")
            continue

        # Here, we"ve found the proper BSDF and bake type input. Set up the scene graph
        # for baking.
        outputSoc = principled_bsdf_node.outputs[0].links[0].to_socket

        # Remove the BSDF link to Output first
        link = principled_bsdf_node.outputs[0].links[0]
        from_socket, to_socket = link.from_socket, link.to_socket
        logger.debug(f"Removing link: {from_socket.name} => {to_socket.name}")
        root_node.node_tree.links.remove(link)
        links_removed.append((root_node, from_socket, to_socket))

        # Get bake_type value
        bake_input = principled_bsdf_node.inputs[ALL_BAKE[bake_type]]
        bake_val = bake_input.default_value
        logger.info(f"{bake_type} value: {bake_val}")

        if bake_val > 0:
            should_bake = True

        # Make a color input matching the metallic value
        col = root_node.node_tree.nodes.new("ShaderNodeRGB")
        col.outputs[0].default_value = (bake_val, bake_val, bake_val, 1.0)

        # Link the color to output
        new_link = root_node.node_tree.links.new(col.outputs[0], outputSoc)
        links_added.append((root_node, col.outputs[0], outputSoc))
        logger.debug(
            f"Linking {col.outputs[0].name} to {outputSoc.name}({outputSoc.bl_idname}): {new_link}"
        )

    # After setting up all materials, bake if applicable
    if should_bake:
        bake_pass(obj, dest, img_size, bake_type, export_usd, export_name)

    # After baking, undo the temporary changes to the scene graph
    for n, from_soc, to_soc in links_added:
        logger.debug(
            f"Removing added link:\t{n.name}: {from_soc.name} => {to_soc.name}"
        )
        for l in n.node_tree.links:
            if l.from_socket == from_soc and l.to_socket == to_soc:
                n.node_tree.links.remove(l)
                logger.debug(
                    f"Removed link:\t{n.name}: {from_soc.name} => {to_soc.name}"
                )

    for n, from_soc, to_soc in links_removed:
        logger.debug(
            f"Adding back link:\t{n.name}: {from_soc.name} => {to_soc.name}"
        )
        n.node_tree.links.new(from_soc, to_soc)


def remove_params(mat, node_tree):
    nodes = node_tree.nodes
    paramDict = {}
    if nodes.get("Material Output"):
        output = nodes["Material Output"]
    elif nodes.get("Group Output"):
        output = nodes["Group Output"]
    else:
        raise ValueError("Could not find material output node")

    if (
        nodes.get("Principled BSDF")
        and output.inputs[0].links[0].from_node.bl_idname
        == "ShaderNodeBsdfPrincipled"
    ):
        principled_bsdf_node = nodes["Principled BSDF"]
        metal = principled_bsdf_node.inputs[
            "Metallic"
        ].default_value  # store metallic value and set to 0
        sheen = principled_bsdf_node.inputs["Sheen Weight"].default_value
        clearcoat = principled_bsdf_node.inputs["Coat Weight"].default_value
        paramDict[mat.name] = {
            "Metallic": metal,
            "Sheen Weight": sheen,
            "Coat Weight": clearcoat,
        }
        principled_bsdf_node.inputs["Metallic"].default_value = 0
        principled_bsdf_node.inputs["Sheen Weight"].default_value = 0
        principled_bsdf_node.inputs["Coat Weight"].default_value = 0
        return paramDict

    for node in nodes:
        if node.type == "GROUP":
            paramDict = remove_params(mat, node.node_tree)
            if len(paramDict) != 0:
                return paramDict

    return paramDict


def process_interfering_params(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        paramDict = remove_params(mat, mat.node_tree)
    return paramDict


def skipBake(obj):
    if not obj.data.materials:
        logger.info("No material on mesh, skipping...")
        return True

    if len(obj.data.vertices) == 0:
        logger.info("Mesh has no vertices, skipping ...")
        return True

    return False


def triangulate_mesh(obj: bpy.types.Object):
    logger.debug("Triangulating Mesh")
    if obj.type == "MESH":
        view_state = obj.hide_viewport
        obj.hide_viewport = False
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        logger.debug(f"Triangulating {obj}")
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
        obj.hide_viewport = view_state


def triangulate_meshes():
    logger.debug("Triangulating Meshes")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            view_state = obj.hide_viewport
            obj.hide_viewport = False
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            logger.debug(f"Triangulating {obj}")
            bpy.ops.mesh.quads_convert_to_tris()
            bpy.ops.object.mode_set(mode="OBJECT")
            obj.select_set(False)
            obj.hide_viewport = view_state


def adjust_wattages():
    logger.info("Keeping original point light wattage for USD export")
    for obj in bpy.context.scene.objects:
        if obj.type == "LIGHT" and obj.data.type == "POINT":
            light = obj.data
            if hasattr(light, "energy"):
                light.energy = float(light.energy)


def set_center_of_mass():
    logger.info("Resetting center of mass of objects")
    for obj in bpy.context.scene.objects:
        if not obj.hide_render:
            view_state = obj.hide_viewport
            obj.hide_viewport = False
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
            obj.select_set(False)
            obj.hide_viewport = view_state


def duplicate_node_groups(node_tree, group_map=None):
    if group_map is None:
        group_map = {}

    for node in node_tree.nodes:
        if node.type == "GROUP":
            group = node.node_tree
            if group not in group_map:
                group_copy = group.copy()
                group_copy.name = f"{group.name}_copy"
                group_map[group] = group_copy

                duplicate_node_groups(group_copy, group_map)
            else:
                group_copy = group_map[group]

            node.node_tree = group_copy

    return group_map


def deep_copy_material(original_material, new_name_suffix="_deepcopy"):
    new_mat = original_material.copy()
    new_mat.name = original_material.name + new_name_suffix
    if new_mat.use_nodes and new_mat.node_tree:
        duplicate_node_groups(new_mat.node_tree)
    return new_mat


def bake_object(obj, dest, img_size, export_usd, export_name=None):
    if not uv_unwrap(obj):
        return

    bpy.ops.object.select_all(action="DESELECT")

    with butil.SelectObjects(obj):
        for slot in obj.material_slots:
            mat = slot.material
            if mat is not None:
                slot.material = deep_copy_material(
                    mat
                )  # we duplicate in the case of distinct meshes sharing materials

        process_glass_materials(obj, export_usd)

        for bake_type in SPECIAL_BAKE:
            bake_special_emit(
                obj, dest, img_size, export_usd, bake_type, export_name
            )

        # bake_normals(obj, dest, img_size, export_usd)
        paramDict = process_interfering_params(obj)
        for bake_type in BAKE_TYPES:
            bake_pass(obj, dest, img_size, bake_type, export_usd, export_name)

        apply_baked_tex(obj, paramDict)


def bake_scene(folderPath: Path, image_res, vertex_colors, export_usd):
    for obj in bpy.data.objects:
        logger.info("---------------------------")
        logger.info(obj.name)

        if obj.type != "MESH" or obj not in list(
            bpy.context.view_layer.objects
        ):
            logger.info("Not mesh, skipping ...")
            continue

        if skipBake(obj):
            continue

        if format == "stl":
            continue

        obj.hide_render = False
        obj.hide_viewport = False

        if vertex_colors:
            bakeVertexColors(obj)
        else:
            bake_object(obj, folderPath, image_res, export_usd)

        obj.hide_render = True
        obj.hide_viewport = True


def run_blender_export(
    exportPath: Path,
    format: str,
    vertex_colors: bool,
    individual_export: bool,
    world_strength: float = 8.0,
):
    assert exportPath.parent.exists()
    exportPath = str(exportPath)
    temp_export_objects: list[bpy.types.Object] = []

    if format == "obj":
        if vertex_colors:
            bpy.ops.wm.obj_export(
                filepath=exportPath,
                export_colors=True,
                export_eval_mode="DAG_EVAL_RENDER",
                export_selected_objects=individual_export,
            )
        else:
            bpy.ops.wm.obj_export(
                filepath=exportPath,
                path_mode="COPY",
                export_materials=True,
                export_pbr_extensions=False,
                export_eval_mode="DAG_EVAL_RENDER",
                export_selected_objects=individual_export,
                export_triangulated_mesh=True,
                export_normals=False,
            )

    if format == "fbx":
        if vertex_colors:
            bpy.ops.export_scene.fbx(
                filepath=exportPath,
                colors_type="SRGB",
                use_selection=individual_export,
            )
        else:
            bpy.ops.export_scene.fbx(
                filepath=exportPath,
                path_mode="COPY",
                embed_textures=True,
                use_selection=individual_export,
            )

    if format == "stl":
        bpy.ops.export_mesh.stl(
            filepath=exportPath, use_selection=individual_export
        )

    if format == "ply":
        bpy.ops.wm.ply_export(
            filepath=exportPath, export_selected_objects=individual_export
        )

    if format in ["usda", "usdc"]:
        temp_export_objects = add_world_export_lights(world_strength)
        try:
            bpy.ops.wm.usd_export(
                filepath=exportPath,
                export_textures=True,
                # use_instancing=True,
                overwrite_textures=True,
                selected_objects_only=individual_export,
                root_prim_path="/World",
            )
        finally:
            remove_temp_export_objects(temp_export_objects)


def export_scene(
    input_blend: Path,
    output_folder: Path,
    pipeline_folder=None,
    task_uniqname=None,
    **kwargs,
):
    folder = output_folder / f"export_{os.path.splitext(input_blend.name)[0]}"
    folder.mkdir(exist_ok=True, parents=True)
    export_curr_scene(folder, **kwargs)

    if pipeline_folder is not None and task_uniqname is not None:
        (pipeline_folder / "logs" / f"FINISH_{task_uniqname}").touch()

    return folder


# side effects: will remove parents of inputted obj and clean its name, hides viewport of all objects
def export_single_obj(
    obj: bpy.types.Object,
    output_folder: Path,
    format="usdc",
    image_res=1024,
    vertex_colors=False,
):
    export_usd = format in ["usda", "usdc"]

    export_folder = output_folder
    export_folder.mkdir(parents=True, exist_ok=True)
    export_file = export_folder / output_folder.with_suffix(f".{format}").name

    logger.info(f"Exporting to directory {export_folder=}")

    remove_obj_parents(obj)
    rename_all_meshes(obj)

    collection_views, obj_views = update_visibility()

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1  # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
        raise ValueError("Object not mesh")

    if export_usd:
        apply_all_modifiers(obj)
    else:
        realizeInstances(obj)
        apply_all_modifiers(obj)

    if not skipBake(obj) and format != "stl":
        if vertex_colors:
            bakeVertexColors(obj)
        else:
            obj.hide_render = False
            obj.hide_viewport = False
            bake_object(obj, export_folder / "textures", image_res, export_usd)
            obj.hide_render = True
            obj.hide_viewport = True

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj, status in obj_views.items():
        obj.hide_render = status

    clean_names(obj)

    old_loc = obj.location.copy()
    obj.location = (0, 0, 0)

    if (
        obj.type != "MESH"
        or obj.hide_render
        or len(obj.data.vertices) == 0
        or obj not in list(bpy.context.view_layer.objects)
    ):
        raise ValueError("Object is not mesh or hidden from render")

    export_subfolder = export_folder / obj.name
    export_subfolder.mkdir(exist_ok=True)
    export_file = export_subfolder / f"{obj.name}.{format}"

    logger.info(f"Exporting file to {export_file=}")
    obj.hide_viewport = False
    obj.select_set(True)
    run_blender_export(
        export_file, format, vertex_colors, individual_export=True
    )
    obj.select_set(False)
    obj.location = old_loc

    return export_file


def export_sim_ready(
    obj: bpy.types.Object,
    output_folder: Path,
    image_res: int = 1024,
    translation: Tuple = (0, 0, 0),
    name: Optional[str] = None,
    visual_only: bool = False,
    collision_only: bool = False,
    separate_asset_dirs: bool = True,
) -> Dict[str, List[Path]]:
    """Exports both the visual and collision assets for a geometry."""
    if not visual_only:
        assert (
            coacd is not None
        ), "coacd is required to export simulation assets."

    asset_exports = defaultdict(list)
    export_name = name if name is not None else obj.name

    if separate_asset_dirs:
        visual_export_folder = output_folder / "visual"
        collision_export_folder = output_folder / "collision"
    else:
        visual_export_folder = output_folder
        collision_export_folder = output_folder

    texture_export_folder = output_folder / "textures"

    visual_export_folder.mkdir(parents=True, exist_ok=True)
    collision_export_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting to directory {output_folder=}")

    collection_views, obj_views = update_visibility()

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1  # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
        raise ValueError("Object not mesh")

    # export the textures
    if not skipBake(obj):
        texture_export_folder.mkdir(parents=True, exist_ok=True)
        obj.hide_render = False
        obj.hide_viewport = False
        bake_object(obj, texture_export_folder, image_res, False, export_name)
        obj.hide_render = True
        obj.hide_viewport = True

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj_tmp, status in obj_views.items():
        obj_tmp.hide_render = status

    # translating object
    old_loc = obj.location.copy()
    obj.location = (
        old_loc[0] + translation[0],
        old_loc[1] + translation[1],
        old_loc[2] + translation[2],
    )

    if (
        obj.type != "MESH"
        or obj.hide_render
        or len(obj.data.vertices) == 0
        or obj not in list(bpy.context.view_layer.objects)
    ):
        raise ValueError("Object is not mesh or hidden from render")

    # export the mesh assets
    visual_export_file = visual_export_folder / f"{export_name}.obj"

    logger.info(f"Exporting file to {visual_export_file=}")
    obj.hide_viewport = False
    obj.select_set(True)

    # export visual asset
    with butil.SelectObjects(obj, active=1):
        bpy.ops.wm.obj_export(
            filepath=str(visual_export_file),
            up_axis="Z",
            forward_axis="Y",
            export_selected_objects=True,
            export_triangulated_mesh=True,  # required for coacd to run properly
        )
    if not collision_only:
        asset_exports["visual"].append(visual_export_file)

    if visual_only:
        obj.select_set(False)
        obj.location = old_loc
        return asset_exports

    clone = butil.deep_clone_obj(obj)
    parts = butil.split_object(clone)

    part_export_obj_file = visual_export_folder / f"{export_name}_part.obj"
    part_export_mtl_file = visual_export_folder / f"{export_name}_part.mtl"

    collision_count = 0
    for part in parts:
        with butil.SelectObjects(part, active=1):
            bpy.ops.wm.obj_export(
                filepath=str(part_export_obj_file),
                up_axis="Z",
                forward_axis="Y",
                export_selected_objects=True,
                export_triangulated_mesh=True,  # required for coacd to run properly
            )

        # export the collision meshes
        mesh_tri = trimesh.load(
            str(part_export_obj_file),
            merge_norm=True,
            merge_tex=True,
            force="mesh",
        )
        trimesh.repair.fix_inversion(mesh_tri)
        preprocess_mode = "off"
        if not mesh_tri.is_volume:
            print(
                mesh_tri.is_watertight,
                mesh_tri.is_winding_consistent,
                np.isfinite(mesh_tri.center_mass).all(),
                mesh_tri.volume > 0.0,
            )
            preprocess_mode = "on"

            if len(mesh_tri.vertices) < 4:
                logger.warning(
                    f"Mesh is not a volume. Only has {len(mesh_tri.vertices)} vertices."
                )
                # raise ValueError(f"Mesh is not a volume. Only has {len(mesh_tri.vertices)} vertices.")
        mesh = coacd.Mesh(mesh_tri.vertices, mesh_tri.faces)

        subparts = coacd.run_coacd(
            mesh=mesh,
            threshold=0.05,
            max_convex_hull=-1,
            preprocess_mode=preprocess_mode,
            mcts_max_depth=3,
        )
        export_name = export_name.replace("vis", "col")
        for vs, fs in subparts:
            collision_export_file = (
                collision_export_folder
                / f"{export_name}_col{collision_count}.obj"
            )
            subpart_mesh = trimesh.Trimesh(vs, fs)

            # if subpart_mesh.is_empty:
            #     raise ValueError(
            #         "Warning: Collision mesh is completely outside the bounds of the original mesh."
            #     )
            subpart_mesh.export(str(collision_export_file))
            asset_exports["collision"].append(collision_export_file)
            collision_count += 1

    # delete temporary part files
    part_export_obj_file.unlink(missing_ok=True)
    part_export_mtl_file.unlink(missing_ok=True)

    obj.select_set(False)
    obj.location = old_loc
    butil.delete(clone)

    return asset_exports


@gin.configurable
def export_curr_scene(
    output_folder: Path,
    format="usdc",
    image_res=1024,
    vertex_colors=False,
    individual_export=False,
    omniverse_export=False,
    pipeline_folder=None,
    task_uniqname=None,
    deconvex=False,
    center_scene=False,
    world_strength=8.0,
    align_quat=(0.7071, 0, 0, 0.7071),  # xyzw
) -> Path:
    export_usd = format in ["usda", "usdc"]
    export_folder = output_folder
    export_folder.mkdir(exist_ok=True)
    export_file = export_folder / output_folder.with_suffix(f".{format}").name
    texture_export_folder = export_folder / "textures"
    bake_texture_folder = texture_export_folder
    if export_usd:
        bake_texture_folder = export_folder / "_usd_bake_textures"
    logger.info(f"Exporting to directory {export_folder=}")

    remove_placeholder_area_lights()
    remove_obj_parents()
    delete_objects()
    triangulate_meshes()
    if omniverse_export and format not in ["usda", "usdc"]:
        split_glass_mats()
    rename_all_meshes()

    # remove 0 polygon meshes
    for obj in bpy.data.objects:
        if obj.type == "MESH" and len(obj.data.polygons) == 0:
            logger.info(f"{obj.name} has no faces, removing...")
            bpy.data.objects.remove(obj, do_unlink=True)

    if center_scene:
        from mathutils import Vector

        positions = []
        view_objs = set(bpy.context.view_layer.objects)
        for obj in bpy.data.objects:
            if (
                obj.type == "MESH"
                and obj.data
                and obj.data.vertices
                and obj.data.polygons
                and not obj.hide_render
                and obj in view_objs
            ):
                pos = np.array(obj.matrix_world.translation)
                if not np.allclose(pos, 0):
                    positions.append(pos)

        if len(positions) > 0:
            positions = np.stack(positions)
            center = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
            center[2] = positions[:, 2].min()  # Set floor to 0 among z-axis.
            for obj in bpy.data.objects:
                pos = np.array(obj.matrix_world.translation)
                if not np.allclose(pos, 0):
                    obj.location -= Vector(center)

    scatter_cols = []
    if export_usd:
        if bpy.data.collections.get("scatter"):
            scatter_cols.append(bpy.data.collections["scatter"])
        if bpy.data.collections.get("scatters"):
            scatter_cols.append(bpy.data.collections["scatters"])
        for col in scatter_cols:
            for obj in col.all_objects:
                remove_shade_smooth(obj)

    collection_views, obj_views = update_visibility()
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj not in list(
            bpy.context.view_layer.objects
        ):
            continue
        if export_usd:
            apply_all_modifiers(obj)
        else:
            realizeInstances(obj)
            apply_all_modifiers(obj)

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1  # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    # iterate through all objects and bake them
    bake_scene(
        folderPath=bake_texture_folder,
        image_res=image_res,
        vertex_colors=vertex_colors,
        export_usd=export_usd,
    )

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj, status in obj_views.items():
        obj.hide_render = status

    clean_names()

    for obj in bpy.data.objects:
        obj.hide_viewport = obj.hide_render

    if omniverse_export:
        adjust_wattages()
        set_center_of_mass()

    if individual_export:
        import math
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        import trimesh
        from scipy.spatial.transform import Rotation
        from embodied_gen.data.convex_decomposer import decompose_convex_mesh

        urdf_root = ET.Element("robot", name="multi_object_scene")
        ET.SubElement(urdf_root, "link", name="base")
        object_info = []
        bpy.ops.object.select_all(action="DESELECT")
        objects = list(bpy.data.objects)
        for obj in objects:
            if (
                obj.type != "MESH"
                or obj.data is None
                or len(obj.data.vertices) == 0
                or len(obj.data.polygons) == 0
                or obj.hide_render
                or obj not in list(bpy.context.view_layer.objects)
            ):
                continue

            obj_name = obj.name.replace("/", "_").replace("-", "_")
            obj_name = obj_name.replace("(", "_").replace(")", "")
            obj.name = obj_name
            export_subfolder = export_folder / obj_name
            export_subfolder.mkdir(exist_ok=True, parents=True)
            export_file = export_subfolder / f"{obj_name}.{format}"

            if "skirtingboard" in obj_name.lower():
                logger.info(f"Skipping skirting board {obj_name}")
                continue

            logger.info(f"Exporting file to {export_file=}")
            obj.hide_viewport = False

            position = obj.matrix_world.to_translation()
            rotation = Rotation.from_quat(align_quat)
            rotation = rotation.as_euler("xyz", degrees=False)

            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.location_clear()

            face_count = len(obj.data.polygons)
            if face_count > 1000:
                if face_count > 1000000:
                    ratio = 0.005
                elif face_count > 100000:
                    ratio = 0.02
                elif face_count > 10000:
                    ratio = 0.1
                else:
                    ratio = 0.2
                angle_threshold = math.radians(5)
                bpy.ops.object.mode_set(mode="OBJECT")
                dec_mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
                dec_mod.decimate_type = "DISSOLVE"
                dec_mod.angle_limit = angle_threshold
                dec_mod.use_collapse_triangulate = False
                dec_mod.ratio = ratio
                bpy.ops.object.modifier_apply(modifier=dec_mod.name)

            run_blender_export(
                export_file,
                format,
                vertex_colors,
                individual_export,
                world_strength=world_strength,
            )
            obj.select_set(False)

            mesh = trimesh.load(export_file)
            if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) == 0:
                shutil.rmtree(export_file.parent)
                continue

            object_info.append(
                {
                    "name": obj_name,
                    "mesh_path": f"{obj_name}/{obj_name}.{format}",
                    "mesh_abs_path": str(export_file),
                    "xyz": tuple(position),
                    "rpy": tuple(rotation),
                }
            )

        for obj in object_info:
            link = ET.SubElement(urdf_root, "link", name=obj["name"])
            visual = ET.SubElement(link, "visual")
            geom = ET.SubElement(visual, "geometry")
            ET.SubElement(
                geom, "mesh", filename=obj["mesh_path"], scale="1 1 1"
            )
            if deconvex:
                print("Deconvexing mesh for collision, waiting...")
                d_params = dict(
                    threshold=0.05, max_convex_hull=128, verbose=False
                )
                mesh_path = obj["mesh_abs_path"]
                output_path = mesh_path.replace(".obj", "_collision.obj")
                decompose_convex_mesh(mesh_path, output_path, **d_params)
                collision_mesh = obj["mesh_path"].replace(
                    ".obj", "_collision.obj"
                )
                collision = ET.SubElement(link, "collision")
                geom2 = ET.SubElement(collision, "geometry")
                ET.SubElement(
                    geom2, "mesh", filename=collision_mesh, scale="1 1 1"
                )

            joint = ET.SubElement(
                urdf_root, "joint", name=f"joint_{obj['name']}", type="fixed"
            )
            ET.SubElement(joint, "parent", link="base")
            ET.SubElement(joint, "child", link=obj["name"])
            ET.SubElement(
                joint,
                "origin",
                xyz="%.4f %.4f %.4f" % obj["xyz"],
                rpy="%.4f %.4f %.4f" % obj["rpy"],
            )

        urdf_str = minidom.parseString(ET.tostring(urdf_root)).toprettyxml(
            indent="  "
        )
        urdf_path = export_folder / "scene.urdf"
        with open(urdf_path, "w") as f:
            f.write(urdf_str)
        logger.info(f"URDF exported to {urdf_path}")

        return urdf_path
    else:
        logger.info(f"Exporting file to {export_file=}")
        try:
            run_blender_export(
                export_file,
                format,
                vertex_colors,
                individual_export,
                world_strength=world_strength,
            )
        finally:
            if export_usd and bake_texture_folder.exists():
                shutil.rmtree(bake_texture_folder, ignore_errors=True)

        return export_file


def main(args):
    args.output_folder.mkdir(exist_ok=True)
    targets = sorted(list(args.input_folder.iterdir()))
    for blendfile in targets:
        if not blendfile.suffix == ".blend":
            print(f"Skipping non-blend file {blendfile}")
            continue

        bpy.ops.wm.open_mainfile(filepath=str(blendfile))

        folder = export_scene(
            blendfile,
            args.output_folder,
            format=args.format,
            image_res=args.resolution,
            vertex_colors=args.vertex_colors,
            individual_export=args.individual,
            omniverse_export=args.omniverse,
            deconvex=args.deconvex,
            center_scene=args.center_scene,
            world_strength=args.world_strength,
        )

    bpy.ops.wm.quit_blender()


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder", type=Path)
    parser.add_argument("--output_folder", type=Path)

    parser.add_argument("-f", "--format", type=str, choices=FORMAT_CHOICES)

    parser.add_argument("-v", "--vertex_colors", action="store_true")
    parser.add_argument("-r", "--resolution", default=1024, type=int)
    parser.add_argument("-i", "--individual", action="store_true")
    parser.add_argument("-o", "--omniverse", action="store_true")
    parser.add_argument("--deconvex", action="store_true")
    parser.add_argument("--center_scene", action="store_true")
    parser.add_argument("--world_strength", default=8.0, type=float)

    args = parser.parse_args()

    if args.format not in FORMAT_CHOICES:
        raise ValueError("Unsupported or invalid file format.")

    if args.vertex_colors and args.format not in ["ply", "fbx", "obj"]:
        raise ValueError("File format does not support vertex colors.")

    if args.format == "ply" and not args.vertex_colors:
        raise ValueError(".ply export must use vertex colors.")

    return args


if __name__ == "__main__":
    args = make_args()
    main(args)
