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


import os
import sys

import numpy as np


def _infinigen_path():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    return os.path.abspath(
        os.path.join(current_dir, "../../..", "thirdparty", "infinigen")
    )


def _ensure_infinigen_on_path():
    path = _infinigen_path()
    if path not in sys.path:
        sys.path.insert(0, path)


def patch_material_assignments():
    """Replace ceramic.tile with ceramic.Tile in utility_floor assignments."""
    _ensure_infinigen_on_path()
    from infinigen.assets.composition import material_assignments
    from infinigen.assets.materials import ceramic

    # utility_floor: ceramic.tile -> ceramic.Tile
    material_assignments.utility_floor = [
        (ceramic.Concrete, 1.0),
        (ceramic.Plaster, 1.0),
        (ceramic.Tile, 1.0),
    ]


def patch_concrete():
    """Filter Concrete.generate kwargs to supported keys."""
    _ensure_infinigen_on_path()
    from infinigen.assets.materials.ceramic import concrete
    from infinigen.core import surface

    shader_concrete = concrete.shader_concrete

    def patched_generate(self, **kwargs):
        # Filter out unsupported keywords and pass remaining arguments
        # Concrete.shader_concrete accepts: scale, base_color_hsv, seed, roughness, crack_amount, crack_scale, snake_crack
        supported_kwargs = {
            'scale',
            'base_color_hsv',
            'seed',
            'roughness',
            'crack_amount',
            'crack_scale',
            'snake_crack',
        }
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in supported_kwargs
        }
        return surface.shaderfunc_to_material(
            shader_concrete, **filtered_kwargs
        )

    concrete.Concrete.generate = patched_generate
    concrete.Concrete.__call__ = patched_generate


def patch_room_constants():
    """Add Office to RoomConstants.home_room_types."""
    _ensure_infinigen_on_path()
    from infinigen.core import tags as t
    from infinigen.core.constraints.constraint_language.constants import (
        RoomConstants,
    )

    _original_home_room_types = RoomConstants.home_room_types.fget

    @property
    def patched_home_room_types(self):
        return _original_home_room_types(self) | {t.Semantics.Office}

    RoomConstants.home_room_types = patched_home_room_types


def patch_doors_base_simple():
    """Override BaseDoorFactory init to customize door dimensions and handles."""
    _ensure_infinigen_on_path()
    from infinigen.assets import colors
    from infinigen.assets.composition import material_assignments
    from infinigen.assets.objects.elements.doors.base import BaseDoorFactory
    from infinigen.core.constraints.constraint_language.constants import (
        RoomConstants,
    )
    from infinigen.core.util.math import FixedSeed
    from infinigen.core.util.random import weighted_sample
    from numpy.random import uniform

    _orig_init = BaseDoorFactory.__init__

    def patched_init(self, factory_seed, coarse=False, constants=None):
        _orig_init(self, factory_seed, coarse=coarse, constants=constants)
        with FixedSeed(self.factory_seed):
            if constants is None:
                constants = RoomConstants()
            self.width = constants.door_width - 0.02
            # Force a rectangular full frame so generated doors can close
            # cleanly against the wall opening.
            self.door_frame_style = "full_frame_square"
            self.door_frame_width = 0.02
            handle_types = ["knob", "lever", "pull", "none", "bar"]
            if self.door_frame_style != "single_column":
                self.width += -0.02
                self.height += -0.04
            self.handle_type = np.random.choice(handle_types)
            if self.handle_type == "bar":
                self.surface = weighted_sample(material_assignments.metals)()
            if self.handle_type == "bar":
                self.handle_info_dict = {
                    "handle_type": self.handle_type,
                    "bar_length": uniform(0.7, 0.9) * self.width,
                    "bar_thickness": uniform(0.025, 0.045) * self.height,
                    "bar_aspect_ratio": uniform(0.4, 0.6),
                    "bar_height_ratio": uniform(0.7, 0.9),
                    "bar_length_ratio": uniform(0.5, 0.8),
                    "bar_end_length_ratio": uniform(0.1, 0.15),
                    "bar_end_height_ratio": uniform(1.8, 3.0),
                    "bar_overall_z_offset": -uniform(0.0, 0.1) * self.height,
                    "shader": weighted_sample(material_assignments.metals)(),
                    "color": colors.hsv2rgba(colors.metal_natural_hsv()),
                }
            else:
                self.handle_info_dict = {"handle_type": self.handle_type}
            if self.handle_type in ["knob", "lever"]:
                self.handle_joint = "hinge"
            elif self.handle_type == "bar":
                self.handle_joint = "slide"
            elif self.handle_type == "pull":
                self.handle_joint = "rigid"
            else:
                self.handle_joint = "none"

    BaseDoorFactory.__init__ = patched_init


def patch_kitchen_cabinet():
    """Add kitchen_space_bottom support to kitchen cabinet factories."""
    _ensure_infinigen_on_path()
    from infinigen.assets.objects.shelves.kitchen_cabinet import (
        KitchenCabinetBaseFactory,
        KitchenCabinetFactory,
    )
    from numpy.random import uniform

    _orig_base_init = KitchenCabinetBaseFactory.__init__

    def patched_base_init(
        self,
        factory_seed,
        params=None,
        coarse=False,
        kitchen_space_bottom=False,
    ):
        if params is None:
            params = {}
        _orig_base_init(self, factory_seed, params=params, coarse=coarse)
        self.bottom_mid = kitchen_space_bottom

    KitchenCabinetBaseFactory.__init__ = patched_base_init

    def patched_factory_init(
        self,
        factory_seed,
        params=None,
        coarse=False,
        dimensions=None,
        drawer_only=False,
        kitchen_space_bottom=False,
    ):
        if params is None:
            params = {}
        self.dimensions = dimensions
        KitchenCabinetBaseFactory.__init__(
            self,
            factory_seed,
            params=params,
            coarse=coarse,
            kitchen_space_bottom=kitchen_space_bottom,
        )
        self.drawer_only = drawer_only

    KitchenCabinetFactory.__init__ = patched_factory_init

    _orig_sample_params = KitchenCabinetFactory.sample_params

    def patched_sample_params(self):
        params = dict()
        if self.dimensions is None:
            dimensions = (
                uniform(0.25, 0.35),
                uniform(0.5, 1.0),
                uniform(0.5, 1.3),
            )
            self.dimensions = dimensions
        else:
            dimensions = self.dimensions
        params["Dimensions"] = dimensions
        # Copy frame_params logic from original
        params["shelf_depth"] = params["Dimensions"][0] - 0.01
        num_h = int((params["Dimensions"][2] - 0.06) / 0.3)
        params["shelf_cell_height"] = [
            (params["Dimensions"][2] - 0.06) / num_h for _ in range(num_h)
        ]
        params["side_board_thickness"] = 0.02
        params["division_board_thickness"] = 0.02
        params["bottom_board_height"] = 0.06
        self.frame_params = params
        n_cells = max(int(params["Dimensions"][1] / 0.45), 1)
        intervals = np.random.uniform(0.55, 1.0, size=(n_cells,))
        intervals = intervals / intervals.sum() * params["Dimensions"][1]
        self.cabinet_widths = intervals.tolist()
        if getattr(self, "bottom_mid", False):
            self.cabinet_widths = [params["Dimensions"][1]]

    KitchenCabinetFactory.sample_params = patched_sample_params


def patch_kitchen_space():
    """Customize kitchen space/island creation with sink and layout tweaks."""
    _ensure_infinigen_on_path()
    from infinigen.assets.objects.shelves.kitchen_cabinet import (
        KitchenCabinetFactory,
    )

    # Need to import geometry_nodes_add_cabinet_top and nodegroup_tag_cube from same module
    from infinigen.assets.objects.shelves.kitchen_space import (
        KitchenIslandFactory,
        KitchenSpaceFactory,
        geometry_nodes_add_cabinet_top,
    )
    from infinigen.assets.objects.table_decorations import SinkFactory
    from infinigen.assets.objects.wall_decorations.range_hood import (
        RangeHoodFactory,
    )
    from infinigen.core import surface, tagging
    from infinigen.core.util import blender as butil
    from infinigen.core.util.math import FixedSeed
    from mathutils import Vector
    from numpy.random import choice, uniform

    _orig_ks_init = KitchenSpaceFactory.__init__

    def patched_ks_init(
        self,
        factory_seed,
        coarse=False,
        dimensions=None,
        island=False,
        has_sink=False,
    ):
        KitchenSpaceFactory.__bases__[0].__init__(
            self, factory_seed, coarse=coarse
        )
        with FixedSeed(factory_seed):
            if dimensions is None:
                dimensions = Vector(
                    (uniform(0.7, 1), uniform(1.7, 5), uniform(2.3, 2.5))
                )
            self.island = island
            if self.island:
                dimensions.x *= uniform(1.5, 2)
                dimensions.y = uniform(1, 2)
            self.dimensions = dimensions
            self.params = self.sample_parameters(dimensions)
            self.has_sink = has_sink

    KitchenSpaceFactory.__init__ = patched_ks_init

    _orig_create_asset = KitchenSpaceFactory.create_asset

    def patched_create_asset(self, **params):
        x, y, z = self.dimensions
        parts = []
        cabinet_bottom_height = self.cabinet_bottom_height
        cabinet_top_height = self.cabinet_top_height
        mid_width = uniform(1.0, 1.3)
        other_width = (y - mid_width) / 2.0
        offset_bm = 0.04
        offset_tm = 0.08
        offset = 0.04
        if other_width >= 0.98:
            offset = 0.08
        elif 0.98 > other_width >= 0.9:
            other_width += -0.04
            mid_width += 0.08
        if other_width >= 1.47:
            offset = 0.12
        elif 1.47 > other_width >= 1.35:
            other_width += -0.04
            mid_width += 0.08
        if other_width >= 1.96:
            offset = 0.16
        elif 1.96 > other_width >= 1.8:
            other_width += -0.04
            mid_width += 0.08

        if self.island and other_width <= 0.3:
            num_cells = False
            offset = 0.08
            if getattr(self, "has_sink", False) or y < 1.35:
                num_cells = True
                offset = 0.04
            island_factory = KitchenCabinetFactory(
                self.factory_seed,
                dimensions=(x, y - offset, cabinet_bottom_height),
                drawer_only=True,
                kitchen_space_bottom=num_cells,
            )
            cabinet_bottom = island_factory(i=0)
        else:
            cabinet_bottom_factory = KitchenCabinetFactory(
                self.factory_seed,
                dimensions=(x, other_width - offset, cabinet_bottom_height),
                drawer_only=True,
            )
            cabinet_bottom_left = cabinet_bottom_factory(i=0)
            cabinet_bottom_right = cabinet_bottom_factory(i=1)
            cabinet_bottom_left.location = (0.0, 0.0, 0.0)
            cabinet_bottom_right.location = (0.0, y - other_width, 0.0)
            cabinet_bottom_mid_factory = KitchenCabinetFactory(
                self.factory_seed,
                dimensions=(x, mid_width - offset_bm, cabinet_bottom_height),
                drawer_only=True,
                kitchen_space_bottom=True,
            )
            bottom_mid = cabinet_bottom_mid_factory(i=0)
            bottom_mid.location = (0.0, y - other_width - mid_width, 0.0)
            cabinet_bottom = butil.join_objects(
                [cabinet_bottom_left, cabinet_bottom_right, bottom_mid]
            )
        parts.append(cabinet_bottom)
        surface.add_geomod(
            cabinet_bottom, geometry_nodes_add_cabinet_top, apply=True
        )

        if getattr(self, "has_sink", False):
            sink_factory = SinkFactory(
                factory_seed=self.factory_seed,
                dimensions=[
                    mid_width * 0.7,
                    min(x * 0.7, 0.4),
                    cabinet_bottom_height * 0.3,
                ],
            )
            sink = sink_factory(i=0)
            sink.location = (
                (x / 2.0) - 0.3,
                y / 2.0,
                cabinet_bottom_height * 0.7 + 0.12,
            )
            sink.parent = cabinet_bottom

        if not self.island:
            cabinet_top_factory = KitchenCabinetFactory(
                self.factory_seed,
                dimensions=(x / 2.0, other_width - offset, cabinet_top_height),
                drawer_only=False,
            )
            cabinet_top_left = cabinet_top_factory(i=0)
            cabinet_top_right = cabinet_top_factory(i=1)
            cabinet_top_left.location = (-x / 4.0, 0.0, z - cabinet_top_height)
            cabinet_top_right.location = (
                -x / 4.0,
                y - other_width,
                z - cabinet_top_height,
            )
            mid_style = choice(["cabinet"])
            if mid_style == "range_hood":
                range_hood_factory = RangeHoodFactory(
                    self.factory_seed,
                    dimensions=(
                        x * 0.66,
                        mid_width + 0.15,
                        cabinet_top_height,
                    ),
                )
                top_mid = range_hood_factory(i=0)
                top_mid.location = (
                    -x * 0.5,
                    y / 2.0,
                    z - cabinet_top_height + 0.05,
                )
            elif mid_style == "cabinet":
                cabinet_top_mid_factory = KitchenCabinetFactory(
                    self.factory_seed,
                    dimensions=(
                        x / 2.0,
                        mid_width - offset_tm,
                        cabinet_top_height,
                    ),
                    drawer_only=False,
                )
                top_mid = cabinet_top_mid_factory(i=0)
                top_mid.location = (
                    -x / 4.0,
                    (y / 2.0) - (mid_width / 2.0),
                    z - cabinet_top_height,
                )
            else:
                raise NotImplementedError
            parts += [cabinet_top_left, cabinet_top_right, top_mid]

        kitchen_space = butil.join_objects(parts)
        if not self.island:
            kitchen_space.dimensions = self.dimensions
        butil.apply_transform(kitchen_space)
        tagging.tag_system.relabel_obj(kitchen_space)
        return kitchen_space

    KitchenSpaceFactory.create_asset = patched_create_asset

    def patched_island_init(self, factory_seed):
        KitchenSpaceFactory.__init__(
            self, factory_seed=factory_seed, island=True, has_sink=False
        )

    KitchenIslandFactory.__init__ = patched_island_init


def patch_sink():
    """Simplify SinkFactory.sample_parameters with fixed sampling ranges."""
    _ensure_infinigen_on_path()
    from infinigen.assets.objects.table_decorations.sink import SinkFactory
    from numpy.random import uniform as U

    def patched_sample_parameters(
        dimensions, upper_height, use_default=False, open=False
    ):
        if not dimensions:
            width = U(0.4, 1.0)
            depth = U(0.4, 0.5)
            upper_height = U(0.2, 0.4)
        else:
            width, depth, upper_height = dimensions
        curvature = U(1.0, 1.0)
        lower_height = U(0.00, 0.01)
        hole_radius = U(0.02, 0.05)
        margin = U(0.02, 0.05)
        watertap_margin = U(0.1, 0.12)
        params = {
            "Width": width,
            "Depth": depth,
            "Curvature": curvature,
            "Upper Height": upper_height,
            "Lower Height": lower_height,
            "HoleRadius": hole_radius,
            "Margin": margin,
            "WaterTapMargin": watertap_margin,
            "ProtrudeAboveCounter": U(0.01, 0.025),
        }
        return params

    SinkFactory.sample_parameters = staticmethod(patched_sample_parameters)


def patch_generate_indoors():
    """Force populate_doors to use all_open=True by default."""
    _ensure_infinigen_on_path()
    from infinigen.core.constraints.example_solver.room import (
        decorate as room_dec,
    )

    _orig_populate_doors = room_dec.populate_doors

    def patched_populate_doors(
        placeholders,
        constants,
        n_doors=3,
        door_chance=1,
        casing_chance=0.0,
        all_open=False,
        **kwargs,
    ):
        return _orig_populate_doors(
            placeholders,
            constants,
            n_doors=n_doors,
            door_chance=door_chance,
            casing_chance=casing_chance,
            all_open=True,
            **kwargs,
        )

    room_dec.populate_doors = patched_populate_doors


def patch_room_types():
    """Include Office in util.room_types."""
    _ensure_infinigen_on_path()
    from infinigen.core import tags as t
    from infinigen_examples.constraints import util as cu

    cu.room_types.add(t.Semantics.Office)


def patch_home_constraints():
    """Add office-only room constraints and desk/chair furniture rules."""
    _ensure_infinigen_on_path()
    from collections import OrderedDict

    import gin
    from infinigen.assets.objects import seating, shelves
    from infinigen.core.constraints import constraint_language as cl
    from infinigen.core.constraints.constraint_language.constants import (
        RoomConstants,
    )
    from infinigen.core.tags import Semantics
    from infinigen_examples.constraints import home as home_module
    from infinigen_examples.constraints import util as cu

    gin.enter_interactive_mode()
    _orig_home_room_constraints = home_module.home_room_constraints

    def _office_room_constraints():
        constraints = OrderedDict()
        score_terms = OrderedDict()
        constants = RoomConstants(
            fixed_contour=False, room_type={Semantics.Office}
        )
        rooms = cl.scene()[Semantics.RoomContour]
        constraints["node_gen"] = rooms[Semantics.Root].all(
            lambda r: (
                rooms[Semantics.Office]
                .related_to(r, cl.Traverse())
                .count()
                .in_range(1, 1, mean=1)
            )
        )
        constraints["node"] = (
            rooms[Semantics.Office].count().in_range(1, 1, mean=1)
            * (rooms[Semantics.Entrance].count() >= 0)
            * (rooms[Semantics.StaircaseRoom].count() == 0)
        )
        all_rooms = cl.scene()[Semantics.RoomContour]
        rooms_filtered = all_rooms[-Semantics.Exterior][-Semantics.Staircase]
        score_terms["room"] = (
            rooms_filtered[Semantics.Office]
            .sum(lambda r: (r.area() / 25).log().hinge(0, 0.4).pow(2))
            .minimize(weight=500.0)
        )
        return cl.Problem(
            constraints=constraints,
            score_terms=score_terms,
            constants=constants,
        )

    @gin.configurable(
        "home_room_constraints", module="infinigen_examples.constraints.home"
    )
    def patched_home_room_constraints(
        has_fewer_rooms=False, office_only=False
    ):
        if office_only:
            return _office_room_constraints()
        return _orig_home_room_constraints(has_fewer_rooms=has_fewer_rooms)

    home_module.home_room_constraints = patched_home_room_constraints

    # --- home_furniture_constraints: Office room (1-2 desks, 1-2 chairs each) ---
    _orig_home_furniture_constraints = home_module.home_furniture_constraints

    def patched_home_furniture_constraints():
        problem = _orig_home_furniture_constraints()
        constraints = OrderedDict(problem.constraints)
        score_terms = OrderedDict(problem.score_terms)
        rooms = cl.scene()[{Semantics.Room, -Semantics.Object}]
        obj = cl.scene()[{Semantics.Object, -Semantics.Room}]
        furniture = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)
        wallfurn = furniture.related_to(rooms, cu.against_wall)
        desks = wallfurn[shelves.SimpleDeskFactory]
        deskchair = furniture[seating.OfficeChairFactory].related_to(
            desks, cu.front_to_front
        )
        offices = rooms[Semantics.Office]
        constraints["office_desks"] = offices.all(
            lambda r: desks.related_to(r).count().in_range(1, 2, mean=1.5)
        )
        constraints["office_desk_chairs"] = offices.all(
            lambda r: desks.related_to(r).all(
                lambda t: (
                    deskchair.related_to(r)
                    .related_to(t)
                    .count()
                    .in_range(1, 2, mean=1.5)
                )
            )
        )
        score_terms["office_desks"] = offices.mean(
            lambda r: desks.related_to(r).mean(
                lambda d: (
                    cl.accessibility_cost(d, furniture.related_to(r)).minimize(
                        weight=3
                    )
                    + cl.accessibility_cost(d, r).minimize(weight=3)
                    + deskchair.related_to(r)
                    .distance(rooms, cu.walltags)
                    .maximize(weight=1)
                )
            )
        )
        return cl.Problem(constraints=constraints, score_terms=score_terms)

    home_module.home_furniture_constraints = patched_home_furniture_constraints


def patch_floor_plan_solver():
    """Guard swap_room against layouts without swap targets."""
    _ensure_infinigen_on_path()
    from infinigen.core.constraints.example_solver.room import (
        solver as solver_module,
    )

    _orig_swap_room = solver_module.FloorPlanMoves.swap_room

    def patched_swap_room(self, state, k):
        candidates = [
            r.target_name for r in state[k].relations if r.value.length > 0
        ]
        if not candidates:
            raise NotImplementedError(
                "No valid swap targets (e.g. single-room layout)"
            )
        return _orig_swap_room(self, state, k)

    solver_module.FloorPlanMoves.swap_room = patched_swap_room


def patch_room_graph_root():
    """Allow single-room graphs to select a valid root without StaircaseRoom."""
    _ensure_infinigen_on_path()
    from infinigen.core.constraints.example_solver.room import (
        base as base_module,
    )
    from infinigen.core.tags import Semantics

    @property
    def patched_root(self):
        if self.entrance is None:
            if self[Semantics.StaircaseRoom]:
                return self.names[self[Semantics.StaircaseRoom][0]]
            if self[Semantics.Root]:
                return self.names[self[Semantics.Root][0]]
            for i, n in enumerate(self.names):
                if base_module.room_type(n) != Semantics.Exterior:
                    return self.names[i]
            raise IndexError(
                "Graph has no StaircaseRoom, Root, or interior room for root"
            )
        return self.names[self._entrance]

    base_module.RoomGraph.root = patched_root


def patch_removed_object_tree() -> None:
    """Skip Blender objects whose RNA handles were removed."""
    _ensure_infinigen_on_path()
    from infinigen.core.util import blender as butil

    def is_removed(obj) -> bool:
        try:
            obj.name
        except ReferenceError:
            return True
        return False

    def patched_iter_object_tree(obj):
        if is_removed(obj):
            return

        yield obj

        try:
            children = list(obj.children)
        except ReferenceError:
            return

        for child in children:
            yield from patched_iter_object_tree(child)

    butil.iter_object_tree = patched_iter_object_tree


def _make_run_main_impl():
    def _run_main_impl():
        import argparse
        from pathlib import Path

        import infinigen_examples.generate_indoors as gi
        from infinigen.core import init

        parser = argparse.ArgumentParser()
        parser.add_argument("--output_folder", type=Path)
        parser.add_argument("--input_folder", type=Path, default=None)
        parser.add_argument("-s", "--seed", default=None)
        parser.add_argument(
            "-t",
            "--task",
            nargs="+",
            default=["coarse"],
            choices=[
                "coarse",
                "populate",
                "fine_terrain",
                "ground_truth",
                "render",
                "mesh_save",
                "export",
            ],
        )
        parser.add_argument("-g", "--configs", nargs="+", default=["base"])
        parser.add_argument("-p", "--overrides", nargs="+", default=[])
        parser.add_argument("--task_uniqname", type=str, default=None)
        parser.add_argument("-d", "--debug", type=str, nargs="*", default=None)

        args = init.parse_args_blender(parser)

        import logging

        logging.getLogger("infinigen").setLevel(logging.INFO)
        logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(
            logging.CRITICAL
        )
        if args.debug is not None:
            for name in logging.root.manager.loggerDict:
                if not name.startswith("infinigen"):
                    continue
                if len(args.debug) == 0 or any(
                    name.endswith(x) for x in args.debug
                ):
                    logging.getLogger(name).setLevel(logging.DEBUG)

        gi.main(args)

    return _run_main_impl


def add_run_main_to_module(module):
    """Inject _run_main into generate_indoors module. Call after 'import infinigen_examples.generate_indoors as gi'."""
    module._run_main = _make_run_main_impl()


def patch_generate_indoors_run_main():
    """Legacy: add _run_main if module already in sys.modules (e.g. when patch runs from generate_indoors top)."""
    mod = sys.modules.get("infinigen_examples.generate_indoors")
    if mod is not None:
        add_run_main_to_module(mod)


def monkey_patch_infinigen(
    *,
    material_assignments=True,
    concrete=True,
    room_constants=True,
    room_types=True,
    home_constraints=True,
    doors=True,
    kitchen_cabinet=True,
    kitchen_space=True,
    sink=True,
    generate_indoors=True,
    removed_object_tree=True,
):
    """Apply selected monkey patches to Infinigen."""
    if material_assignments:
        patch_material_assignments()
    if concrete:
        patch_concrete()
    if room_constants:
        patch_room_constants()
    if room_types:
        patch_room_types()
    if home_constraints:
        patch_home_constraints()
    if doors:
        patch_doors_base_simple()
    if kitchen_cabinet:
        patch_kitchen_cabinet()
    if kitchen_space:
        patch_kitchen_space()
    if sink:
        patch_sink()
    if generate_indoors:
        patch_generate_indoors()
    patch_floor_plan_solver()
    patch_room_graph_root()
    if removed_object_tree:
        patch_removed_object_tree()
    patch_generate_indoors_run_main()
