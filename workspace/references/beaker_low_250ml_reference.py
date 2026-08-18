#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate and render a 250 mL low-form borosilicate beaker in Blender 5.2."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import bpy
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_DIR = SCRIPT_DIR.parent / "toolkit"
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

import lab_blender_toolkit as lab

importlib.reload(lab)


CLEAR_SCENE = True
AUTO_RENDER = True
SAVE_BLEND = True
RENDER_ENGINE = "BLENDER_EEVEE"  # Change to "CYCLES" for slower path-traced glass.
RESOLUTION = 768
RADIAL_SEGMENTS = 192

NAME = "Beaker_LowForm_250mL"
VOLUME_ML = 250
HEIGHT_MM = 88.0
OUTPUT_DIR = SCRIPT_DIR / "output" / NAME

OUTER_PROFILE = lab.profile_from_mm(
    [
        (34.0, 0.0, 0.0),
        (35.0, 1.2, 0.0),
        (35.0, 4.0, 0.0),
        (36.0, 78.0, 0.0),
        (36.5, 84.0, 0.35),
        (37.0, HEIGHT_MM, 1.0),
    ]
)
INNER_PROFILE = lab.profile_from_mm(
    [
        (32.5, 4.0, 0.0),
        (32.5, 78.0, 0.0),
        (33.8, 84.0, 0.35),
        (34.5, HEIGHT_MM, 1.0),
    ]
)

SPOUT_DEFORMER = lab.make_angular_deformer(
    center_deg=-130.0,
    half_width_deg=18.0,
    radial_extension=lab.mm(8.0),
    vertical_rise=lab.mm(2.0),
)

GRADUATION_STYLE = lab.GraduationStyle(
    end_angle_deg=-34.0,
    major_length=lab.mm(16.0),
    minor_length=lab.mm(8.0),
    line_radius=lab.mm(0.28),
    surface_clearance=lab.mm(0.01),
    label_tangent_offset=lab.mm(5.5),
    text_size=lab.mm(4.4),
    text_extrude=lab.mm(0.012),
    text_outline=lab.mm(0.006),
    text_surface_clearance=lab.mm(0.008),
    text_curve_resolution=12,
    text_subdivision_cuts=1,
    top_clearance=lab.mm(3.0),
    arc_samples=24,
)

VIEWS = [
    lab.RenderView("front_three_quarter", -51.0, 19.0, 0.278, 62.0),
    lab.RenderView("spout_three_quarter", -129.0, 22.0, 0.290, 62.0),
    lab.RenderView("opposite_high", 34.0, 29.0, 0.305, 64.0),
]


def build_asset() -> bpy.types.Object:
    if CLEAR_SCENE:
        lab.clear_scene()

    lab.configure_scene(
        resolution=RESOLUTION,
        transparent_background=False,
        render_engine=RENDER_ENGINE,
    )

    asset_collection = lab.create_collection("ASSET_Beaker")
    marking_collection = lab.create_collection("ASSET_Graduations")
    studio_collection = lab.create_collection("STUDIO")

    glass = lab.create_borosilicate_glass_material()
    markings = lab.create_marking_material()

    beaker = lab.create_hollow_revolved_mesh(
        name=NAME,
        outer_profile=OUTER_PROFILE,
        inner_profile=INNER_PROFILE,
        collection=asset_collection,
        material=glass,
        radial_segments=RADIAL_SEGMENTS,
        deformer=SPOUT_DEFORMER,
    )
    lab.assign_metadata(
        beaker,
        {
            "instrument_type": "beaker",
            "style": "low_form",
            "nominal_volume_ml": VOLUME_ML,
            "outer_height_mm": HEIGHT_MM,
            "calculated_inner_capacity_ml": lab.profile_capacity_ml(INNER_PROFILE),
            "material_type": "borosilicate_glass",
        },
    )

    lab.add_volume_graduations(
        inner_profile=INNER_PROFILE,
        outer_profile=OUTER_PROFILE,
        nominal_volume_ml=VOLUME_ML,
        minor_interval_ml=10,
        major_interval_ml=50,
        style=GRADUATION_STYLE,
        material=markings,
        collection=marking_collection,
        outer_deformer=SPOUT_DEFORMER,
    )

    lab.create_grid_floor(
        collection=studio_collection,
        size=0.55,
        z=lab.mm(-0.15),
        spacing=lab.mm(10.0),
        minor_width=lab.mm(0.22),
        major_width=lab.mm(0.55),
        major_every=5,
    )

    lab.enable_freestyle_outline(
        thickness_px=3,
        color=(0.0, 0.0, 0.0, 1.0),
        include_open_borders=True,
        include_creases=False,
    )

    camera = lab.create_camera(studio_collection)
    target = Vector((0.0, 0.0, lab.mm(44.0)))
    lab.setup_glass_product_lighting(studio_collection, target)

    camera.location = lab.orbit_location(target, VIEWS[0])
    camera.data.lens = VIEWS[0].lens_mm
    lab.look_at(camera, target)
    lab.select_only(beaker)

    print("=" * 72)
    print(f"Generated: {NAME}")
    print(f"Integrated inner capacity: {lab.profile_capacity_ml(INNER_PROFILE):.2f} mL")
    print(f"Render engine: {RENDER_ENGINE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 72)

    if AUTO_RENDER:
        lab.render_views(
            camera=camera,
            target=target,
            views=VIEWS,
            output_directory=OUTPUT_DIR,
            filename_prefix=NAME,
        )
    if SAVE_BLEND:
        lab.save_blend(OUTPUT_DIR / f"{NAME}.blend")
    return beaker


if __name__ == "__main__":
    build_asset()
