#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blender 5.2 laboratory-asset toolkit.

The module is instrument-agnostic. It provides reusable geometry, volume,
graduation, glass-material, studio-lighting, camera, outline, and rendering
helpers. All low-level geometric values use Blender metres; use ``mm()`` for
readable millimetre specifications.

Recommended generated-script workflow for the coding model:

1. ``clear_scene()`` and ``configure_scene(...)``.
2. Create asset, marking, and studio collections with ``create_collection``.
3. Build outer/inner vertical profiles with ``profile_from_mm``. For rounded
   shoulders, bellies, or neck transitions, call ``smooth_profile`` or the
   convenience wrapper ``smooth_profile_from_mm`` before mesh construction.
4. Build closed vessel walls using ``create_hollow_revolved_mesh`` and verify
   capacity with ``profile_capacity_ml``.
5. Add volumetrically correct markings with ``add_volume_graduations``.
6. Add floor, camera, and lights; optionally call ``enable_freestyle_outline``
   for a diagnostic silhouette overlay.
7. Render multiple views with ``render_views`` and save with ``save_blend``.

Profile smoothing changes the geometric profile and therefore capacity. Always
compute graduations from the same *smoothed inner profile* used to build the
mesh. Keep intentionally sharp rim/base/joint control points in ``sharp_indices``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import bmesh
import bpy
from mathutils import Vector


@dataclass(frozen=True)
class ProfilePoint:
    """One radius-height control point of an axisymmetric vessel profile.

    ``radius`` and ``z`` are in metres. ``deform_weight`` controls how strongly
    an optional angular deformer affects this ring (0 = none, 1 = full effect).
    """

    radius: float
    z: float
    deform_weight: float = 0.0


@dataclass(frozen=True)
class GraduationStyle:
    """Appearance and placement settings for curved ticks and wrapped labels."""

    end_angle_deg: float = -34.0
    major_length: float = 0.016
    minor_length: float = 0.008
    line_radius: float = 0.00028
    surface_clearance: float = 0.00001
    label_tangent_offset: float = 0.0055
    text_size: float = 0.0044
    text_extrude: float = 0.000012
    text_outline: float = 0.000006
    text_surface_clearance: float = 0.000008
    text_curve_resolution: int = 12
    text_subdivision_cuts: int = 1
    top_clearance: float = 0.003
    arc_samples: int = 24


@dataclass(frozen=True)
class RenderView:
    """One orbit-camera diagnostic view around a target point."""

    name: str
    azimuth_deg: float
    elevation_deg: float
    distance: float
    lens_mm: float = 62.0


RingDeformer = Callable[[float, float, float, float], tuple[float, float]]


def mm(value: float) -> float:
    """Convert millimetres to Blender metres."""

    return value / 1000.0


def ml_to_m3(value_ml: float) -> float:
    """Convert millilitres to cubic metres."""

    return value_ml * 1e-6


def m3_to_ml(value_m3: float) -> float:
    """Convert cubic metres to millilitres."""

    return value_m3 * 1e6


def _positive_whole_ml(value: int | float, *, name: str) -> int:
    """Normalize a positive whole-millilitre value for range/modulo logic.

    Instrument specifications are parsed through Pydantic and may therefore
    arrive as integer-valued floats (for example ``250.0``).  Python's
    ``range`` rejects floats even when they represent exact integers, so the
    toolkit normalizes them at its public API boundary instead of requiring
    every generated script to reproduce a fragile cast.
    """

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive whole number of millilitres.") from exc
    if not math.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive whole number of millilitres.")
    return int(numeric)


def profile_from_mm(rows: Iterable[Sequence[float]]) -> list[ProfilePoint]:
    """Create and validate a profile from ``(radius_mm, z_mm[, weight])`` rows.

    Rows must be ordered by strictly increasing height. This function preserves
    the control points exactly; use ``smooth_profile_from_mm`` when a densified,
    smoothly interpolated contour is desired.
    """

    points: list[ProfilePoint] = []
    for row in rows:
        if len(row) == 2:
            radius_mm, z_mm = row
            weight = 0.0
        elif len(row) == 3:
            radius_mm, z_mm, weight = row
        else:
            raise ValueError("Each profile row must contain 2 or 3 values.")
        points.append(ProfilePoint(mm(float(radius_mm)), mm(float(z_mm)), float(weight)))
    validate_profile(points)
    return points


def validate_profile(profile: Sequence[ProfilePoint]) -> None:
    """Raise a clear error unless a profile has positive radii and increasing z."""

    if len(profile) < 2:
        raise ValueError("A profile requires at least two points.")
    previous_z = -math.inf
    for point in profile:
        if point.radius <= 0.0:
            raise ValueError("Profile radii must be positive.")
        if point.z <= previous_z:
            raise ValueError("Profile heights must be strictly increasing.")
        previous_z = point.z


def _pchip_slopes(xs: Sequence[float], ys: Sequence[float]) -> list[float]:
    """Compute shape-preserving cubic Hermite slopes for ordered samples."""

    count = len(xs)
    if count != len(ys):
        raise ValueError("xs and ys must have the same length.")
    if count < 2:
        raise ValueError("At least two samples are required.")
    if count == 2:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return [slope, slope]

    intervals = [xs[index + 1] - xs[index] for index in range(count - 1)]
    secants = [
        (ys[index + 1] - ys[index]) / intervals[index]
        for index in range(count - 1)
    ]
    slopes = [0.0] * count

    for index in range(1, count - 1):
        previous = secants[index - 1]
        following = secants[index]
        if previous == 0.0 or following == 0.0 or previous * following < 0.0:
            slopes[index] = 0.0
            continue
        left_weight = 2.0 * intervals[index] + intervals[index - 1]
        right_weight = intervals[index] + 2.0 * intervals[index - 1]
        slopes[index] = (left_weight + right_weight) / (
            left_weight / previous + right_weight / following
        )

    def endpoint_slope(
        first_interval: float,
        second_interval: float,
        first_secant: float,
        second_secant: float,
    ) -> float:
        slope = (
            (2.0 * first_interval + second_interval) * first_secant
            - first_interval * second_secant
        ) / (first_interval + second_interval)
        if slope * first_secant <= 0.0:
            return 0.0
        if first_secant * second_secant < 0.0 and abs(slope) > 3.0 * abs(first_secant):
            return 3.0 * first_secant
        return slope

    slopes[0] = endpoint_slope(
        intervals[0], intervals[1], secants[0], secants[1]
    )
    slopes[-1] = endpoint_slope(
        intervals[-1], intervals[-2], secants[-1], secants[-2]
    )
    return slopes


def _cubic_hermite(
    y0: float,
    y1: float,
    slope0: float,
    slope1: float,
    interval: float,
    t: float,
) -> float:
    """Evaluate one cubic Hermite segment at normalized position ``t``."""

    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00 * y0
        + h10 * interval * slope0
        + h01 * y1
        + h11 * interval * slope1
    )


def smooth_profile(
    profile: Sequence[ProfilePoint],
    samples_per_segment: int = 8,
    sharp_indices: Iterable[int] = (),
) -> list[ProfilePoint]:
    """Densify a profile with shape-preserving smooth interpolation.

    Radius and deformation weight are interpolated as functions of height using
    monotone piecewise cubic Hermite interpolation (PCHIP). It avoids the large
    overshoots of unconstrained Catmull-Rom splines and is therefore suitable for
    vessel shoulders, bellies, and neck transitions.

    ``samples_per_segment`` is the number of subsegments generated between each
    pair of controls. ``sharp_indices`` contains control-point indices whose
    adjacent segments must remain linear, preserving intentional corners such as
    a flat base transition, rim, flange, or ground-glass joint.

    The returned profile includes every original control point and is ready for
    ``create_hollow_revolved_mesh`` and all volume/graduation helpers. Always use
    the same smoothed inner profile for capacity and graduation calculations.
    """

    validate_profile(profile)
    if isinstance(samples_per_segment, bool) or samples_per_segment < 1:
        raise ValueError("samples_per_segment must be a positive integer.")
    samples_per_segment = int(samples_per_segment)

    sharp = {int(index) for index in sharp_indices}
    invalid = sorted(index for index in sharp if index < 0 or index >= len(profile))
    if invalid:
        raise ValueError(f"sharp_indices contains invalid indices: {invalid}")

    heights = [point.z for point in profile]
    radii = [point.radius for point in profile]
    weights = [point.deform_weight for point in profile]
    radius_slopes = _pchip_slopes(heights, radii)
    weight_slopes = _pchip_slopes(heights, weights)

    result: list[ProfilePoint] = []
    for segment_index, (lower, upper) in enumerate(zip(profile[:-1], profile[1:])):
        interval = upper.z - lower.z
        keep_linear = segment_index in sharp or (segment_index + 1) in sharp
        for sample_index in range(samples_per_segment):
            t = sample_index / samples_per_segment
            z = lower.z + interval * t
            if keep_linear:
                radius = lower.radius + (upper.radius - lower.radius) * t
                weight = lower.deform_weight + (
                    upper.deform_weight - lower.deform_weight
                ) * t
            else:
                radius = _cubic_hermite(
                    lower.radius,
                    upper.radius,
                    radius_slopes[segment_index],
                    radius_slopes[segment_index + 1],
                    interval,
                    t,
                )
                weight = _cubic_hermite(
                    lower.deform_weight,
                    upper.deform_weight,
                    weight_slopes[segment_index],
                    weight_slopes[segment_index + 1],
                    interval,
                    t,
                )
            result.append(ProfilePoint(radius=max(radius, 1e-12), z=z, deform_weight=weight))

    result.append(profile[-1])
    validate_profile(result)
    return result


def smooth_profile_from_mm(
    rows: Iterable[Sequence[float]],
    samples_per_segment: int = 8,
    sharp_indices: Iterable[int] = (),
) -> list[ProfilePoint]:
    """Create a millimetre profile and immediately smooth/densify it.

    This convenience wrapper is equivalent to
    ``smooth_profile(profile_from_mm(rows), ...)``.
    """

    return smooth_profile(
        profile_from_mm(rows),
        samples_per_segment=samples_per_segment,
        sharp_indices=sharp_indices,
    )


def clear_scene() -> None:
    """Delete scene objects/collections and purge unused core data blocks.

    Call this once at the beginning of a generated script to avoid inheriting
    objects from the startup file. The function safely leaves Edit Mode first.
    """

    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    for data_blocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for data_block in list(data_blocks):
            if data_block.users == 0:
                data_blocks.remove(data_block)


def create_collection(name: str) -> bpy.types.Collection:
    """Create a new collection linked directly under the active scene."""

    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def select_only(obj: bpy.types.Object) -> None:
    """Make exactly one object selected and active for deterministic operators."""

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def assign_metadata(obj: bpy.types.Object, metadata: Mapping[str, object]) -> None:
    """Store simple custom properties on an asset object for downstream indexing."""

    for key, value in metadata.items():
        obj[key] = value


def sample_profile(profile: Sequence[ProfilePoint], z: float) -> ProfilePoint:
    """Linearly sample radius and deformation weight at height ``z``.

    Heights outside the profile are clamped to the nearest endpoint. A smoothed
    profile should already be densified before calling this helper.
    """

    validate_profile(profile)
    if z <= profile[0].z:
        return profile[0]
    if z >= profile[-1].z:
        return profile[-1]

    for lower, upper in zip(profile[:-1], profile[1:]):
        if lower.z <= z <= upper.z:
            t = (z - lower.z) / (upper.z - lower.z)
            return ProfilePoint(
                radius=lower.radius + (upper.radius - lower.radius) * t,
                z=z,
                deform_weight=(
                    lower.deform_weight
                    + (upper.deform_weight - lower.deform_weight) * t
                ),
            )
    raise RuntimeError("Profile interpolation failed.")


def frustum_segment_volume_m3(lower: ProfilePoint, upper: ProfilePoint) -> float:
    """Return the exact volume of one linear-radius frustum segment."""

    height = upper.z - lower.z
    return (
        math.pi
        * height
        * (
            lower.radius * lower.radius
            + lower.radius * upper.radius
            + upper.radius * upper.radius
        )
        / 3.0
    )


def _partial_segment_volume_m3(
    lower: ProfilePoint,
    upper: ProfilePoint,
    local_height: float,
) -> float:
    """Return volume from a segment's lower point up to ``local_height``."""

    segment_height = upper.z - lower.z
    x = min(max(local_height, 0.0), segment_height)
    slope = (upper.radius - lower.radius) / segment_height
    return math.pi * (
        lower.radius * lower.radius * x
        + lower.radius * slope * x * x
        + slope * slope * x * x * x / 3.0
    )


def profile_capacity_m3(profile: Sequence[ProfilePoint]) -> float:
    """Integrate the full piecewise-frustum capacity in cubic metres."""

    validate_profile(profile)
    return sum(
        frustum_segment_volume_m3(lower, upper)
        for lower, upper in zip(profile[:-1], profile[1:])
    )


def profile_capacity_ml(profile: Sequence[ProfilePoint]) -> float:
    """Integrate the full piecewise-frustum capacity in millilitres."""

    return m3_to_ml(profile_capacity_m3(profile))


def volume_below_height_m3(profile: Sequence[ProfilePoint], z: float) -> float:
    """Integrate profile volume from the bottom through world height ``z``."""

    validate_profile(profile)
    if z <= profile[0].z:
        return 0.0

    volume = 0.0
    for lower, upper in zip(profile[:-1], profile[1:]):
        if z >= upper.z:
            volume += frustum_segment_volume_m3(lower, upper)
            continue
        if z > lower.z:
            volume += _partial_segment_volume_m3(lower, upper, z - lower.z)
        break
    return volume


def height_for_volume_ml(
    profile: Sequence[ProfilePoint],
    target_volume_ml: float,
    iterations: int = 64,
) -> float:
    """Invert a piecewise-linear radius profile to find a volume height.

    Each adjacent profile pair forms an exact frustum segment. The function
    first locates the segment containing the target cumulative volume, then
    solves the cubic partial-volume relation by stable bisection. This avoids
    assuming a constant-radius cylinder and remains valid for tapered vessels.
    """

    if target_volume_ml < 0.0:
        raise ValueError("Target volume cannot be negative.")

    target = ml_to_m3(target_volume_ml)
    capacity = profile_capacity_m3(profile)
    if target > capacity + 1e-12:
        raise ValueError(
            f"Requested {target_volume_ml:g} mL exceeds profile capacity "
            f"{m3_to_ml(capacity):.3f} mL."
        )
    if target == 0.0:
        return profile[0].z

    accumulated = 0.0
    for lower, upper in zip(profile[:-1], profile[1:]):
        segment_volume = frustum_segment_volume_m3(lower, upper)
        if accumulated + segment_volume < target:
            accumulated += segment_volume
            continue

        local_target = target - accumulated
        low = 0.0
        high = upper.z - lower.z
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            partial = _partial_segment_volume_m3(lower, upper, middle)
            if partial < local_target:
                low = middle
            else:
                high = middle
        return lower.z + 0.5 * (low + high)

    return profile[-1].z


def graduation_levels(
    inner_profile: Sequence[ProfilePoint],
    maximum_volume_ml: int | float,
    interval_ml: int | float,
    top_clearance: float = 0.0,
) -> list[tuple[int, float]]:
    """Return ``(millilitres, z)`` pairs for equal-volume graduation marks.

    Values are found by inverting integrated inner-profile volume, so tapered
    vessels correctly produce non-uniform vertical tick spacing.
    """

    maximum_volume_ml = _positive_whole_ml(
        maximum_volume_ml,
        name="maximum_volume_ml",
    )
    interval_ml = _positive_whole_ml(interval_ml, name="interval_ml")

    maximum_z = inner_profile[-1].z - top_clearance
    levels: list[tuple[int, float]] = []
    # ``maximum_volume_ml + 1`` includes an exact final mark while never
    # generating a mark above the requested maximum when it is not divisible
    # by the interval.
    for value_ml in range(interval_ml, maximum_volume_ml + 1, interval_ml):
        z = height_for_volume_ml(inner_profile, value_ml)
        if z <= maximum_z:
            levels.append((value_ml, z))
    return levels


def angular_difference(angle: float, center: float) -> float:
    """Return the wrapped signed angular difference in ``[-pi, pi]``."""

    return math.atan2(math.sin(angle - center), math.cos(angle - center))


def cosine_lobe(theta: float, center: float, half_width: float) -> float:
    """Return a smooth 0..1 angular influence lobe around ``center``."""

    distance = abs(angular_difference(theta, center))
    if distance >= half_width:
        return 0.0
    normalized = distance / half_width
    return math.cos(normalized * math.pi / 2.0) ** 2


def make_angular_deformer(
    center_deg: float,
    half_width_deg: float,
    radial_extension: float,
    vertical_rise: float = 0.0,
) -> RingDeformer:
    """Create a smooth local ring deformer, typically for a pouring spout.

    The profile point's ``deform_weight`` scales radial extension and optional
    vertical rise, allowing deformation to grow only near the vessel rim.
    """

    center = math.radians(center_deg)
    half_width = math.radians(half_width_deg)

    def deform(theta: float, radius: float, z: float, weight: float) -> tuple[float, float]:
        shape = cosine_lobe(theta, center, half_width) * weight
        return radius + radial_extension * shape, z + vertical_rise * shape

    return deform


def identity_deformer(
    theta: float,
    radius: float,
    z: float,
    weight: float,
) -> tuple[float, float]:
    """Return unchanged radius/height; default when no angular deformation exists."""

    del theta, weight
    return radius, z


def surface_point_at(
    outer_profile: Sequence[ProfilePoint],
    z: float,
    theta: float,
    deformer: RingDeformer | None = None,
) -> tuple[float, float]:
    """Return deformed surface ``(radius, z)`` at a height and polar angle."""

    sample = sample_profile(outer_profile, z)
    transform = deformer or identity_deformer
    return transform(theta, sample.radius, sample.z, sample.deform_weight)


def surface_radius_at(
    outer_profile: Sequence[ProfilePoint],
    z: float,
    theta: float,
    deformer: RingDeformer | None = None,
) -> float:
    """Return only the deformed surface radius at a height and polar angle."""

    radius, _ = surface_point_at(outer_profile, z, theta, deformer)
    return radius


def _add_profile_ring(
    vertices: list[tuple[float, float, float]],
    point: ProfilePoint,
    radial_segments: int,
    deformer: RingDeformer,
) -> list[int]:
    """Append one deformed circular vertex ring and return its vertex indices."""

    indices: list[int] = []
    for index in range(radial_segments):
        theta = 2.0 * math.pi * index / radial_segments
        radius, z = deformer(theta, point.radius, point.z, point.deform_weight)
        indices.append(len(vertices))
        vertices.append((radius * math.cos(theta), radius * math.sin(theta), z))
    return indices


def _connect_rings(
    faces: list[tuple[int, ...]],
    lower: Sequence[int],
    upper: Sequence[int],
    inward: bool,
) -> None:
    """Append quad faces between equal-length rings with requested winding."""

    count = len(lower)
    for index in range(count):
        nxt = (index + 1) % count
        if inward:
            faces.append((lower[index], upper[index], upper[nxt], lower[nxt]))
        else:
            faces.append((lower[index], lower[nxt], upper[nxt], upper[index]))


def create_hollow_revolved_mesh(
    name: str,
    outer_profile: Sequence[ProfilePoint],
    inner_profile: Sequence[ProfilePoint],
    collection: bpy.types.Collection,
    material: bpy.types.Material | None = None,
    radial_segments: int = 192,
    deformer: RingDeformer | None = None,
) -> bpy.types.Object:
    """Build a closed hollow vessel directly from two vertical profiles.

    Separate outer and inner profile rings create real wall thickness. Their
    winding is reversed so outer normals face the room and inner normals face
    the cavity. The top rings form a solid rim, while two triangle fans close
    the external base and internal floor. A deformer can alter each ring point
    by angle and profile weight without changing the generic mesh algorithm.
    """

    validate_profile(outer_profile)
    validate_profile(inner_profile)
    if radial_segments < 12:
        raise ValueError("radial_segments must be at least 12.")

    transform = deformer or identity_deformer
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    outer_rings = [
        _add_profile_ring(vertices, point, radial_segments, transform)
        for point in outer_profile
    ]
    inner_rings = [
        _add_profile_ring(vertices, point, radial_segments, transform)
        for point in inner_profile
    ]

    for lower, upper in zip(outer_rings[:-1], outer_rings[1:]):
        _connect_rings(faces, lower, upper, inward=False)
    for lower, upper in zip(inner_rings[:-1], inner_rings[1:]):
        _connect_rings(faces, lower, upper, inward=True)

    outer_top = outer_rings[-1]
    inner_top = inner_rings[-1]
    for index in range(radial_segments):
        nxt = (index + 1) % radial_segments
        faces.append((outer_top[index], outer_top[nxt], inner_top[nxt], inner_top[index]))

    outer_bottom_center = len(vertices)
    vertices.append((0.0, 0.0, outer_profile[0].z))
    outer_bottom = outer_rings[0]
    for index in range(radial_segments):
        nxt = (index + 1) % radial_segments
        faces.append((outer_bottom_center, outer_bottom[nxt], outer_bottom[index]))

    inner_floor_center = len(vertices)
    vertices.append((0.0, 0.0, inner_profile[0].z))
    inner_bottom = inner_rings[0]
    for index in range(radial_segments):
        nxt = (index + 1) % radial_segments
        faces.append((inner_floor_center, inner_bottom[index], inner_bottom[nxt]))

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=False)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    if material is not None:
        mesh.materials.append(material)
    return obj


def _new_principled_material(
    name: str,
    base_color: tuple[float, float, float, float],
    roughness: float,
) -> bpy.types.Material:
    """Create a minimal Principled BSDF material with color and roughness."""

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Roughness"].default_value = roughness

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (320, 0)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _new_emission_material(
    name: str,
    color: tuple[float, float, float, float],
    strength: float = 1.0,
) -> bpy.types.Material:
    """Create a camera-visible, shadeless material for reference graphics."""

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (0, 0)
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (320, 0)
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def create_borosilicate_glass_material(
    name: str = "MAT_Borosilicate_Glass",
    tint: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    roughness: float = 0.014,
    ior: float = 1.47,
    absorption_density: float = 0.030,
) -> bpy.types.Material:
    """Create clear borosilicate glass suited to restrained studio rendering.

    The material stays physically glass-like through transmission and IOR, while
    a very light volume absorption gives thick areas such as the base and rim a
    readable body. Roughness remains low so the vessel keeps crisp strip-light
    highlights without sliding into a mirror-like chrome appearance.
    """

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (0, 20)
    principled.inputs["Base Color"].default_value = tint
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["IOR"].default_value = ior
    principled.inputs["Transmission Weight"].default_value = 1.0
    principled.inputs["Alpha"].default_value = 1.0

    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    absorption.location = (0, -180)
    absorption.inputs["Color"].default_value = (0.97, 0.985, 1.0, 1.0)
    absorption.inputs["Density"].default_value = absorption_density

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (320, 0)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    links.new(absorption.outputs["Volume"], output.inputs["Volume"])
    return material


def create_marking_material(name: str = "MAT_Graduation_White") -> bpy.types.Material:
    """Create the slightly rough white material used for ticks and labels."""

    return _new_principled_material(name, (0.92, 0.94, 0.97, 1.0), 0.28)


def create_floor_material(name: str = "MAT_Studio_Floor") -> bpy.types.Material:
    """Create a dark, rough studio-floor material that reveals glass silhouettes."""

    # A rough mid-dark floor avoids a white specular wash while preserving
    # enough background contrast to reveal the silhouette of clear glass.
    return _new_principled_material(name, (0.20, 0.205, 0.215, 1.0), 0.93)


def create_minor_grid_material(name: str = "MAT_Studio_Grid_Minor") -> bpy.types.Material:
    """Create a shadeless material for thin reference-grid lines."""

    # Emission strength 1 makes the grid camera-visible and independent of lamps.
    return _new_emission_material(name, (0.050, 0.058, 0.072, 1.0), 0.82)


def create_major_grid_material(name: str = "MAT_Studio_Grid_Major") -> bpy.types.Material:
    """Create a darker shadeless material for major reference-grid lines."""

    return _new_emission_material(name, (0.016, 0.020, 0.030, 1.0), 0.92)


def create_curve_polyline(
    name: str,
    points: Sequence[Vector],
    bevel_radius: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Create a bevelled 3D polyline curve through explicit world-space points."""

    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.fill_mode = "FULL"
    curve_data.resolution_u = 2
    curve_data.bevel_depth = bevel_radius
    curve_data.bevel_resolution = 3

    spline = curve_data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, coordinate in zip(spline.points, points):
        point.co = (*coordinate, 1.0)

    obj = bpy.data.objects.new(name, curve_data)
    collection.objects.link(obj)
    curve_data.materials.append(material)
    return obj


def create_surface_arc_tick(
    name: str,
    z: float,
    physical_length: float,
    outer_profile: Sequence[ProfilePoint],
    style: GraduationStyle,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    outer_deformer: RingDeformer | None = None,
) -> bpy.types.Object:
    """Create a tick that follows the actual vessel surface along its full arc."""

    end_angle = math.radians(style.end_angle_deg)
    reference_radius = surface_radius_at(outer_profile, z, end_angle, outer_deformer)
    centreline_reference = reference_radius + style.line_radius + style.surface_clearance
    angular_length = physical_length / centreline_reference
    start_angle = end_angle - angular_length

    points: list[Vector] = []
    for index in range(style.arc_samples):
        t = index / (style.arc_samples - 1)
        theta = start_angle + (end_angle - start_angle) * t
        wall_radius, surface_z = surface_point_at(
            outer_profile, z, theta, outer_deformer
        )
        centreline_radius = wall_radius + style.line_radius + style.surface_clearance
        points.append(
            Vector(
                (
                    centreline_radius * math.cos(theta),
                    centreline_radius * math.sin(theta),
                    surface_z,
                )
            )
        )

    return create_curve_polyline(
        name=name,
        points=points,
        bevel_radius=style.line_radius,
        material=material,
        collection=collection,
    )


def _font_curve_to_mesh(
    name: str,
    text: str,
    style: GraduationStyle,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Create text as a curve, evaluate it, and return a standalone mesh object."""

    text_data = bpy.data.curves.new(name=f"{name}_Font", type="FONT")
    text_data.body = text
    text_data.align_x = "LEFT"
    text_data.align_y = "CENTER"
    text_data.size = style.text_size
    text_data.extrude = style.text_extrude
    text_data.offset = style.text_outline
    text_data.resolution_u = style.text_curve_resolution

    source = bpy.data.objects.new(f"{name}_Source", text_data)
    collection.objects.link(source)
    bpy.context.view_layer.update()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = source.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    result = bpy.data.objects.new(name, mesh)
    collection.objects.link(result)

    bpy.data.objects.remove(source, do_unlink=True)
    if text_data.users == 0:
        bpy.data.curves.remove(text_data)
    return result


def _subdivide_label_mesh(mesh: bpy.types.Mesh, cuts: int) -> None:
    """Triangulate and subdivide text so it can wrap smoothly onto a surface."""

    if cuts <= 0:
        return

    bm = bmesh.new()
    bm.from_mesh(mesh)
    if bm.faces:
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
    if bm.edges:
        bmesh.ops.subdivide_edges(
            bm,
            edges=list(bm.edges),
            cuts=cuts,
            use_grid_fill=True,
        )
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def create_conformal_surface_label(
    name: str,
    text: str,
    z: float,
    outer_profile: Sequence[ProfilePoint],
    style: GraduationStyle,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    outer_deformer: RingDeformer | None = None,
) -> bpy.types.Object:
    """Convert text to a mesh and analytically wrap every vertex onto the vessel.

    A rotated flat text object only has the correct facing direction; its glyphs
    remain planar. Here local X is interpreted as arc length, local Y as vertical
    displacement, and local Z as ink thickness. Each vertex is remapped to the
    true outer profile at its own height and angle. This follows tapered walls
    and local angular deformation without depending on Shrinkwrap context or a
    fixed-radius cylindrical Bend modifier.
    """

    obj = _font_curve_to_mesh(name, text, style, collection)
    _subdivide_label_mesh(obj.data, style.text_subdivision_cuts)

    base_theta = math.radians(style.end_angle_deg)
    reference_radius = surface_radius_at(
        outer_profile, z, base_theta, outer_deformer
    )

    minimum_depth = min(vertex.co.z for vertex in obj.data.vertices)

    for vertex in obj.data.vertices:
        local = vertex.co.copy()
        tangent_distance = style.label_tangent_offset + local.x
        theta = base_theta + tangent_distance / reference_radius
        sample_z = z + local.y
        wall_radius, surface_z = surface_point_at(
            outer_profile, sample_z, theta, outer_deformer
        )
        # Normalize the font's extrusion depth so its nearest face, rather
        # than its object origin, receives the requested surface clearance.
        ink_depth = local.z - minimum_depth
        radius = wall_radius + style.text_surface_clearance + ink_depth
        vertex.co = (
            radius * math.cos(theta),
            radius * math.sin(theta),
            surface_z,
        )

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()

    obj.data.validate(verbose=False)
    obj.data.update()
    obj.data.materials.append(material)
    return obj


def add_volume_graduations(
    inner_profile: Sequence[ProfilePoint],
    outer_profile: Sequence[ProfilePoint],
    nominal_volume_ml: int | float,
    minor_interval_ml: int | float,
    major_interval_ml: int | float,
    style: GraduationStyle,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    outer_deformer: RingDeformer | None = None,
) -> list[bpy.types.Object]:
    """Add calculated curved ticks and conformal labels to a vessel surface."""

    nominal_volume_ml = _positive_whole_ml(
        nominal_volume_ml,
        name="nominal_volume_ml",
    )
    minor_interval_ml = _positive_whole_ml(
        minor_interval_ml,
        name="minor_interval_ml",
    )
    major_interval_ml = _positive_whole_ml(
        major_interval_ml,
        name="major_interval_ml",
    )

    if major_interval_ml % minor_interval_ml != 0:
        raise ValueError("major_interval_ml must be divisible by minor_interval_ml.")

    created: list[bpy.types.Object] = []
    levels = graduation_levels(
        inner_profile,
        maximum_volume_ml=nominal_volume_ml,
        interval_ml=minor_interval_ml,
        top_clearance=style.top_clearance,
    )

    for value_ml, z in levels:
        is_major = value_ml % major_interval_ml == 0
        created.append(
            create_surface_arc_tick(
                name=f"Tick_{value_ml:03d}mL",
                z=z,
                physical_length=style.major_length if is_major else style.minor_length,
                outer_profile=outer_profile,
                style=style,
                material=material,
                collection=collection,
                outer_deformer=outer_deformer,
            )
        )
        if is_major:
            created.append(
                create_conformal_surface_label(
                    name=f"Label_{value_ml}mL",
                    text=str(value_ml),
                    z=z,
                    outer_profile=outer_profile,
                    style=style,
                    material=material,
                    collection=collection,
                    outer_deformer=outer_deformer,
                )
            )
    return created


def _create_plane_object(
    name: str,
    size: float,
    z: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Create a square horizontal plane with one assigned material."""

    half = size / 2.0
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(
        [(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(material)
    return obj


def create_grid_floor(
    collection: bpy.types.Collection,
    size: float = 0.55,
    z: float = -0.00015,
    spacing: float = 0.010,
    minor_width: float = 0.00038,
    major_width: float = 0.00090,
    major_every: int = 5,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Create a real geometric grid instead of a lighting-sensitive texture.

    The base plane and the grid are separate meshes. Grid lines sit a tiny
    distance above the base and use dark materials, so the pattern remains
    visible under strong lighting and gives transparent glass a reliable
    refraction/background cue. Every fifth line is wider and darker.
    """

    if spacing <= 0.0 or major_every <= 0:
        raise ValueError("Grid spacing and major_every must be positive.")

    floor = _create_plane_object(
        "Studio_Floor",
        size,
        z,
        create_floor_material(),
        collection,
    )

    half = size / 2.0
    # Keep the strips safely above the floor to avoid raster depth fighting.
    line_z = z + mm(0.060)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    material_indices: list[int] = []

    def add_strip(x0: float, y0: float, x1: float, y1: float, width: float, index: int) -> None:
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        nx = -dy / length * width / 2.0
        ny = dx / length * width / 2.0
        start = len(vertices)
        vertices.extend(
            [
                (x0 + nx, y0 + ny, line_z),
                (x1 + nx, y1 + ny, line_z),
                (x1 - nx, y1 - ny, line_z),
                (x0 - nx, y0 - ny, line_z),
            ]
        )
        faces.append((start, start + 1, start + 2, start + 3))
        material_indices.append(index)

    max_index = int(math.floor(half / spacing))
    for grid_index in range(-max_index, max_index + 1):
        coordinate = grid_index * spacing
        is_major = grid_index % major_every == 0
        width = major_width if is_major else minor_width
        material_index = 1 if is_major else 0
        add_strip(coordinate, -half, coordinate, half, width, material_index)
        add_strip(-half, coordinate, half, coordinate, width, material_index)

    mesh = bpy.data.meshes.new("Studio_Grid_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(create_minor_grid_material())
    mesh.materials.append(create_major_grid_material())
    for polygon, material_index in zip(mesh.polygons, material_indices):
        polygon.material_index = material_index
    mesh.update(calc_edges=True)

    grid = bpy.data.objects.new("Studio_Grid", mesh)
    collection.objects.link(grid)
    return floor, grid


def _enable_eevee_raytracing(scene: bpy.types.Scene) -> None:
    """Enable Blender 5.2 Eevee ray tracing when the active build exposes it."""

    eevee = scene.eevee
    if hasattr(eevee, "use_raytracing"):
        eevee.use_raytracing = True

    if hasattr(eevee, "ray_tracing_method"):
        enum_items = eevee.bl_rna.properties["ray_tracing_method"].enum_items
        identifiers = {item.identifier for item in enum_items}
        if "SCREEN" in identifiers:
            eevee.ray_tracing_method = "SCREEN"

    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 128


def configure_scene(
    resolution: int = 768,
    transparent_background: bool = False,
    render_engine: str = "BLENDER_EEVEE",
) -> bpy.types.Scene:
    """Configure a restrained product-render scene for Blender 5.2.

    Eevee and Cycles need different exposure and world-energy balances. Cycles
    tends to accumulate much more diffuse and refracted light, so it gets a
    dimmer background, a lower exposure, and a slightly softer contrast look.
    """

    if bpy.app.version[:2] != (5, 2):
        raise RuntimeError(
            f"This toolkit targets Blender 5.2 only; current version is "
            f"{bpy.app.version_string}."
        )
    if render_engine not in {"BLENDER_EEVEE", "CYCLES"}:
        raise ValueError("render_engine must be BLENDER_EEVEE or CYCLES.")

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene.unit_settings.scale_length = 1.0

    scene.render.engine = render_engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.render.film_transparent = transparent_background

    if render_engine == "BLENDER_EEVEE":
        _enable_eevee_raytracing(scene)
        world_color = (0.72, 0.75, 0.80, 1.0)
        world_strength = 0.20
        exposure = -0.35
        look_name = "AgX - Medium High Contrast"
    else:
        scene.cycles.samples = 192
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, "caustics_reflective"):
            scene.cycles.caustics_reflective = False
        if hasattr(scene.cycles, "caustics_refractive"):
            scene.cycles.caustics_refractive = False
        world_color = (0.60, 0.62, 0.66, 1.0)
        world_strength = 0.085
        exposure = -0.85
        look_name = "AgX - Medium Contrast"

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    background = nodes.new("ShaderNodeBackground")
    background.location = (0, 0)
    background.inputs["Color"].default_value = world_color
    background.inputs["Strength"].default_value = world_strength

    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (320, 0)
    links.new(background.outputs["Background"], output.inputs["Surface"])

    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = exposure
    scene.view_settings.gamma = 1.0
    try:
        scene.view_settings.look = look_name
    except (TypeError, ValueError):
        pass
    return scene


def enable_freestyle_outline(
    thickness_px: float = 2.0,
    color: tuple[float, float, float, float] = (0.02, 0.02, 0.02, 1.0),
    include_open_borders: bool = True,
    include_creases: bool = False,
) -> bpy.types.FreestyleLineSet:
    """Enable visible Freestyle outlines on the active scene.

    This function uses Freestyle's parameter-editor mode rather than Python
    style modules. It draws silhouettes and external contours directly over
    the Combined render result.

    Parameters
    ----------
    thickness_px:
        Final line thickness in pixels.
    color:
        RGBA line color. Values must be in the range [0, 1].
    include_open_borders:
        Draw boundaries belonging to open meshes.
    include_creases:
        Draw sufficiently sharp mesh creases. Normally disabled for glass
        vessels because it can produce noisy internal lines.

    Returns
    -------
    bpy.types.FreestyleLineSet
        The configured Freestyle line set.
    """

    if thickness_px <= 0:
        raise ValueError("thickness_px must be positive.")

    if len(color) != 4:
        raise ValueError("color must contain RGBA values.")

    scene = bpy.context.scene

    # Master Freestyle switch.
    scene.render.use_freestyle = True

    # Use the line-style thickness directly instead of multiplying it by
    # another global thickness value.
    if hasattr(scene.render, "line_thickness_mode"):
        scene.render.line_thickness_mode = "ABSOLUTE"
    if hasattr(scene.render, "line_thickness"):
        scene.render.line_thickness = 1.0

    # Configure every enabled view layer. This is safer than using only
    # bpy.context.view_layer in background/headless rendering.
    configured_line_set = None

    for view_layer in scene.view_layers:
        if not view_layer.use:
            continue

        freestyle = view_layer.freestyle_settings

        # Critical: Line Sets are evaluated in EDITOR mode.
        # The API default may be SCRIPT, which expects Python style modules.
        freestyle.mode = "EDITOR"

        # Critical: keep lines overlaid on the normal Combined result.
        # Otherwise render_views() saves the Combined PNG without the lines.
        if hasattr(freestyle, "use_freestyle_as_render_pass"):
            freestyle.use_freestyle_as_render_pass = False

        # Remove Blender's default or previously generated line sets so that
        # stale selection settings cannot interfere with this configuration.
        while len(freestyle.linesets) > 0:
            freestyle.linesets.remove(freestyle.linesets[0])

        line_set = freestyle.linesets.new("Geometry_Outline")

        # Logical relationship between enabled edge types.
        line_set.edge_type_combination = "OR"

        # Visible camera-facing silhouette edges.
        line_set.select_silhouette = True

        # Outer contours are important for closed, smooth vessels.
        # Depending on the geometry and view, relying on silhouette alone can
        # miss part of the apparent outer boundary.
        line_set.select_external_contour = True

        # Borders mainly matter for open surfaces.
        line_set.select_border = include_open_borders

        # Creases are usually undesirable on dense glass meshes.
        line_set.select_crease = include_creases

        # Explicitly disable unrelated edge categories.
        line_set.select_edge_mark = False
        line_set.select_material_boundary = False
        line_set.select_ridge_valley = False
        line_set.select_suggestive_contour = False

        # Only draw edges that are visible from the camera.
        line_set.visibility = "VISIBLE"

        line_style = line_set.linestyle
        line_style.color = color[:3]
        line_style.alpha = color[3]
        line_style.thickness = float(thickness_px)

        configured_line_set = line_set

    if configured_line_set is None:
        raise RuntimeError("No enabled View Layer was available for Freestyle.")

    print("[Freestyle] enabled")
    print(f"[Freestyle] engine={scene.render.engine}")
    print(f"[Freestyle] thickness={thickness_px}px")
    print(f"[Freestyle] view_layers={len(scene.view_layers)}")

    return configured_line_set


def look_at(
    obj: bpy.types.Object,
    target: Vector,
    track_axis: str = "-Z",
    up_axis: str = "Y",
) -> None:
    """Rotate an object so its tracking axis points at ``target``."""

    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat(track_axis, up_axis).to_euler()


def create_camera(
    collection: bpy.types.Collection,
    name: str = "Product_Camera",
) -> bpy.types.Object:
    """Create, link, and activate a perspective product camera."""

    camera_data = bpy.data.cameras.new(name)
    camera_data.sensor_width = 36.0
    camera_data.clip_start = 0.01
    camera_data.clip_end = 10.0
    camera = bpy.data.objects.new(name, camera_data)
    collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def create_area_light(
    name: str,
    location: Sequence[float],
    energy: float,
    size: float,
    target: Vector,
    collection: bpy.types.Collection,
    shape: str = "RECTANGLE",
    size_y: float | None = None,
) -> bpy.types.Object:
    """Create an aimed area light with explicit world-space dimensions."""

    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.shape = shape
    light_data.size = size
    if shape == "RECTANGLE" and size_y is not None:
        light_data.size_y = size_y
    if hasattr(light_data, "use_shadow"):
        light_data.use_shadow = True
    if hasattr(light_data, "shadow_soft_size"):
        light_data.shadow_soft_size = max(0.001, min(size, size_y or size) * 0.35)
    light = bpy.data.objects.new(name, light_data)
    collection.objects.link(light)
    light.location = Vector(location)
    look_at(light, target)
    return light


def setup_glass_product_lighting(
    collection: bpy.types.Collection,
    target: Vector,
) -> list[bpy.types.Object]:
    """Use restrained strip-and-fill lighting for transparent laboratory glass.

    The left and right strips remain the main shape-defining sources. Front and
    top fills are intentionally modest so Cycles does not wash the vessel and
    floor toward white. A faint rear fill keeps the rim and wall thickness from
    disappearing when the global exposure is lowered.
    """

    return [
        create_area_light(
            "Left_Strip",
            (0.145, -0.105, 0.145),
            28.0,
            0.032,
            target,
            collection,
            size_y=0.190,
        ),
        create_area_light(
            "Right_Strip",
            (-0.135, 0.020, 0.135),
            34.0,
            0.028,
            target,
            collection,
            size_y=0.175,
        ),
        create_area_light(
            "Front_Fill",
            (-0.040, -0.160, 0.110),
            14.0,
            0.135,
            target,
            collection,
            size_y=0.100,
        ),
        create_area_light(
            "Top_Softbox",
            (0.018, 0.010, 0.270),
            9.0,
            0.120,
            target,
            collection,
            size_y=0.085,
        ),
        create_area_light(
            "Rear_Soft",
            (0.010, 0.145, 0.135),
            6.0,
            0.110,
            target,
            collection,
            size_y=0.085,
        ),
    ]


def orbit_location(target: Vector, view: RenderView) -> Vector:
    """Convert an orbit-view description into a camera world location."""

    azimuth = math.radians(view.azimuth_deg)
    elevation = math.radians(view.elevation_deg)
    horizontal = view.distance * math.cos(elevation)
    return Vector(
        (
            target.x + horizontal * math.cos(azimuth),
            target.y + horizontal * math.sin(azimuth),
            target.z + view.distance * math.sin(elevation),
        )
    )


def render_views(
    camera: bpy.types.Object,
    target: Vector,
    views: Sequence[RenderView],
    output_directory: str | Path,
    filename_prefix: str,
) -> list[Path]:
    """Render each diagnostic orbit view to PNG and return the output paths."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.camera = camera

    rendered: list[Path] = []
    for view in views:
        camera.location = orbit_location(target, view)
        camera.data.lens = view.lens_mm
        look_at(camera, target)
        output_path = output_directory / f"{filename_prefix}_{view.name}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        rendered.append(output_path)
        print(f"Rendered: {output_path}")
    return rendered


def save_blend(filepath: str | Path) -> Path:
    """Save the current Blender scene to an absolute or relative ``.blend`` path."""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return path
