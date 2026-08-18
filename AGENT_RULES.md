# Laboratory Asset Script Contract

The model may create or revise exactly one Blender Python instrument script. The shared toolkit, project
documentation, and reference implementation are immutable context.

The generated script must:

1. Import `workspace/toolkit/lab_blender_toolkit.py` as `lab` using `LAB_TOOLKIT_DIR` (with a `__file__`-derived fallback).
2. Define `build_asset() -> bpy.types.Object` and call it under `if __name__ == "__main__":`.
3. Honor `LAB_ASSET_OUTPUT_DIR`, `LAB_RENDER_ENGINE`, and `LAB_RENDER_RESOLUTION`.
4. Clear/configure the scene; create separate asset, marking, and studio collections; configure camera and lighting;
   render at least three diagnostic views; and save a `.blend` file. The views must jointly show the complete
   object, use meaningfully different angles, and keep silhouette details and graduations inspectable.
5. Save outputs only below `LAB_ASSET_OUTPUT_DIR` when it is set, and add meaningful metadata to the main asset.
6. Prefer toolkit functions. Add local helper geometry only when the toolkit lacks the required topology.
7. Keep physical dimensions explicit and plausible. Use real wall thickness, closed meshes, correct normals, and
   non-self-intersecting profiles.
8. When the target describes a capacity, ensure `lab.profile_capacity_ml(inner_profile)` reaches it. Enlarge an
   undersized inner profile rather than lowering the declared capacity.
9. Fix actual geometry instead of hiding it through camera, lighting, cropping, exposure, background, or material
   changes.
10. Compute vessel graduations from the same inner liquid profile used for capacity. Start integration at the true
    zero-volume point of the usable cavity, accounting for inner-bottom height and base thickness. Non-uniform
    equal-volume spacing is correct for non-uniform cross-sections.
11. For whole-millilitre capacities and intervals, prefer integer literals when calling `add_volume_graduations`.
