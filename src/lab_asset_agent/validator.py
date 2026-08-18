from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]


FORBIDDEN_IMPORT_ROOTS = {
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "urllib",
    "ftplib",
    "telnetlib",
    "paramiko",
}
FORBIDDEN_CALL_NAMES = {"eval", "exec", "compile", "__import__"}
FORBIDDEN_ATTR_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("shutil", "rmtree"),
    ("shutil", "move"),
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("os", "removedirs"),
    ("os", "rename"),
    ("os", "replace"),
}


def validate_generated_script(path: Path) -> ValidationResult:
    errors: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ValidationResult(False, [f"Cannot read script: {exc}"])

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return ValidationResult(False, [f"Syntax error: {exc}"])

    has_build_asset = False
    has_main_guard = False
    imports_bpy = False
    imports_toolkit = "lab_blender_toolkit" in source

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_asset":
            has_build_asset = True
        elif isinstance(node, ast.If):
            test = ast.dump(node.test, include_attributes=False)
            if "__name__" in test and "__main__" in test:
                has_main_guard = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                imports_bpy = imports_bpy or root == "bpy"
                if root in FORBIDDEN_IMPORT_ROOTS:
                    errors.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                errors.append(f"Forbidden import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                errors.append(f"Forbidden dynamic call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"unlink", "rmdir"}:
                errors.append(f"Forbidden destructive method: {node.func.attr}")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                pair = (node.func.value.id, node.func.attr)
                if pair in FORBIDDEN_ATTR_CALLS:
                    errors.append(f"Forbidden call: {pair[0]}.{pair[1]}")

    if not imports_bpy:
        errors.append("Script must import bpy.")
    if not imports_toolkit:
        errors.append("Script must import lab_blender_toolkit.")
    if not has_build_asset:
        errors.append("Script must define build_asset().")
    if not has_main_guard:
        errors.append('Script must contain if __name__ == "__main__" guard.')
    if "LAB_ASSET_OUTPUT_DIR" not in source:
        errors.append("Script must honor LAB_ASSET_OUTPUT_DIR.")
    if "render_views" not in source:
        errors.append("Script should use lab.render_views for diagnostic images.")
    if "save_blend" not in source:
        errors.append("Script should save a .blend file with lab.save_blend.")

    return ValidationResult(not errors, errors)
