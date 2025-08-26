import os

import open3d as o3d

_MODULE_PATH = os.path.dirname(__file__)


def get_asset_path(asset_name: str) -> str:
    return os.path.join(_MODULE_PATH, asset_name)


def get_panda_gripper_mesh() -> o3d.geometry.TriangleMesh:
    asset_path = get_asset_path("panda_gripper_visual.obj")
    return o3d.io.read_triangle_mesh(asset_path)
