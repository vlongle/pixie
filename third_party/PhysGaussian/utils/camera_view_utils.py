import os
import json
import numpy as np
import torch
from scene.cameras import Camera as GSCamera
from utils.graphics_utils import focal2fov


# === Utility ===
# Rodrigues formula for rotation matrix around arbitrary axis
def _rodrigues_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Return 3×3 rotation matrix that rotates *angle_deg* degrees around *axis* (right-hand rule)."""
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def generate_camera_rotation_matrix(camera_to_object, object_vertical_downward):
    camera_to_object = camera_to_object / np.linalg.norm(
        camera_to_object
    )  # last column
    # the second column of rotation matrix is pointing toward the downward vertical direction
    camera_y = (
        object_vertical_downward
        - np.dot(object_vertical_downward, camera_to_object) * camera_to_object
    )
    camera_y = camera_y / np.linalg.norm(camera_y)  # second column
    first_column = np.cross(camera_y, camera_to_object)
    R = np.column_stack((first_column, camera_y, camera_to_object))
    return R


# supply vertical vector in world space
def generate_local_coord(vertical_vector):
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
    horizontal_1 = np.array([1, 1, 1])
    if np.abs(np.dot(horizontal_1, vertical_vector)) < 0.01:
        horizontal_1 = np.array([0.72, 0.37, -0.67])
    # gram schimit
    horizontal_1 = (
        horizontal_1 - np.dot(horizontal_1, vertical_vector) * vertical_vector
    )
    horizontal_1 = horizontal_1 / np.linalg.norm(horizontal_1)
    horizontal_2 = np.cross(horizontal_1, vertical_vector)

    return vertical_vector, horizontal_1, horizontal_2


# scalar (in degrees), scalar (in degrees), scalar, vec3, mat33 = [horizontal_1; horizontal_2; vertical];  -> vec3
def get_point_on_sphere(azimuth, elevation, radius, center, observant_coordinates):
    canonical_coordinates = (
        np.array(
            [
                np.cos(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(elevation / 180.0 * np.pi),
            ]
        )
        * radius
    )

    return center + observant_coordinates @ canonical_coordinates


def get_camera_position_and_rotation(
    azimuth,
    elevation,
    radius,
    roll,
    view_center,
    observant_coordinates,
):
    """Return (position, 3×3 rotation) for the given spherical coordinates & roll.

    Parameters
    ----------
    azimuth, elevation : float
        Standard spherical coordinates in *degrees* (0° azimuth = +h1 axis).
    radius : float
        Distance from camera to *view_center*.
    roll : float
        Rotation *around the viewing axis* in degrees.  0° keeps the camera's +X
        aligned with *observant_coordinates*[:,0]; positive values rotate towards
        *observant_coordinates*[:,1] (right-hand rule).
    view_center : (3,) np.ndarray
    observant_coordinates : (3,3) np.ndarray
        Local orthonormal basis returned by
        `get_center_view_worldspace_and_observant_coordinate` (h1, h2, vertical).
    """

    # --- position on sphere -------------------------------------------------
    position = get_point_on_sphere(
        azimuth, elevation, radius, view_center, observant_coordinates
    )

    # --- base rotation (no roll) -------------------------------------------
    R_base = generate_camera_rotation_matrix(
        view_center - position, -observant_coordinates[:, 2]
    )

    # --- apply roll around forward axis ------------------------------------
    if roll is None:
        roll = 0.0
    R_roll = _rodrigues_rotation(R_base[:, 2], roll)
    R = R_roll @ R_base

    return position, R


def get_current_radius_azimuth_elevation_roll(
    camera_position,
    camera_rotation,
    view_center,
    observant_coordinates,
):
    center2camera = -view_center + camera_position
    radius = np.linalg.norm(center2camera)
    dot_product = np.dot(center2camera, observant_coordinates[:, 2])
    cosine = dot_product / (
        np.linalg.norm(center2camera) * np.linalg.norm(observant_coordinates[:, 2])
    )
    elevation = np.rad2deg(np.pi / 2.0 - np.arccos(cosine))
    proj_onto_hori = center2camera - dot_product * observant_coordinates[:, 2]
    dot_product2 = np.dot(proj_onto_hori, observant_coordinates[:, 0])
    cosine2 = dot_product2 / (
        np.linalg.norm(proj_onto_hori) * np.linalg.norm(observant_coordinates[:, 0])
    )

    if np.dot(proj_onto_hori, observant_coordinates[:, 1]) > 0:
        azimuth = np.rad2deg(np.arccos(cosine2))
    else:
        azimuth = -np.rad2deg(np.arccos(cosine2))

    # --- compute roll via relative rotation --------------------------------
    cam2world_R = camera_rotation.T  # convert to camera->world

    # Base orientation with zero roll (constructed like in forward synthesis)
    R_base = generate_camera_rotation_matrix(
        view_center - camera_position,
        -observant_coordinates[:, 2],
    )

    # rotation that maps base frame to the actual camera frame
    R_rel = cam2world_R @ R_base.T

    # For a pure roll, R_rel is a rotation about the Z (forward) axis, so
    # we can recover the angle from the upper-left 2×2 block.
    roll_rad = np.arctan2(R_rel[1, 0], R_rel[0, 0])
    roll = np.rad2deg(roll_rad)

    return radius, azimuth, elevation, roll


def get_camera_view(
    model_path,
    default_camera_index=0,
    center_view_world_space=None,
    observant_coordinates=None,
    show_hint=False,
    init_azimuthm=None,
    init_elevation=None,
    init_radius=None,
    init_roll=None,
    move_camera=False,
    current_frame=0,
    delta_a=0,
    delta_e=0,
    delta_r=0,
    delta_roll=0,
):
    """Load one of the default cameras for the scene."""
    ## check if model_path is a directory
    if os.path.isdir(model_path):
        cam_path = os.path.join(model_path, "cameras.json")
    else:
        cam_path = os.path.join(os.path.dirname(model_path), "cameras.json")

    with open(cam_path) as f:
        data = json.load(f)

        if show_hint:
            if default_camera_index < 0:
                default_camera_index = 0
            r, a, e, roll = get_current_radius_azimuth_elevation_roll(
                data[default_camera_index]["position"],
                np.asarray(data[default_camera_index]["rotation"]),
                center_view_world_space,
                observant_coordinates,
            )
            print("Default camera ", default_camera_index, " has")
            print("azimuth:    ", a)
            print("elevation:  ", e)
            print("radius:     ", r)
            print("roll:       ", roll)
            print("Now exit program and set your own input!")
            exit()

        if default_camera_index > -1:
            raw_camera = data[default_camera_index]

        else:
            raw_camera = data[0]  # get data to be modified

            assert init_azimuthm is not None
            assert init_elevation is not None
            assert init_radius is not None
            if init_roll is None:
                init_roll = 0.0

            if move_camera:
                assert delta_a is not None
                assert delta_e is not None
                assert delta_r is not None
                assert delta_roll is not None
                position, R = get_camera_position_and_rotation(
                    init_azimuthm + current_frame * delta_a,
                    init_elevation + current_frame * delta_e,
                    init_radius + current_frame * delta_r,
                    init_roll + current_frame * delta_roll,
                    center_view_world_space,
                    observant_coordinates,
                )
            else:
                position, R = get_camera_position_and_rotation(
                    init_azimuthm,
                    init_elevation,
                    init_radius,
                    init_roll,
                    center_view_world_space,
                    observant_coordinates,
                )
            raw_camera["rotation"] = R.tolist()
            raw_camera["position"] = position.tolist()

        tmp = np.zeros((4, 4))
        tmp[:3, :3] = raw_camera["rotation"]
        tmp[:3, 3] = raw_camera["position"]
        tmp[3, 3] = 1
        C2W = np.linalg.inv(tmp)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]

        width = raw_camera["width"]
        height = raw_camera["height"]
        fovx = focal2fov(raw_camera["fx"], width)
        fovy = focal2fov(raw_camera["fy"], height)

        return GSCamera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros((3, height, width)),  # fake
            gt_alpha_mask=None,
            image_name="fake",
            uid=0,
        )
