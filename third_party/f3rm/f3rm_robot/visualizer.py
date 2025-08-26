import tempfile
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
import open3d as o3d


class BaseVisualizer(ABC):
    @abstractmethod
    def add_point_cloud(self, key: str, points: np.ndarray, colors: np.ndarray, **kwargs):
        raise NotImplementedError

    def add_o3d_point_cloud(self, key: str, point_cloud: o3d.geometry.PointCloud, **kwargs):
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        return self.add_point_cloud(key, points, colors, **kwargs)

    @abstractmethod
    def add_mesh(self, key: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, **kwargs):
        raise NotImplementedError

    def add_o3d_mesh(self, key: str, mesh: o3d.geometry.TriangleMesh, **kwargs):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors)
        return self.add_mesh(key, vertices, faces, colors, **kwargs)


class ViserVisualizer(BaseVisualizer):
    """Viser visualizer: https://github.com/nerfstudio-project/viser"""

    def __init__(self, host: str, port: int):
        from viser import ViserServer
        from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

        self.server = ViserServer(host=host, port=port)
        self.host = host
        # Port may have changed...
        # self.port = self.server._server._port
        self.port = self.server.get_port()

        # Configure the theme following: https://viser.studio/examples/13_theming/
        buttons = (
            TitlebarButton(text="GitHub", icon="GitHub", href="https://github.com/f3rm/f3rm"),
            TitlebarButton(
                text="Visualizer Guide",
                icon="Description",
                href="https://github.com/f3rm/f3rm/tree/main/f3rm_robot#using-the-visualizer",
            ),
        )

        image_url = "https://raw.githubusercontent.com/f3rm/f3rm/main/assets/images/ff_icon.png"
        image = TitlebarImage(
            image_url_light=image_url,
            image_url_dark=image_url,
            image_alt="F3RM Icon",
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        self.server.configure_theme(titlebar_content=titlebar_theme)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def add_query_gui(self) -> Tuple[Callable[[], str], Callable[[], None]]:
        """
        Add the GUI for accepting user queries for language-guided manipulation. Returns two callables:
            1. wait_for_input: block until the user submits a query, then returns the query
            2. enable_gui: enable the GUI, so we can accept another query

        This is a bit hacky and uses threading.Event to block the main thread until the user submits a query.
        """
        from threading import Event

        with self.server.add_gui_folder("Language Query"):
            gui_text = self.server.add_gui_text(
                label="Query", initial_value="", hint="Enter the object you want to grasp."
            )
            gui_submit = self.server.add_gui_button("Submit")

        click_event = Event()

        def click_handler(*_):
            # Set the event to unblock the main thread, and disable the GUI
            click_event.set()
            gui_text.disabled = True
            gui_submit.disabled = True

        gui_submit.on_click(lambda _: click_handler())

        def wait_for_input():
            """Wait for user to submit a query using the button."""
            # Block until event is set and clear it
            click_event.wait()
            click_event.clear()
            return gui_text.value

        def enable_gui():
            # Enable the GUI, so we can accept another query and clear the text
            gui_text.disabled = False
            gui_text.value = ""
            gui_submit.disabled = False

        return wait_for_input, enable_gui

    def add_point_cloud(self, key: str, points: np.ndarray, colors: np.ndarray, **kwargs):
        return self.server.add_point_cloud(key, points, colors, **kwargs)

    def add_mesh(self, key: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, **kwargs):
        return self.server.add_mesh_simple(key, vertices, faces, colors, **kwargs)

    def add_o3d_mesh(self, key: str, mesh: o3d.geometry.TriangleMesh, **kwargs):
        # Viser only supports single RGB colors for meshes, so we write it to a GLB file. Viser does this for
        # trimesh. Since we write and read it isn't optimal, but it suffices for now.
        with tempfile.NamedTemporaryFile(suffix=".glb") as tmp_file:
            o3d.io.write_triangle_mesh(tmp_file.name, mesh)
            with open(tmp_file.name, "rb") as f:
                glb_data = f.read()
            self.server.add_glb(key, glb_data, **kwargs)
