import moderngl
from pyrr import Matrix44, Vector3

import models
from base_window import BaseWindow


class ShapesWindow(BaseWindow):
    def __init__(self, **kwargs):
        super(ShapesWindow, self).__init__(**kwargs)

    def load_models(self):
        self.cube_3d = models.load_cube(self.program)
        self.cone = models.load_cone(self.program)
        self.cylynder_3d = models.load_cylinder(self.program)

    def init_shaders_variables(self):
        self.uniform_projection = self.program["projection"]
        self.uniform_view = self.program["view"]
        self.uniform_model = self.program["model"]
        self.uniform_color = self.program["color"]
        self.uniform_camera_position = self.program["camera_position"]
        self.program["material_ambient"].value = (1.0, 0.5, 0.31)
        self.program["material_diffuse"].value = (1.0, 0.5, 0.31)
        self.program["material_specular"].value = (0.5, 0.5, 0.5)
        self.program["material_shininess"].value = 32.0

    def render(self, time: float, frame_time: float):
        self.ctx.clear(0.8, 0.8, 0.8, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        projection = Matrix44.perspective_projection(
            45.0, self.aspect_ratio, 0.1, 1000.0
        )
        view = Matrix44.look_at(
            (5.0, 5.0, -15.0),
            (0.0, 2.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        self.uniform_projection.write(projection.astype("f4"))
        self.uniform_view.write(view.astype("f4"))
        self.uniform_camera_position.value = (5.0, 5.0, -15.0)

        head = Matrix44.from_translation((0, 5, 0))
        head_scale = Matrix44.from_scale((1.9, 1.9, 1.9))
        head *= head_scale
        self.uniform_model.write(head.astype("f4"))
        color = Vector3((0.0, 1.0, 0.0))
        self.uniform_color.write(color.astype("f4"))
        self.cone.render(moderngl.TRIANGLES)

        body = Matrix44.from_translation((0, 2, 0))
        body_scale = Matrix44.from_scale((2, 4, 2))
        body *= body_scale
        self.uniform_model.write(body.astype("f4"))
        color = Vector3((0.6, 0.7, 0.9))
        self.uniform_color.write(color.astype("f4"))
        self.cylynder_3d.render(moderngl.TRIANGLES)

        limb_translations = [(2.5, 4, 0), (-2.5, 4, 0), (-2, -2, 0), (2, -2, 0)]
        limb_angles = [45, -45, 60, -60]
        limb_scales = [(0.75, 2.5, 0.75), (0.75, 2.5, 0.75), (1, 3, 1), (1, 3, 1)]
        for limb_translation, limb_angle, limb_scale in zip(
            limb_translations, limb_angles, limb_scales
        ):
            limb = Matrix44.from_translation(limb_translation)
            limb_scale = Matrix44.from_scale(limb_scale)
            limb_angle = Matrix44.from_z_rotation(limb_angle)
            limb = limb * limb_angle * limb_scale
            self.uniform_model.write(limb.astype("f4"))
            color = Vector3((0.5, 0, 0.5))
            self.uniform_color.write(color.astype("f4"))
            self.cube_3d.render(moderngl.TRIANGLES)
