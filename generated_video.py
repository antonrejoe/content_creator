
from manim import *

class GeneratedVideo(Scene):
    def construct(self):
        images = ['project_assets/session_20241208_164417/images/scene_1.png', 'project_assets/session_20241208_164417/images/scene_2.png', 'project_assets/session_20241208_164417/images/scene_3.png', 'project_assets/session_20241208_164417/images/scene_4.png', 'project_assets/session_20241208_164417/images/scene_5.png', 'project_assets/session_20241208_164417/images/scene_6.png']
        for img in images:
            image = ImageMobject(img)
            image.set_width(config.frame_width)  # Use config.frame_width for proper scaling
            self.add(image)
            self.wait(1.6666666666666667)
            self.remove(image)
