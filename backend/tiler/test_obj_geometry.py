# Copyright © 2021, United States Government, as represented by the Administrator of the
# National Aeronautics and Space Administration. All rights reserved.
#
# The “ISAAC - Integrated System for Autonomous and Adaptive Caretaking platform” software is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np

from obj_geometry import Geometry
from tile_system import BoundingBox

geom = Geometry.read_obj("fuze.obj")

bbox = geom.get_bounding_box()
min_c = bbox.min_corner
max_c = bbox.max_corner
center_z = np.mean((min_c[2], max_c[2]))
max_cp = max_c.copy()
max_cp[2] = center_z

# cut off top half of bottle
geom2 = geom.get_cropped(BoundingBox(min_c, max_cp))

geom.write_obj("fuze_copy.obj")
geom2.write_obj("fuze_copy_cropped.obj")
