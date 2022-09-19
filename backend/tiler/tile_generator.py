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

import logging
import os

import numpy as np

from tile_system import Tile, TileSystem
from obj_geometry import Geometry


def dosys(cmd):
    logging.info("%s", cmd)
    ret = os.system(cmd)
    if ret != 0:
        logging.warning("warning: command returned with non-zero return value %s", ret)
    return ret


class TileGenerator(object):
    def __init__(self, out_path, ts, min_zoom, max_zoom, target_texels_per_tile):
        self.out_path = out_path
        self.ts = ts
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.target_texels_per_tile = target_texels_per_tile

        self.zoom_texture_map = {}
        self.input_texel_size = None

    def generate(self, geom):
        if geom.is_empty():
            return

        self.input_texel_size = geom.get_median_texel_size()

        bbox = geom.get_bounding_box()
        min_idx = self.ts.get_index_vec_for_pt(bbox.min_corner, self.min_zoom)
        max_idx = self.ts.get_index_vec_for_pt(bbox.max_corner, self.min_zoom) + 1
        print(f"idx {min_idx} {max_idx}")
        for xi in range(min_idx[0], max_idx[0]):
            for yi in range(min_idx[1], max_idx[1]):
                for zi in range(min_idx[2], max_idx[2]):
                    tile = Tile(self.min_zoom, xi, yi, zi)
                    self.generate_tile(geom, tile)

    def write_tile(self, geom, tile):
        tile_path = os.path.join(self.out_path, self.ts.get_path(tile)) + ".obj"
        os.makedirs(os.path.dirname(tile_path), exist_ok=True)
        geom.write_obj(tile_path, self.zoom_texture_map.get(tile.zoom, {}))

    def scale_texture_images(self, geom, tile):
        texture_map = self.zoom_texture_map.setdefault(tile.zoom, {})
        assert(len(geom.mtl.texture_images) == 1)
        for input_img_path, input_img in geom.mtl.texture_images:
            if input_img_path in texture_map:
                continue

            # calculate scale factor
            input_texels_per_tile = self.ts.get_scale(tile) / self.input_texel_size
            scale_percent = int(100 * self.target_texels_per_tile / input_texels_per_tile)

            # only downsample if space savings are significant
            if scale_percent >= 80:
                continue

            # downsample input texture image for this zoom level
            input_base = os.path.basename(input_img_path)
            output_base = f"zoom{tile.zoom}_{input_base}"
            output_img_path = os.path.realpath(os.path.join(self.out_path, str(tile.zoom), output_base))
            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            dosys(f"convert -resize {scale_percent}% {input_img_path} {output_img_path}")
            rel_output_img_path = os.path.join("..", "..", output_base)
            texture_map[input_img_path] = rel_output_img_path

    def generate_tile(self, geom, tile):
        print(f"generate_tile {tile}")
        print(f"geom bbox {geom.get_bounding_box()}")
        geom = geom.get_cropped(self.ts.get_bounding_box(tile))
        print(f"tile bbox {self.ts.get_bounding_box(tile)}")
        print(f"empty {geom.is_empty()}")
        if geom.is_empty():
            return

        self.scale_texture_images(geom, tile)
        self.write_tile(geom, tile)

        # don't expand children if at max_zoom
        if tile.zoom == self.max_zoom:
            return

        children = (
            Tile(tile.zoom + 1, 2 * tile.xi, 2 * tile.yi, 2 * tile.zi),
            Tile(tile.zoom + 1, 2 * tile.xi, 2 * tile.yi, 2 * tile.zi + 1),
            Tile(tile.zoom + 1, 2 * tile.xi, 2 * tile.yi + 1, 2 * tile.zi),
            Tile(tile.zoom + 1, 2 * tile.xi, 2 * tile.yi + 1, 2 * tile.zi + 1),
            Tile(tile.zoom + 1, 2 * tile.xi + 1, 2 * tile.yi, 2 * tile.zi),
            Tile(tile.zoom + 1, 2 * tile.xi + 1, 2 * tile.yi, 2 * tile.zi + 1),
            Tile(tile.zoom + 1, 2 * tile.xi + 1, 2 * tile.yi + 1, 2 * tile.zi),
            Tile(tile.zoom + 1, 2 * tile.xi + 1, 2 * tile.yi + 1, 2 * tile.zi + 1),
        )
        for child in children:
            self.generate_tile(geom, child)
