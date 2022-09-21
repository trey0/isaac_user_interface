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

import itertools
import logging
import math
import os

import cv2
import numpy as np
from obj_geometry import Geometry
from tile_system import Tile, TileSystem

CONVERT_CMD = "convert"
# CONVERT_CMD = "convert -limit memory 8GiB -limit width 30KP -limit height 30KP"
UPSAMPLE_FACTOR = 3.0


def dosys(cmd, exc_on_error=True):
    logging.info("%s", cmd)
    ret = os.system(cmd)
    if ret != 0:
        logging.warning("warning: command returned with non-zero return value %s", ret)
        if exc_on_error:
            raise RuntimeError("bailing out after dosys() error")
    return ret


def resize_to(out_dim, in_path, out_path):
    in_img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    in_h, in_w, num_channels = in_img.shape

    # Adjust out_dim to preserve the input aspect ratio. This is the same
    # behavior as ImageMagick "convert -resize WxH".
    out_w, out_h = out_dim
    scale_factor = min(out_w0 / in_w, out_h0 / in_h)
    out_dim_aspect = tuple([int(round(scale_factor * val)) for val in (out_w0, out_h0)])

    out_img = cv2.resize(in_img, out_dim_aspect, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, out_img)

    return scale_factor


def resize_scale_percent(scale_percent, in_path, out_path):
    print(f"in_path {in_path}")
    in_img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    in_h, in_w, num_channels = in_img.shape

    scale_factor = scale_percent / 100
    out_dim0 = tuple([scale_factor * val for val in (in_w, in_h)])
    out_dim = tuple([int(round(scale_factor * val)) for val in (in_w, in_h)])

    out_img = cv2.resize(in_img, out_dim, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, out_img)


class TileGenerator(object):
    def __init__(self, out_path, ts, min_zoom, target_texels_per_tile):
        self.out_path = out_path
        self.ts = ts
        self.min_zoom = min_zoom
        self.target_texels_per_tile = target_texels_per_tile

        self.zoom_texture_map = {}
        self.input_texel_size = None

    def generate(self, geom):
        if geom.is_empty():
            return

        self.input_texel_size = geom.get_median_texel_size()
        texel_size_zoom0 = self.ts.get_zoom_scale(0) / self.target_texels_per_tile
        # first zoom level at which target texel size is smaller than source
        # imagery texel size
        self.max_zoom = math.ceil(math.log(texel_size_zoom0 / self.input_texel_size, 2))

        bbox = geom.get_bounding_box()
        min_idx = self.ts.get_index_vec_for_pt(bbox.min_corner, self.min_zoom)
        max_idx = self.ts.get_index_vec_for_pt(bbox.max_corner, self.min_zoom) + 1
        for xi in range(min_idx[0], max_idx[0]):
            for yi in range(min_idx[1], max_idx[1]):
                for zi in range(min_idx[2], max_idx[2]):
                    tile = Tile(self.min_zoom, xi, yi, zi)
                    self.generate_tile(geom, tile)

    def get_crop_tile_path(self, geom, tile):
        return (
            os.path.join(self.out_path, "build", self.ts.get_path(tile)) + "_crop"
        )

    def get_repack_tile_path(self, geom, tile):
        return os.path.join(self.out_path, "build", self.ts.get_path(tile)) + "_repack"

    def get_downsample_tile_path(self, geom, tile):
        return os.path.join(self.out_path, "build", self.ts.get_path(tile)) + "_downsample"

    def crop_tile(self, geom, tile):
        crop_tile_path = self.get_crop_tile_path(geom, tile)
        os.makedirs(os.path.dirname(crop_tile_path), exist_ok=True)
        geom.write(crop_tile_path + ".obj", self.zoom_texture_map.get(tile.zoom, {}))

    def scale_texture_images(self, geom, tile):
        texture_map = self.zoom_texture_map.setdefault(tile.zoom, {})
        for input_img_path, input_img in geom.mtllib.materials.values():
            if input_img_path in texture_map:
                continue

            # calculate scale factor
            input_texels_per_tile = self.ts.get_scale(tile) / self.input_texel_size
            scale_percent = 100 * self.target_texels_per_tile / input_texels_per_tile
            # no point in producing final tiles higher-res than the original texture
            scale_percent = min(scale_percent, 100)

            # The texture repack image processing step later produces pixel
            # aliasing artifacts. To mitigate these artifacts, we'll run the
            # repack at higher resolution than the final target resolution
            # (UPSAMPLE_FACTOR), then downsample to the final resolution at the
            # end.
            tmp_scale_percent = scale_percent * UPSAMPLE_FACTOR

            # scale input texture image for this zoom level
            input_base = os.path.basename(input_img_path)
            output_base = f"zoom{tile.zoom}_{input_base}"
            output_base = os.path.splitext(output_base)[0] + ".png"
            output_img_path = os.path.realpath(
                os.path.join(self.out_path, "build", str(tile.zoom), output_base)
            )
            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            #dosys(
            #    f"{CONVERT_CMD} -resize {tmp_scale_percent}% {input_img_path} {output_img_path}"
            #)
            resize_scale_percent(tmp_scale_percent, input_img_path, output_img_path)
            rel_output_img_path = os.path.join("..", "..", output_base)
            texture_map[input_img_path] = rel_output_img_path

    def repack_texture(self, geom, tile):
        crop_tile_path = self.get_crop_tile_path(geom, tile)
        repack_tile_path = self.get_repack_tile_path(geom, tile)

        # this fails because example_repack has oversimplified path logic
        # dosys(f"example_repack {crop_tile_path}.obj {repack_tile_path}")

        # instead do this
        common_dir = os.path.dirname(crop_tile_path)
        crop_tile_base = os.path.basename(crop_tile_path)
        repack_tile_base = os.path.basename(repack_tile_path)
        dosys(f"cd {common_dir} && example_repack {crop_tile_base}.obj {repack_tile_base}")

    def downsample_texture(self, geom, tile):
        repack_tile_path = self.get_repack_tile_path(geom, tile)
        downsample_tile_path = self.get_downsample_tile_path(geom, tile)

        scale_percent = 100. / UPSAMPLE_FACTOR
        #dosys(
        #    f"{CONVERT_CMD} -resize {scale_percent}% {repack_tile_path}.png {downsample_tile_path}.jpg"
        #)
        resize_scale_percent(scale_percent, repack_tile_path + ".png", downsample_tile_path + ".jpg")
        dosys(f"cp {repack_tile_path}.obj {downsample_tile_path}.obj")

    def generate_tile(self, geom, tile):
        geom = geom.get_cropped(self.ts.get_bounding_box(tile))
        if geom.is_empty():
            return

        self.scale_texture_images(geom, tile)
        self.crop_tile(geom, tile)
        self.repack_texture(geom, tile)
        self.downsample_texture(geom, tile)

        # don't expand children if at max_zoom
        if tile.zoom == self.max_zoom:
            return

        for xo, yo, zo in itertools.product([0, 1], repeat=3):
            child = Tile(
                tile.zoom + 1, 2 * tile.xi + xo, 2 * tile.yi + yo, 2 * tile.zi + zo
            )
            self.generate_tile(geom, child)
