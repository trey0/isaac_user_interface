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


# UPSAMPLE_FACTOR = 3.0

# A larger value improves quality, but unfortunately, we run into an error in
# example_repack's image loading library if it tries to unpack an image that
# will occupy more than ~ 1 GB memory (it uses 32-bit byte offsets).
UPSAMPLE_FACTOR = 1.0


def dosys(cmd, exc_on_error=True):
    logging.info("%s", cmd)
    ret = os.system(cmd)
    if ret != 0:
        logging.warning("warning: command returned with non-zero return value %s", ret)
        if exc_on_error:
            raise RuntimeError("bailing out after dosys() error")
    return ret


def resize_to(out_dim, in_path, out_path, scale_factor_limit=None):
    logging.info(f"resize_to {out_dim} {in_path} {out_path} {scale_factor_limit}")
    in_img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    in_h, in_w, num_channels = in_img.shape

    # Adjust out_dim to preserve the input aspect ratio. This is the same
    # behavior as ImageMagick "convert -resize WxH".
    out_w, out_h = out_dim
    scale_factor = min(float(out_w) / in_w, float(out_h) / in_h)
    if scale_factor_limit:
        scale_factor = min(scale_factor_limit, scale_factor)
    out_dim_aspect = tuple([int(round(scale_factor * val)) for val in (in_w, in_h)])

    out_img = cv2.resize(in_img, out_dim_aspect, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, out_img)

    return scale_factor


def resize_scale(scale_factor, in_path, out_path):
    logging.info(f"resize_scale {scale_factor} {in_path} {out_path}")
    in_img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    in_h, in_w, num_channels = in_img.shape

    out_dim = tuple([int(round(scale_factor * val)) for val in (in_w, in_h)])

    out_img = cv2.resize(in_img, out_dim, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, out_img)


class TileGenerator(object):
    def __init__(self, out_path, ts, min_zoom, target_texels_per_tile):
        self.out_path = out_path
        self.ts = ts
        self.min_zoom = min_zoom
        self.target_texels_per_tile = target_texels_per_tile

        self.texture_map = {}
        self.input_texel_size = None

    def generate(self, geom):
        if geom.is_empty():
            return

        self.input_texel_size = geom.get_median_texel_size()
        self.scale_texture_images(geom)

        leaf_tiles_path = os.path.join(self.out_path, "build", "leaf_tiles.txt")
        with open(leaf_tiles_path, "w", encoding="utf-8") as leaf_tiles:
            self.leaf_tiles = leaf_tiles
            self.generate_tiles(geom)
            del self.leaf_tiles

    def generate_tiles(self, geom):
        bbox = geom.get_bounding_box()
        min_idx = self.ts.get_index_vec_for_pt(bbox.min_corner, self.min_zoom)
        max_idx = self.ts.get_index_vec_for_pt(bbox.max_corner, self.min_zoom) + 1
        for xi in range(min_idx[0], max_idx[0]):
            for yi in range(min_idx[1], max_idx[1]):
                for zi in range(min_idx[2], max_idx[2]):
                    tile = Tile(self.min_zoom, xi, yi, zi)
                    self.generate_tile(geom, tile, True)

    def get_crop_tile_path(self, tile):
        return (
            os.path.join(self.out_path, "build", self.ts.get_path(tile)) + "_crop"
        )

    def get_repack_tile_path(self, tile):
        return os.path.join(self.out_path, "build", self.ts.get_path(tile)) + "_repack"

    def get_downsample_tile_path(self, tile):
        return os.path.join(self.out_path, "build", self.ts.get_path(tile)) + "_downsample"

    def write_cropped_tile(self, geom, tile):
        crop_tile_path = self.get_crop_tile_path(tile)
        os.makedirs(os.path.dirname(crop_tile_path), exist_ok=True)
        geom.write(crop_tile_path + ".obj", self.texture_map)

    def scale_texture_images(self, geom):
        for input_img_path, input_img in geom.mtllib.materials.values():
            input_base = os.path.basename(input_img_path)
            output_base = f"up_{input_base}"
            output_base = os.path.splitext(output_base)[0] + ".png"
            output_img_path = os.path.realpath(
                os.path.join(self.out_path, "build", output_base)
            )
            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            resize_scale(UPSAMPLE_FACTOR, input_img_path, output_img_path)
            rel_output_img_path = os.path.join("..", "..", "..", output_base)
            self.texture_map[input_img_path] = rel_output_img_path

    def repack_texture(self, geom, tile):
        crop_tile_path = self.get_crop_tile_path(tile)
        repack_tile_path = self.get_repack_tile_path(tile)

        # this fails because example_repack has oversimplified path logic
        # dosys(f"example_repack {crop_tile_path}.obj {repack_tile_path}")

        # instead do this
        common_dir = os.path.dirname(crop_tile_path)
        crop_tile_base = os.path.basename(crop_tile_path)
        repack_tile_base = os.path.basename(repack_tile_path)
        dosys(f"cd {common_dir} && example_repack {crop_tile_base}.obj {repack_tile_base}")

    def get_scale_factor_limit(self):
        return (1.0 / UPSAMPLE_FACTOR)

    def downsample_texture(self, geom, tile, force_full_res):
        repack_tile_path = self.get_repack_tile_path(tile)
        downsample_tile_path = self.get_downsample_tile_path(tile)

        repack_tile_image = repack_tile_path + ".png"
        downsample_tile_image = downsample_tile_path + ".jpg"

        if force_full_res:
            scale_factor = self.get_scale_factor_limit()
            resize_scale(
                scale_factor,
                repack_tile_image,
                downsample_tile_image,
            )
        else:
            scale_factor = resize_to(
                (self.target_texels_per_tile, self.target_texels_per_tile),
                repack_tile_image,
                downsample_tile_image,
                scale_factor_limit=self.get_scale_factor_limit(),
            )

        repack_geom = Geometry.read(repack_tile_path + ".obj")
        assert len(repack_geom.mtllib.materials) == 1
        rel_repack_tile_image, repack_mtl_img = next(iter(repack_geom.mtllib.materials.values()))
        rel_downsample_tile_image = os.path.basename(downsample_tile_image)
        downsample_texture_map = {
            rel_repack_tile_image: rel_downsample_tile_image,
        }
        repack_geom.write(downsample_tile_path + ".obj", downsample_texture_map)

        return scale_factor

    def generate_tile(self, parent_geom, tile, root):
        geom = parent_geom.get_cropped(self.ts.get_bounding_box(tile))
        if geom.is_empty():
            return

        self.write_cropped_tile(geom, tile)
        self.repack_texture(geom, tile)

        # If cropping didn't discard any faces, go ahead and force full-res
        # output at this zoom level. Further splitting the tile is not likely
        # to help. This is a corner case for our poor-man's cropping approach
        # that discards faces but can't crop a face to fit a tile. The
        # recursion can blow up if it encounters a face whose full-res texture
        # image is larger than the target tile image size.
        force_full_res = (not root and (len(parent_geom.f) == len(geom.f)))

        scale_factor = self.downsample_texture(geom, tile, force_full_res)

        # don't expand children if this tile is already at the full
        # source resolution
        if scale_factor == self.get_scale_factor_limit():
            self.leaf_tiles.write(self.get_downsample_tile_path(tile) + ".obj\n")
            return

        for xo, yo, zo in itertools.product([0, 1], repeat=3):
            child = Tile(
                tile.zoom + 1, 2 * tile.xi + xo, 2 * tile.yi + yo, 2 * tile.zi + zo
            )
            self.generate_tile(geom, child, False)
