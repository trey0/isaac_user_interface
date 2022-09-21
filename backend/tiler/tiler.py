#!/usr/bin/env python3
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

"""
Turn an OBJ file into 3D Tiles
"""

import argparse
import logging
import os

import numpy as np
from obj_geometry import Geometry
from tile_generator import TileGenerator
from tile_system import TileSystem


def mkdir_for_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


ISS_TILE_CONFIG = {
    "origin": [-8, -12, 3],
    "scale": 32,
    "min_zoom": 0,
    "target_texels_per_tile": 512,
}

FUZE_TILE_CONFIG = {
    "origin": [-0.04, -0.04, 0],
    "scale": 0.32,
    "min_zoom": 0,
    "target_texels_per_tile": 512,
}


def tiler(in_obj, out_dir):
    if os.path.exists(out_dir):
        logging.warning(f"Output directory {out_dir} already exists, not overwriting")
        return

    # config = ISS_TILE_CONFIG
    config = FUZE_TILE_CONFIG

    ts = TileSystem(
        np.array(config["origin"], dtype=np.float64),
        config["scale"],
        "{zoom}/{xi}/{yi}/{zi}",
    )
    generator = TileGenerator(
        "out",
        ts,
        config["min_zoom"],
        config["target_texels_per_tile"],
    )
    geom = Geometry.read(in_obj)

    if 0:
        test_out = os.path.join(out_dir, "test.obj")
        mkdir_for_file(test_out)
        geom.write_obj(test_out)

    print(f"texel size {geom.get_median_texel_size()}")

    generator.generate(geom)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("in_obj", help="input obj file")
    parser.add_argument("out_dir", help="output directory for 3D tiles")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tiler(args.in_obj, args.out_dir)


if __name__ == "__main__":
    main()
