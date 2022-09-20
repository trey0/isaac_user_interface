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

import array
import os
import re

import cv2
import numpy as np
from tile_system import BoundingBox


def parse_face_vertex(v):
    idx_strings = v.split("/")
    idx = [(-1 if s == "" else int(s)) for s in idx_strings]
    idx.extend([-1] * (3 - len(idx)))
    return idx


def dump_face_vertex(idx):
    if idx[1] == -1:
        if idx[2] == -1:
            return str(idx[0])
        else:
            return "%s//%s" % (idx[0], idx[2])
    elif idx[2] == -1:
        return "%s/%s" % (idx[0], idx[1])
    else:
        return "%s/%s/%s" % (*idx,)


INT32_MAX = np.iinfo(np.int32).max


def garbage_collect(in_refs, in_objects):
    keep_refs = np.unique(in_refs)

    # fill invalid entries in reference index array with INT32_MAX in order to
    # force an error if we accidentally dereference them
    remap_refs = np.full(in_objects.shape[0], INT32_MAX, dtype=np.int32)
    remap_refs[keep_refs] = np.arange(keep_refs.size)

    out_refs = remap_refs[in_refs]
    out_objects = in_objects[keep_refs]

    return out_refs, out_objects


def get_side_lengths(tris):
    print(f"tris shape {tris.shape}")
    # dtris = np.diff(tris, axis=1, append=tris[:, 0, :])
    dtris = np.diff(tris, axis=1)
    return np.linalg.norm(dtris, axis=2)


class MtlLib(object):
    def __init__(self, input_path, materials, lines):
        self.input_path = input_path
        self.materials = materials
        self.lines = lines

    @classmethod
    def read(cls, input_path):
        input_path = os.path.realpath(input_path)

        lines = []
        materials = {}
        mtl_name = None
        with open(input_path, "r", encoding="utf-8") as inp:
            for line in inp:
                lines.append(line)
                if line == "#":
                    continue
                fields = line.split(None, 1)
                if len(fields) < 2:
                    continue
                cmd, arg = fields
                arg = arg.rstrip()

                if cmd == "newmtl":
                    mtl_name = arg

                elif cmd == "map_Kd":
                    mtl_image_path = arg
                    full_image_path = os.path.realpath(
                        os.path.join(os.path.dirname(input_path), mtl_image_path)
                    )
                    img = cv2.imread(full_image_path)
                    materials[mtl_name] = (mtl_image_path, img)

        return MtlLib(input_path, materials, lines)

    def write(self, output_path, texture_map):
        with open(output_path, "w", encoding="utf-8") as out:
            for line in self.lines:
                if line == "#":
                    out.write(line)
                    continue
                fields = line.split(None, 1)
                if len(fields) < 2:
                    out.write(line)
                    continue
                cmd, arg = fields
                arg = arg.rstrip()

                if cmd == "map_Kd":
                    input_texture_image = arg
                    output_texture_image = texture_map.get(
                        input_texture_image, input_texture_image
                    )
                    out.write(f"map_Kd {output_texture_image}")
                else:
                    out.write(line)


class Geometry(object):
    def __init__(self, input_path, v, vt, vn, f, mtllib, other_cmds):
        self.input_path = input_path
        self.v = v
        self.vt = vt
        self.vn = vn
        self.f = f
        self.mtllib = mtllib
        self.other_cmds = other_cmds

    @classmethod
    def read(cls, input_path):
        input_path = os.path.realpath(input_path)

        v = array.array("d")
        vt = array.array("d")
        vn = array.array("d")
        f = array.array("l")
        mtllib = None
        other_cmds = []

        with open(input_path, "r", encoding="utf-8") as inp:
            for line in inp:
                if line.startswith("#"):
                    continue

                fields = line.split()
                if not fields:
                    continue

                cmd = fields[0]
                args = fields[1:]

                if cmd == "v":
                    assert len(args) == 3
                    v.extend((float(a) for a in args))

                elif cmd == "vt":
                    assert len(args) == 2
                    vt.extend((float(a) for a in args))

                elif cmd == "vn":
                    assert len(args) == 3
                    vn.extend((float(a) for a in args))

                elif cmd == "f":
                    assert len(args) == 3
                    for a in args:
                        f.extend(parse_face_vertex(a))

                elif cmd == "mtllib":
                    assert len(args) == 1
                    mtl_path = args[0]
                    input_mtl_path = os.path.join(os.path.dirname(input_path), mtl_path)
                    mtllib = MtlLib.read(input_mtl_path)

                elif cmd in ("usemtl",):
                    other_cmds.append(line)

                else:
                    print(
                        f"WARNING: Geometry.load(): unknown command '{cmd}', ignoring"
                    )

        # convert to numpy
        v = np.array(v, dtype=np.float64).reshape((-1, 3))
        vt = np.array(vt, dtype=np.float64).reshape((-1, 2))
        vn = np.array(vn, dtype=np.float64).reshape((-1, 3))
        f = np.array(f, dtype=np.int32).reshape((-1, 3, 3))

        # convert from OBJ 1-based indexing to Python 0-based indexing
        f = f - 1

        return Geometry(input_path, v, vt, vn, f, mtllib, other_cmds)

    def write(self, output_path, texture_map={}):
        output_path = os.path.realpath(output_path)

        if self.mtllib:
            output_mtl_path = os.path.splitext(output_path)[0] + ".mtl"
            self.mtllib.write(output_mtl_path, texture_map)

        with open(output_path, "w", encoding="utf-8") as out:
            if self.mtllib:
                mtl_from_output_path = os.path.relpath(
                    output_mtl_path, os.path.dirname(output_path)
                )
                out.write(f"mtllib {mtl_from_output_path}\n")
            for line in self.other_cmds:
                out.write(line)
            for v in self.v:
                out.write("v %s %s %s\n" % (*v,))
            for vt in self.vt:
                out.write("vt %s %s\n" % (*vt,))
            for vn in self.vn:
                out.write("vn %s %s %s\n" % (*vn,))

            # convert from Python 0-based indexing to OBJ 1-based indexing
            out_f = self.f + 1

            for f in out_f:
                out.write("f %s %s %s\n" % tuple(dump_face_vertex(v) for v in f))

    def get_cropped(self, bbox):
        # keep faces whose centroids are within the bounding box
        face_centroids = np.mean(self.v[self.f[:, :, 0], :], axis=1)
        keep_faces = bbox.is_inside(face_centroids)
        f = self.f[keep_faces]

        # keep only the vertex information that is referenced by the remaining
        # faces
        f[:, :, 0], v = garbage_collect(f[:, :, 0], self.v)
        f[:, :, 1], vt = garbage_collect(f[:, :, 1], self.vt)
        f[:, :, 2], vn = garbage_collect(f[:, :, 2], self.vn)

        return Geometry(self.input_path, v, vt, vn, f, self.mtllib, self.other_cmds)

    def get_bounding_box(self):
        return BoundingBox(np.amin(self.v, axis=0), np.amax(self.v, axis=0))

    def is_empty(self):
        return self.f.size == 0

    def get_median_texel_size(self):
        xyz_tris = self.v[self.f[:, :, 0], :]
        # This is only using two sides of each triangle. We could probably fix
        # it to use all three with the append=... arg to np.diff. A bit hard to figure
        # out exactly what to pass it. Same goes for uv_diffs below.
        xyz_diffs = np.diff(xyz_tris, axis=1).reshape((-1, 3))
        xyz_lengths = np.linalg.norm(xyz_diffs, axis=1)

        uv_tris = self.vt[self.f[:, :, 1], :]
        uv_diffs = np.diff(uv_tris, axis=1).reshape((-1, 2))

        assert len(self.mtllib.materials) == 1
        texture_path, texture_img = next(iter(self.mtllib.materials.values()))
        img_size = np.array(texture_img.shape[:2], dtype=np.int32)[:, np.newaxis]
        texel_diffs = np.matmul(uv_diffs, img_size)
        texel_lengths = np.linalg.norm(texel_diffs, axis=1)

        non_zero = texel_lengths != 0
        texel_size = xyz_lengths[non_zero] / texel_lengths[non_zero]

        return np.median(texel_size)
