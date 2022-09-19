import numpy as np

DEFAULT_TILE_PATH = "{zoom}/{xi}/{yi}/{zi}"


class Tile(object):
    def __init__(self, zoom, xi, yi, zi):
        self.zoom = zoom
        self.xi = xi
        self.yi = yi
        self.zi = zi

    def __repr__(self):
        return f"<Tile {self.zoom}/{self.xi}/{self.yi}/{self.zi}>"

    @classmethod
    def from_zoom_int_vec(zoom, int_vec):
        self.zoom = zoom
        self.xi, self.yi, self.zi = int_vec


class BoundingBox(object):
    def __init__(self, min_corner, max_corner):
        self.min_corner = min_corner
        self.max_corner = max_corner

    def __repr__(self):
        return f"<BoundingBox {self.min_corner} {self.max_corner}>"

    def is_inside(self, pts):
        return np.logical_and(
            np.all(self.min_corner <= pts, axis=1),
            np.all(pts < self.max_corner, axis=1),
        )


class TileSystem(object):
    def __init__(self, tile_origin, tile_scale, tile_path=DEFAULT_TILE_PATH):
        self.tile_origin = tile_origin
        self.tile_scale = tile_scale
        self.tile_path = tile_path

    def get_path(self, tile):
        return self.tile_path.format_map(tile.__dict__)

    def get_zoom_scale(self, zoom):
        return self.tile_scale / 2**zoom

    def get_scale(self, tile):
        return self.get_zoom_scale(tile.zoom)

    def get_index_vec(self, tile):
        return np.array([tile.xi, tile.yi, tile.zi], dtype=np.int32)

    def get_bounding_box(self, tile):
        scale = self.get_scale(tile)
        min_corner = self.tile_origin + scale * self.get_index_vec(tile)
        return BoundingBox(min_corner, min_corner + scale)

    def get_centroid(self, tile):
        return self.tile_origin + self.get_scale() * (
            self.get_index_vec(tile) + 0.5 * np.ones(3)
        )

    def get_index_vec_for_pt(self, pt, zoom):
        return ((pt - self.tile_origin) / self.get_zoom_scale(zoom)).astype(np.int32)
