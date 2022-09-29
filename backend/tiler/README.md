# NASA ISAAC User Interface: 3D Tiles Subsystem

[![GitHub license](https://img.shields.io/github/license/nasa/isaac_user_interface)](https://github.com/nasa/isaac_user_interface/blob/master/LICENSE)

### This section contains the 3D Tiles subsystem of the UI which is part of the backend.

![3d tiles diagram (credit: CesiumGS)](https://raw.githubusercontent.com/CesiumGS/3d-tiles/main/specification/figures/tree.png)

---

## Installation

If working in an Ubuntu 20.04 environment, you can install with:
```bash
setup/install_local.sh
```

Or install in a Docker container with:
```bash
setup/build.sh
```

---

## Example usage

First, install some sample data with:
```bash
setup/install_sample_data.sh
```

Then, in your local environment, you can run with:
```bash
./tiler.py --debug-glb --debug-tileset setup/data/fuze.obj setup/data/out
```

Or in the Docker container, you can run with:
```bash
setup/run.sh /tiler/tiler.py --debug-glb --debug-tileset /data/fuze.obj /data/out
```

In either case, output should be written to the `data/out` directory,
and you should be able to follow the directions at the end of the build
to use one of the debug viewers to view the tile set in your browser.

Run `tiler.py --help` for more information about options.

---

## Notes

The main idea behind 3D Tiles is segmenting a 3D object into multiple tiles. Each tile can contain several smaller sub-tiles each with a higher resolution texture than the parent tile. The viewer will intelligently select, load, and render the correct tiles according to the viewer's orientation and zoom level. Each tile has a pre-determined geometric error that would result from not loading the tile (in our case, based on the texture resolution). The viewer chooses to load a higher resolution tile in place of a lower resolution tile based on a geometric error threshold. A leaf tile has no sub-tiles, uses full resolution texture, and has zero geometric error.

Below is an example of a 3D tiles tileset.

```json
{
    "asset": {
        "version": "1.0",
        "gltfUpAxis": "Z"
    },
    "root": {
        "children": [...]
    },
    "geometricError": 100
}
```

Within the `children` array, tiles are added, with the first tile representing the total bounding volume of the full 3D model. Below is an example of a tile definition. Note that each tile can contain sub-tiles within the `children` array.

```json
{
    "boundingVolume": {
        "region": [...]
    },
    "refine": "REPLACE",
    "geometricError": 25,
    "content": {
        "uri": "tile.b3dm"
    },
    "children": [...]
}
```

Each tile is bound by a `boundingVolume`, which is a bounding box in 3D space. For our use case, tiles will replace their parent tile when they are loaded, and therefore we define `refine` as `REPLACE`. Geometric error of the tile is estimated based on the resolution of the tile's texture. Finally, the content of the tile is a link to the `b3dm` file represented by the tile, which is a glTF optimized for modern web browsers.

The `tiler.py` Python script will do most of the work for you when it comes to segmenting the OBJ file into 3D tiles and generating a conformant 3D tiles tileset JSON file. Given an input OBJ file, the `tiler.py` script will perform the following steps.

1. Read the OBJ mesh, create a voxel grid from the mesh, then use this grid to create an octree. See below for an example of how the octree segments the mesh into smaller cuboid volumes.
2. The script will traverse the octree and visit each cuboid. At each traversal, the script will approximately crop the mesh to the bounding volume of that particular cuboid and repack the texture image, keeping only texture data relevant to the remaining faces.
3. According to the depth of this cuboid within the octree, the script will downsample the PNG textures accordingly. Smaller bounding volumes that are deep within the octree will use *less* downsampling and lossy compression compared to larger bounding volumes that are closer to the root node of the octree. The larger the volume of the cuboid, the greater the compression.
4. Using the cropped mesh and compressed textures, the script will generate a `b3dm` file. This represents the data for that specific tile.
5. With the `b3dm` ready, the script generates the correct JSON for the tile, and places the tile within the tileset JSON at the correct location. The location is determined such that the parent tile contains the current tile.
6. After the entire octree has been traversed, the script will save the tileset JSON in the same directory as the `b3dm` files. The tileset is now ready to be used by a 3D tiles viewer.
7. If the `--debug-glb` and/or `--debug-tileset` flags are provided, a debug viewer will be output along with the tile set, and you should see instructions on how to load the viewer in your browser.

![Credit: Open3D Octree documentation](http://www.open3d.org/docs/latest/_images/tutorial_geometry_octree_5_3.png)

---

## References

* [3D Tiles Format Specification](https://github.com/CesiumGS/3d-tiles/tree/main/specification)
* [Open3D](http://www.open3d.org/docs/release/)
