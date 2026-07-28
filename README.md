# Sky Map to 3D Model Converter

[中文](./README-zh.md) | [English](./README.md)

A complete Python tool to convert Sky: Children of the Light map data (`.meshes` + `Objects.level.bin`) into 3D models (GLB/OBJ) with marker points as colored spheres.

## Features

- Parses all `.mesh` versions (0x17–0x20) and `.meshes` (LVL0) files.
- Extracts terrain, cloud meshes, and object instances from `Objects.level.bin` using a built-in TGCL parser.
- Merges all geometry into a single GLB or OBJ model with proper transforms.
- Automatically colors terrain vertices using the game's material palette (grass, rock, snow, sand, etc.).
- Exports marker points (CandleObject, WingBuff, Pickup, NPC, Quest markers, etc.) as small colored spheres and also saves their coordinates to a JSON file.
- Supports batch conversion of multiple maps.

## Dependencies

Install required packages:

```bash
pip install lz4 trimesh numpy
```

## Usage

Run the script and follow the interactive prompts:

```bash
python sky_map_converter.py
```

You can choose:
- **Single map mode**: process one map folder containing `.meshes` and `Objects.level.bin`.
- **Batch mode**: process all subfolders that contain a `.meshes` file within a parent directory.

You'll also need to specify:
- The **mesh root directory** – the folder where all `.mesh` asset files are stored (the script will search recursively).
- The **output root directory** – models and JSON files will be saved in subdirectories named after each map folder.

## Output

For each map, the tool produces:
- A merged 3D model (`<map_name>.glb` or `.obj`) containing:
  - Terrain (with vertex colors)
  - Cloud meshes
  - All placed objects (applied with their transform matrices)
  - Marker points as small colored spheres (if enabled)
- A JSON file (`<map_name>_markers.json`) with the coordinates, type, label, and color of every marker point.

## Supported Versions

- `.mesh`: versions 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20
- `.meshes`: LVL0 (all known versions)

## Notes

- The tool uses a built-in TGCL parser (transplanted from `bintojson.py`) to read `Objects.level.bin`.
- All geometry is merged into a single scene; each original mesh is kept as a separate node.
- Marker point spheres have a fixed radius of 0.5 units and 8 segments; you can adjust these values inside the script.
- If you only need the terrain model, you can disable marker export when prompted.

## License

This tool is provided as-is for educational and archival purposes.
