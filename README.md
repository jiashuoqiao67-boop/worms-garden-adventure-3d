# Worm's Garden Adventure 3D

Computer Graphics Programming (GEIAL31L-B2a) — Labs 10 & 11
A guided 2D-to-3D conversion of the supplied Pygame game, implemented with
Python, NumPy, Pillow and Pygame. The 3D pipeline (mathematics, projection,
rasterization, visibility, lighting, texture sampling) is our own software
renderer; no game engine or high-level 3D library is used.

## Environment (versions tested)

- Python: 3.12.0 (project virtual environment `.venv`)
- NumPy: 2.5.5
- Pillow: 13.0.0
- Pygame: 2.6.1

## Install

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The original supplied 2D reference (kept for comparison) is in
`reference/Start2Dto3D.py`.

## Controls

| Key | Effect |
|---|---|
| WASD / Arrows | Move worm on the X/Z ground plane |
| SPACE | Vertical jump (gravity acts on world Y) |
| P | Toggle perspective / orthographic projection |
| 1 / 2 / 3 | Camera preset: elevated overview / side / follow |
| F5 | Cycle render-resolution preset (FAST / BALANCED / HIGH / NATIVE) |
| F | Toggle atmospheric depth fog (Part B extension 1) |
| T | Toggle nearest / bilinear texture filtering (Part B extension 2) |
| R | Restart the current run (score and level reset) |
| Left mouse click | Firework particle burst |
| Window close / ESC | Quit |

The HUD shows the active projection, camera view, internal framebuffer size,
render time, triangle count, fog state and texture filter.

## Project layout

| File | Responsibility |
|---|---|
| `main.py` | Pygame loop, events, input, frame timing, HUD |
| `game.py` | Levels, apples, birds, score, timer, finish rules |
| `maze.py` | Procedural maze generation, neighbours, coordinate conversion |
| `geometry.py` | Cube / wall / ground meshes and UVs |
| `transforms.py` | 4×4 matrices and vector utilities |
| `camera.py` | Camera basis, perspective/orthographic projection, near clip |
| `renderer.py` | Triangle rasterization, Z-buffer, lighting, textures, fog |
| `materials.py` | Materials, procedural textures, nearest/bilinear sampling |
| `physics.py` | Movement, wall/water/bird/apple collisions, jump |
| `particles.py` | 3D spark and firework particle system |
| `first_cube.py` | Appendix E wireframe unit-cube orbit demo (Stage 1) |
| `benchmark.py` | Offscreen performance benchmark across quality presets |
| `make_screenshots.py` | Offscreen generator for the handbook evidence screenshots |
| `tools/build_report.py` | Rebuilds the technical report DOCX (optional; needs `python-docx`) |
| `tools/make_demo_video.py` | Renders the demo MP4 offscreen (optional; needs `imageio`, `imageio-ffmpeg`) |
| `tests/` | Small numerical and gameplay checks (run each file directly) |
| `screenshots/` | Labeled evidence images (projections, occlusion, quality, extensions) |
| `results/` | Measured benchmark table (`benchmark.txt` / `.json`) |
| `reference/` | Original supplied 2D file, kept for comparison |

## Tests and measurements

Run every check from the project root with the venv interpreter:

```bash
.venv\Scripts\python.exe tests\test_maze.py
.venv\Scripts\python.exe tests\test_transforms.py
.venv\Scripts\python.exe tests\test_geometry.py
.venv\Scripts\python.exe tests\test_camera.py
.venv\Scripts\python.exe tests\test_renderer.py
.venv\Scripts\python.exe tests\test_maze_world.py
.venv\Scripts\python.exe tests\test_materials.py
.venv\Scripts\python.exe tests\test_physics.py
.venv\Scripts\python.exe tests\test_game.py
.venv\Scripts\python.exe tests\test_particles.py
.venv\Scripts\python.exe tests\test_extensions.py
.venv\Scripts\python.exe tests\test_resize.py
```

Performance benchmark (writes `results/benchmark.txt` and `benchmark.json`):

```bash
python benchmark.py
```

Evidence screenshots (writes `screenshots/`):

```bash
python make_screenshots.py
```

## Extensions (Part B)

### 1. Atmospheric depth fog (graphics)

Distant fragments are blended toward a fixed fog colour according to their
camera-space depth. For a fragment at depth `z`, the fog factor is
`t = clamp((z - near) / (far - near), 0, 1)`, and the final colour is
`C = C_frag·(1 - t) + C_fog·t`. Implemented per fragment in the software
renderer; toggle with `F`. Before/after: `screenshots/09_fog_off.png` and
`screenshots/10_fog_on.png`.

### 2. Bilinear texture filtering (image processing)

The default Part A sampler is nearest-neighbour lookup. The extension adds
bilinear interpolation between the four neighbouring texels at fractional
UV coordinates, so magnified textures soften instead of showing hard blocks.
Toggle with `T`. 3D comparison: `screenshots/11_filter_nearest.png` vs
`12_filter_bilinear.png`; pure 2D magnification comparison:
`13_tex_nearest_zoom.png` vs `14_tex_bilinear_zoom.png`.

## External code / assets

- The starting 2D game `reference/Start2Dto3D.py` is the file supplied by the
  course instructor; its maze generation, placement rules and scoring logic are
  reused and adapted to 3D.
- All textures (`grass`, `stone`) are generated procedurally at runtime with
  Pillow; no external image, sound, font or model asset is bundled.
- Runtime dependencies are listed in `requirements.txt`: NumPy (numerical
  kernels), Pillow (image/texture work), Pygame (input, timing, window,
  display, HUD).
- The optional report/video builder scripts additionally use `python-docx` and
  `imageio`/`imageio-ffmpeg`; these are build-time helpers for regenerating
  the DOCX and MP4 deliverables, not requirements to run the game.
- No game engine, GPU renderer, or third-party 3D mesh library is used.
