# Worm's Garden Adventure 3D

Computer Graphics Programming (GEIAL31L-B2a) — Labs 10 & 11
A guided 2D-to-3D conversion of the supplied Pygame game, implemented with
Python, NumPy, Pillow and Pygame (pure software renderer; no game engine).

## Environment (record the exact versions you tested on)

- Python: 3.13.18
- NumPy: 2.5.4
- Pillow: 13.0.0
- Pygame: 2.6.1

## Install

```bash
python -m pip install numpy pillow pygame
```

## Run

```bash
python main.py
```

## Controls

| Key | Effect |
|---|---|
| WASD / Arrows | Move worm on the X/Z ground plane |
| SPACE | Vertical jump (gravity in world Y) |
| P | Toggle perspective / orthographic projection |
| 1 / 2 / 3 | Camera preset: elevated / side / follow |
| Mouse wheel / Q / E | Zoom / orbit camera |
| R | Restart current run |
| F1 / F3 (optional) | Wireframe / depth-buffer debug views |
| ESC | Quit |

(Only document the controls you actually implement.)

## Project layout

| File | Responsibility |
|---|---|
| `main.py` | Pygame loop, events, input, frame timing |
| `game.py` | Levels, apples, birds, score, timer, rules |
| `maze.py` | Procedural maze generation and neighbours |
| `geometry.py` | Cube / wall / ground meshes |
| `transforms.py` | 4×4 matrices and vector utilities |
| `camera.py` | Camera basis, perspective/orthographic projection |
| `renderer.py` | Triangle rasterization, Z-buffer, lighting, textures |
| `physics.py` | Movement, collisions, jump |
| `particles.py` | 3D particles |
| `assets/` | Procedural textures and materials |
| `tests/` | Small numerical and gameplay checks |
| `reference/` | Original supplied 2D file, kept for comparison |

## Extensions (Part B)

_(Fill in after implementing: feature name, principle, inputs/outputs, before/after evidence.)_

## External code / assets

_(List any third-party code or assets used, with links and licenses.)_
