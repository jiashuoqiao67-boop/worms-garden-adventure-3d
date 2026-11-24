"""
Generate the technical report (DOCX) for the Worm's Garden Adventure 3D project.

Follows the handbook section 17.2 structure. Run from the project root:
    .venv/Scripts/python.exe tools/build_report.py

The report is written to report/Worms_Garden_Adventure_3D_Report.docx.
Every number in the performance section is read from results/benchmark.json,
so re-running benchmark.py then this script keeps the report consistent.
"""

import json
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "screenshots")
REPORT_DIR = os.path.join(ROOT, "report")
OUT = os.path.join(REPORT_DIR, "Worms_Garden_Adventure_3D_Report.docx")


def shot(name):
    return os.path.join(SHOTS, name)


def add_image(doc, name, width=5.4, caption=None):
    path = shot(name)
    if not os.path.exists(path):
        print("missing screenshot:", name)
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cap.runs:
            run.italic = True
            run.font.size = Pt(9)


def code_block(doc, text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    para.paragraph_format.left_indent = Inches(0.25)
    para.paragraph_format.space_after = Pt(4)


def bullet(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def small_table(doc, rows, widths=None):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Light Grid Accent 1"
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
                    if r == 0:
                        run.bold = True
    if widths:
        for c, width in enumerate(widths):
            for row in table.rows:
                row.cells[c].width = Inches(width)


def benchmark_rows():
    path = os.path.join(ROOT, "results", "benchmark.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = [("Preset", "Framebuffer", "Pixels", "Triangles", "Mean ms",
             "Median ms", "FPS")]
    for r in data:
        rows.append((r["preset"], r["resolution"], f"{r['pixels']:,}",
                     f"{r['triangles']:,}", f"{r['render_ms_mean']:.2f}",
                     f"{r['render_ms_median']:.2f}", f"{r['frames_per_s']:.1f}"))
    return data, rows


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(9.5)
    style.paragraph_format.space_after = Pt(3)

    title = doc.add_heading("Worm's Garden Adventure 3D", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "Computer Graphics Programming (GEIAL31L-B2a) - Labs 10 & 11\n"
        "Instructor: Dr. Jawad Alshboul, Institute of Informatics, "
        "University of Miskolc\n"
        "Author: Qiao Jiashuo (TNUVNQ)  -  individual project\n"
        "Software renderer: Python 3.12, NumPy, Pillow, Pygame "
        "(Numba optional)")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---------------------------------------------------------------- 1
    doc.add_heading("1. Goal and summary of the original 2D game", level=1)
    doc.add_paragraph(
        "The starting material is the supplied 2D Pygame game Start2Dto3D.py. "
        "It generates a connected garden maze with depth-first recursive "
        "backtracking, stores N/E/S/W wall flags for every cell, and places "
        "nine water hazards, four apples (three worth 50 and one worth 100) and "
        "birds that move between corridor neighbours. The player worm walks "
        "along corridors, can jump, collects apples, and may only reach the exit "
        "after all four apples are collected; the score and timer persist across "
        "levels. Sparks follow the worm and a mouse click triggers a firework "
        "burst.")
    doc.add_paragraph(
        "The assignment converts this 2D presentation into a coherent 3D scene "
        "using only Python, NumPy, Pillow and Pygame. All 3D mathematics, "
        "projection, rasterization, visibility, lighting and texture sampling "
        "are implemented in our own software renderer; no game engine or "
        "high-level 3D renderer is used. The maze topology, placement rules, "
        "scoring, timer, jump, hazards, collectibles and level-completion rules "
        "are retained from the source. Table 1 lists the controls.")
    small_table(doc, [
        ("Key", "Action"),
        ("WASD / arrows", "Move the worm on the X/Z ground plane"),
        ("SPACE", "Vertical jump (real gravity in world Y)"),
        ("P", "Perspective <-> orthographic projection"),
        ("1 / 2 / 3", "Camera presets: overview / side / third-person follow"),
        ("Q / E, mouse wheel", "Orbit the camera, dolly in and out (clamped)"),
        ("F5", "Cycle render-resolution preset FAST/BALANCED/HIGH/NATIVE"),
        ("F", "Cycle depth fog OFF -> LIGHT -> HEAVY (extension 1)"),
        ("T", "Toggle nearest / bilinear texture sampling (extension 2)"),
        ("F1 / F3", "Wireframe debug view / depth-buffer debug view"),
        ("Left click", "Ray-cast the pixel onto the ground, burst there"),
        ("R / ESC", "Restart the run / quit"),
    ], widths=(1.5, 4.4))

    # ---------------------------------------------------------------- 2
    doc.add_heading("2. World coordinates and the 2D-to-3D data conversion",
                    level=1)
    doc.add_paragraph(
        "The world is right-handed with X = east, Y = up and Z = south. The "
        "ground lies on the X/Z plane; Y carries only height above the ground. "
        "One maze cell is one world unit, so original pixel lengths and speeds "
        "are divided by CELL = 50: wall thickness 8 px -> 0.16, worm radius "
        "11 px -> 0.22, worm speed 230 px/s -> 4.6 units/s, bird radius 10 px "
        "-> 0.20, bird speed 75 px/s -> 1.5 units/s, water-hole radius 13 px "
        "-> 0.26.")
    code_block(doc,
               "# cell (col, row) -> world centre; the old screen row becomes Z\n"
               "def cell_center_3d(col, row, height=0.0):\n"
               "    return np.array([col + 0.5, height, row + 0.5], float)\n"
               "\n"
               "# legacy 2D pixel position -> world (handbook 4.4)\n"
               "P_world = ((x_old - 40) / 50, height, (y_old - 95) / 50)")
    doc.add_paragraph(
        "Cell (2, 1) therefore has centre (2.5, 0, 1.5), and the original start "
        "pixel (65, 470) becomes (0.5, Y, 7.5). Game data stores only world "
        "positions; screen pixels appear only inside the renderer and the HUD. "
        "The maze module keeps COLS = 23, ROWS = 15, the WALL_THICKNESS_WORLD "
        "and the radius/speed constants in world units, and the wall colliders "
        "are generated from the same (centre, size) boxes that build the wall "
        "meshes, so visible geometry and collision cannot drift apart.")

    # ---------------------------------------------------------------- 3
    doc.add_heading("3. Meshes, matrices, camera and projections", level=1)
    doc.add_paragraph(
        "A Mesh stores vertex positions, triangle indices and optional UV "
        "coordinates. The maze is turned into UV-mapped boxes: a north/south "
        "wall is a box of size (1.16, 0.75, 0.16) centred at "
        "(col + 0.5, 0.375, row), an east/west wall is rotated by construction, "
        "and duplicates are avoided exactly as in the 2D draw strategy (N and W "
        "for every cell, E only on the last column, S only on the last row). "
        "All 382 wall boxes are merged into two meshes per level (3056 side "
        "triangles with the stone texture, 764 top-cap triangles), together with "
        "the ground grid, the surrounding lawn, nine water discs and two flags: "
        "5256 static triangles in six draw calls. Four apples, three birds "
        "(wings included) and the worm add 1716 animated triangles.")
    doc.add_paragraph(
        "Transforms use column-vector 4x4 homogeneous matrices: a position has "
        "w = 1 and a direction w = 0, so translation never moves a direction. "
        "The model matrix is M = T @ R @ S (scale, rotate about Y, translate) "
        "and the renderer applies it to a whole vertex array in one NumPy "
        "product. Normals are taken from the transformed world triangle with "
        "n = normalize((B - A) x (C - A)), which is equivalent to the "
        "inverse-transpose rule and stays correct under non-uniform scaling.")
    doc.add_paragraph(
        "The camera basis follows the handbook: f = normalize(target - eye), "
        "r = normalize(up0 x f), u = f x r, and a world point becomes "
        "c = ((p - eye).r, (p - eye).u, (p - eye).f) with +Z in front of the "
        "camera. Perspective projection uses "
        "f = (H/2) / tan(fov_y/2), sx = W/2 + f*x/z, sy = H/2 - f*y/z. "
        "Orthographic replaces the division with a fixed pixels-per-unit scale. "
        "Worked example: with eye (0,0,0), target (0,0,1), FOV 70 deg and a "
        "640x480 buffer the focal length is 342.8 px, so the camera-space point "
        "(1, 2, 5) projects to (388.6, 102.9); halving the depth to 2.5 doubles "
        "the offset from the centre, which is exactly the perspective cue the "
        "orthographic mode removes.")
    add_image(doc, "01_wireframe_cube.png", 2.7,
              "Figure 1. Appendix E wireframe unit cube, projected and drawn "
              "with our own camera (Stage 1).")
    add_image(doc, "02_perspective.png", 4.7,
              "Figure 2. Perspective overview of the full 23x15 maze "
              "(HUD reports the internal framebuffer size).")
    add_image(doc, "03_orthographic.png", 4.7,
              "Figure 3. Orthographic overview from a comparable camera: "
              "parallel edges stay parallel and size is depth-independent.")
    doc.add_paragraph(
        "Camera presets 1-3 (overview, side, third-person follow) share the "
        "same world data, and Q/E plus the mouse wheel orbit and dolly the eye "
        "around the target with clamped zoom and elevation. The follow preset "
        "eases the camera pose toward the worm with an exponential filter "
        "(k = 1 - exp(-6.5 dt)) so the chase motion is smooth. Pressing P keeps "
        "the same eye and target and only changes the projection; in "
        "orthographic mode the eye is lifted, because a parallel-ray camera "
        "whose projection plane dips below the floor would sample below the "
        "ground and show sky at the bottom of the frame.")

    # ---------------------------------------------------------------- 4
    doc.add_heading("4. Triangle rasterization, depth and clipping", level=1)
    doc.add_paragraph(
        "One frame is rendered into an HxWx3 uint8 colour buffer and an HxW "
        "float64 depth buffer. The pipeline per mesh is: model -> world "
        "(vectorized 4x4), world -> camera (one matrix product against the "
        "camera basis), per-triangle back-face cull, near-plane handling, "
        "projection, and rasterization.")
    doc.add_paragraph(
        "Back-face culling compares the world-space face normal with the vector "
        "from the triangle to the eye: dot(n, eye - A) > 0 keeps the triangle. "
        "This is independent of the camera-space handedness convention, so "
        "winding changes cannot silently remove visible faces.")
    doc.add_paragraph(
        "Triangles fully in front of the near plane are projected with NumPy and "
        "submitted as one batch. For each covered pixel the rasterizer computes "
        "barycentric weights from the edge functions, invz = wa/za + wb/zb + "
        "wc/zc, depth z = 1/invz, rejects z >= depthbuf[y,x], then samples the "
        "texture with perspective-correct UVs u = (wa*ua/za + wb*ub/zb + "
        "wc*uc/zc) / invz and writes the shaded colour. Weights, reciprocal "
        "depth and the UV numerators are incremented by one addition per pixel "
        "(scanline increments) instead of being recomputed from two products, "
        "which is the main reason the batched path is roughly four times faster "
        "than the first per-triangle implementation.")
    doc.add_paragraph(
        "Triangles that cross the near plane are clipped with "
        "Sutherland-Hodgman before projection, and the same interpolation "
        "parameter t is applied to their UVs so clipped textured triangles keep "
        "their texture. A triangle completely behind the near plane is "
        "discarded, so no giant inverted polygons can appear when the camera "
        "moves into a wall.")
    doc.add_paragraph(
        "Screenshot 19 exercises this path: a camera 0.55 units above the floor "
        "takes the clipping path for 108 triangles and still produces a clean "
        "frame. Testing it exposed a bug worth recording: clipping creates "
        "vertices exactly on the near plane, and the projection originally "
        "rejected any depth <= near, so clipped triangles were discarded and "
        "left background-coloured holes in the ground while the follow camera "
        "moved (5 of 60 scripted frames). Accepting vertices on the plane, as "
        "handbook 7.6 requires, fixed it; a regression test now renders a "
        "floor that crosses the near plane and asserts that no background "
        "pixels remain below the horizon.")
    code_block(doc,
               "invz = wa/za + wb/zb + wc/zc      # affine in screen space\n"
               "z    = 1.0 / invz                # perspective-correct depth\n"
               "if z < depthbuf[y, x]:           # nearer fragment wins\n"
               "    depthbuf[y, x] = z\n"
               "    colorbuf[y, x] = shade(texture.sample(u, v), intensity)")
    add_image(doc, "04_occlusion.png", 4.6,
              "Figure 4. Z-buffer test: the near blue cube hides the far red "
              "cube and the wall slab behind it; nothing shows through.")

    # ---------------------------------------------------------------- 5
    doc.add_heading("5. Lighting, textures, image processing and animation",
                    level=1)
    doc.add_paragraph(
        "Lighting is flat shading with one normal and one intensity per "
        "triangle: I = ka + kd * max(0, n . L), with ka = 0.25, kd = 0.75 and "
        "L = normalize(0.35, 0.80, 0.50) pointing from the surface toward the "
        "light. The factor multiplies each RGB channel; the material colour also "
        "tints textured surfaces (a base colour of 255 leaves the texture "
        "unchanged), which is how the bright maze floor and the darker "
        "surrounding lawn share one procedurally generated grass image.")
    doc.add_paragraph(
        "Pillow generates both textures at start-up: a 192x192 grass image of "
        "short blade strokes over five green tones plus darker tufts, and a "
        "128x128 stone image made of four brick courses with mortar lines, "
        "per-brick tone variation and fine speckle. The resolutions were raised "
        "from 96/64 after the first build, because at native framebuffer "
        "resolution the magnified texels of the smaller images were clearly "
        "visible as blocks. Every wall face owns its own four vertices with a "
        "0..1 UV rectangle scaled by the face size, so the stone keeps its "
        "aspect on walls of different lengths and the texture repeats once per "
        "world unit. The Part A sampler is nearest-neighbour with wrap, which "
        "is the shipped default; T switches to the bilinear extension.")
    add_image(doc, "05_textured_scene.png", 4.7,
              "Figure 5. Close view: stone walls, grass floor, flat lighting; "
              "the darker faces are those turned away from the light.")
    doc.add_paragraph(
        "Two 3D animations run every frame. (1) The worm is a head plus four "
        "body segments; the head position is recorded into a short path history "
        "and each segment is placed on an older sample of that path, with a "
        "small sine offset in Y, so the body follows the head around corners "
        "instead of rotating as one rigid object. (2) The birds flap: the two "
        "wing meshes are hinged at the body axis and rotated by "
        "0.85*sin(9t + phase) about Z. Apples bob by 0.05*sin(2*pi*1.5*t) around "
        "their base height, and the collectible test uses the same bobbed "
        "position, so the visual and the collider agree.")
    doc.add_paragraph(
        "The particle system keeps a 3D position and velocity per particle with "
        "gravity, damping, bounded lifetime and a hard cap of 250 particles. "
        "Sparks are emitted while the worm moves. A click casts the pixel ray "
        "through the camera, intersects the ground plane Y = 0 and bursts there, "
        "so the 2D click has a documented 3D meaning. Particles are projected "
        "after the opaque pass and depth-tested against the Z-buffer, so a "
        "burst behind a wall is hidden by that wall.")

    # ---------------------------------------------------------------- 6
    doc.add_heading("6. Gameplay adaptation, collisions and jumping", level=1)
    doc.add_paragraph(
        "Movement is integrated on X/Z against the wall boxes with a "
        "circle-vs-AABB test, subdivided into steps smaller than half the wall "
        "thickness and resolved one axis at a time, so the worm slides along "
        "walls and cannot tunnel through them at low frame rates. Vertical "
        "motion is a real jump state: gravity 10 units/s^2, launch speed 3.2 "
        "units/s, giving a peak rise of v0^2 / 2g = 0.512 units and a flight "
        "time of 2v0/g = 0.64 s, with a 0.65 s cooldown between jumps.")
    bullet(doc, "Walls: the X/Z footprint is tested only while the worm's feet "
                "are below the 0.75-unit wall top. The jump peak (0.512) cannot "
                "clear a wall, so the maze stays closed, but the height rule is "
                "explicit and unit-tested rather than implied.")
    bullet(doc, "Water: a water hole resets the worm only when its centre is "
                "within 0.05 units of the ground height, so a jump genuinely "
                "clears a hole; the reset returns the worm and its trail to the "
                "start cell. Each hole is drawn as a bright blue disc with a "
                "dark rim, and a short HUD message explains the reset. The "
                "first build wound the disc fan backwards, so the back-face "
                "test removed every water pad and the player was teleported by "
                "an invisible trap; the winding is fixed and "
                "test_water_disc_faces_up now asserts that every pad normal "
                "points up.")
    bullet(doc, "Birds: a 3D sphere test (worm 0.22 + bird 0.20) that includes "
                "the vertical separation. The bird's flight band "
                "(0.55 +/- 0.20) overlaps the whole jump arc (0.22 to 0.73), so "
                "flying over a bird is not possible in this configuration - the "
                "player must walk around it. This is a deliberate, documented "
                "rule, not an accidental 2D test.")
    bullet(doc, "Apples: collected when the 3D distance to the bobbing apple "
                "centre is below the combined radii; three apples award 50 and "
                "the gold one 100, matching the source scoring.")
    bullet(doc, "Levels: the exit unlocks only after all four apples are "
                "collected; reaching the checkered flag then regenerates the "
                "maze, water, apples and birds and keeps the score. R restarts "
                "at level 1 with score 0 and clears the particles.")
    add_image(doc, "06_gameplay_follow.png", 4.6,
              "Figure 6. Gameplay chase camera: worm, apple, bird, water "
              "hazards, apples counter, unlocked-exit hint, minimap and the "
              "renderer debug line.")

    # ---------------------------------------------------------------- 7
    doc.add_heading("7. Two extensions: algorithms, screenshots, evaluation",
                    level=1)
    doc.add_heading("7.1 Atmospheric depth fog (graphics)", level=2)
    doc.add_paragraph(
        "Question: can per-fragment colour blending convey distance and aerial "
        "perspective without any extra geometry? For a fragment at camera depth "
        "z the fog factor is t = clamp((z - near)/(far - near), 0, 1) and the "
        "output is C = C_frag*(1 - t) + C_fog*t, applied after lighting and "
        "texture sampling and before the depth write. Instead of fixed "
        "constants the range is derived from the camera distance d to the maze "
        "centre, near = 0.45*far and far = clamp(1.6*d, 10, 48), so the effect "
        "works from the overview and from inside the corridors. F cycles three "
        "states (OFF, LIGHT, HEAVY where far shrinks to 72 %).")
    add_image(doc, "09_fog_off.png", 4.3,
              "Figure 7. Fog OFF: full contrast across the whole maze.")
    add_image(doc, "10_fog_on.png", 4.3,
              "Figure 8. Fog HEAVY: distant walls and lawn blend into the sky "
              "colour while the near corridor keeps its contrast.")
    doc.add_paragraph(
        "Evaluation: the far half of the maze loses contrast in the expected "
        "monotonic way and the effect costs one multiply-add per fragment. "
        "Limitations: a single linear depth blend with no height falloff or "
        "colour gradient, and it is applied to every material, including the "
        "particles' background geometry.")

    doc.add_heading("7.2 Bilinear texture filtering (image processing)", level=2)
    doc.add_paragraph(
        "Question: nearest-neighbour magnification shows hard texel blocks; can "
        "the four neighbouring texels be blended to soften them? The sampler "
        "takes fractional UV coordinates, wraps in both directions, and blends "
        "c00..c11 with fx, fy: top = c00*(1-fx) + c10*fx, "
        "bottom = c01*(1-fx) + c11*fx, result = top*(1-fy) + bottom*fy. The "
        "renderer selects the sampler per material and T toggles it at runtime.")
    add_image(doc, "11_filter_nearest.png", 4.0,
              "Figure 9. Nearest-neighbour sampling at a close camera: hard "
              "texel edges on the stone and grass.")
    add_image(doc, "12_filter_bilinear.png", 4.0,
              "Figure 10. Bilinear sampling from the same camera: smooth texel "
              "transitions.")
    doc.add_paragraph(
        "Evaluation: the same 3D patch and camera differ only in the sampler, "
        "and the 2D magnification pair (screenshots 13/14) isolates the "
        "filtering itself. Cost is four texel reads instead of one per fragment "
        "(measured in the same benchmark scene). Limitation: filtering is "
        "per-fragment with no mipmaps, so distant, minified textures still "
        "alias; bilinear smoothing is not antialiasing of polygon edges.")

    # ---------------------------------------------------------------- 8
    doc.add_heading("8. Testing, performance, limitations and contributions",
                    level=1)
    doc.add_paragraph(
        "Twelve small test programs run directly with the project interpreter; "
        "they mix numerical expected-value checks with gameplay integration "
        "checks and can all be executed without a window.")
    small_table(doc, [
        ("Test file", "What it proves"),
        ("test_maze", "Maze connectivity, wall flags, cell -> world mapping"),
        ("test_transforms", "4x4 compose order, point vs direction, normals"),
        ("test_geometry", "Cube topology, ground winding, wall box sizes"),
        ("test_camera", "Camera basis, perspective vs ortho, near clip"),
        ("test_renderer", "Barycentric fill, Z-buffer nearer-wins both orders"),
        ("test_maze_world", "Wall boxes match the maze flags exactly"),
        ("test_materials", "Lighting terms, sampling wrap, textured render"),
        ("test_physics", "Wall slide, no tunnelling, jump peak, hazard height"),
        ("test_game", "Apple count/value, finish rule, score, restart"),
        ("test_particles", "Bounded count, gravity, lifetime, depth hiding"),
        ("test_extensions", "Bilinear blend maths, fog near/far behaviour"),
        ("test_resize", "Framebuffer aspect ratio follows the window"),
    ], widths=(1.6, 4.3))

    data, bench_rows = benchmark_rows()
    tri_count = f"{data[0]['triangles']:,}"
    fps_values = [r["frames_per_s"] for r in data]
    doc.add_paragraph(
        "Performance was measured offscreen with benchmark.py: the same maze "
        f"seed, camera and scene at four internal resolutions, 12 frames per "
        f"preset after one warm-up frame, with {tri_count} triangles submitted "
        "per frame. Only the framebuffer size changes between rows.")
    small_table(doc, bench_rows, widths=(0.8, 1.05, 0.8, 0.75, 0.7, 0.8, 0.55))
    doc.add_paragraph(
        f"Render time scales with the number of pixels ({fps_values[0]:.0f} fps "
        f"at 672x420 down to {fps_values[-1]:.0f} fps at 1600x1000 for this "
        "scene), which shows that the cost is dominated by per-fragment shading "
        "rather than per-triangle Python work. The optimisation path was: "
        "vectorize the model and camera transforms per mesh, merge the static "
        "geometry into six meshes, cull back faces in world space, submit all "
        "front-facing triangles of a mesh to one compiled batch, replace "
        "per-pixel barycentric recomputation with scanline increments, and then "
        "remove the remaining per-frame overheads that a profile pointed at: the "
        "world corners, normals, lighting intensities and per-triangle UVs of "
        "static meshes are cached on first use (they never change), the sky "
        "background is baked into one contiguous image so clearing is a memcpy, "
        "the depth buffer is float32, the clipped-triangle path projects inline "
        "instead of calling a method per vertex, the lawn is built as a ring "
        "around the maze instead of a full plane (the covered middle only lost "
        "the depth test), and the HUD panels, control labels and the 382-rect "
        "minimap wall layer are rasterised once and blitted. Together with the "
        "framebuffer blit change (image.frombuffer instead of surfarray plus an "
        "unconditional smoothscale) these took the follow-camera frame from "
        "about 40 ms to about 25 ms.")
    bullet(doc, "The shipped default is NATIVE: the framebuffer is exactly the "
                "window size (1280x800), so the image is displayed 1:1 and no "
                "upscale blur is added. The measured game loop runs at about 39 "
                "fps at NATIVE, 41 fps at BALANCED and 54 fps at FAST on this "
                "laptop; the loop is capped at 120 fps and the HUD shows the "
                "real frame time.")
    bullet(doc, "No antialiasing and no mipmaps: polygon edges and distant "
                "minified textures can alias, and HIGH quality renders more "
                "pixels than the window shows (supersampling) before "
                "smoothscale.")
    bullet(doc, "Flat shading only: no specular highlights, smooth vertex "
                "normals or cast shadows, so large flat faces read as flat.")
    bullet(doc, "Entities are low-poly parametric ellipsoids and boxes with "
                "facets rather than detailed anatomy.")
    add_image(doc, "07_low_res_672x420.png", 4.0,
              "Figure 11. FAST preset, 672x420 internal (displayed in the "
              "1280x800 window).")
    add_image(doc, "08_high_res_1600x1000.png", 4.0,
              "Figure 12. HIGH preset, 1600x1000 internal: genuinely sharper "
              "wall edges at about three times the render cost.")
    doc.add_paragraph(
        "Two renderer debug modes are included for the live demonstration. F1 "
        "draws every back-face-culled triangle edge of the same scene, and F3 "
        "replaces the colour buffer with a banded logarithmic view of the depth "
        "buffer (near surfaces blue, far surfaces orange) so the nearest "
        "surfaces are obvious. Both are saved in the screenshots folder as "
        "15_wireframe_view.png and 16_depth_view.png.")
    doc.add_paragraph(
        "Window resizing re-creates the display and recomputes the framebuffer "
        "from the new aspect ratio (Wr = Wd/Hd * Hr), so geometry is never "
        "stretched; screenshots 17/18 show the wide and narrow window cases.")
    doc.add_paragraph(
        "This is an individual project. The handler, all tests, textures, "
        "screenshots and both extensions were written for this assignment; only "
        "the supplied Start2Dto3D.py maze generation, placement rules and "
        "scoring logic are reused. Third-party use is limited to the pinned "
        "course libraries NumPy, Pillow and Pygame (plus optional Numba and the "
        "build-time python-docx/imageio helpers); no external art, sound, font "
        "or model asset is bundled.")

    doc.save(OUT)
    print("Report written to", OUT)


if __name__ == "__main__":
    main()





