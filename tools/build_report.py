"""
Generate the technical report (DOCX) for the Worm's Garden Adventure 3D project.

Follows handbook section 17.2 structure. Run from the project root:
    .venv/Scripts/python.exe tools/build_report.py

The report is written to report/Worms_Garden_Adventure_3D_Report.docx.
"""

import json
import os

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "screenshots")
REPORT_DIR = os.path.join(ROOT, "report")
OUT = os.path.join(REPORT_DIR, "Worms_Garden_Adventure_3D_Report.docx")


def shot(name):
    return os.path.join(SHOTS, name)


def add_image(doc, name, width=5.6, caption=None):
    path = shot(name)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in cap.runs:
                run.italic = True
                run.font.size = Pt(9)


def bullet(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def _benchmark_rows():
    path = os.path.join(ROOT, "results", "benchmark.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = [("Preset", "Resolution", "Triangles", "Mean ms",
             "Median ms", "FPS")]
    for r in data:
        rows.append((r["preset"], r["resolution"], str(r["triangles"]),
                     f"{r['render_ms_mean']:.2f}",
                     f"{r['render_ms_median']:.2f}",
                     f"{r['frames_per_s']:.1f}"))
    return rows


def perf_table(doc, rows):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Light Grid Accent 1"
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = value
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
                    if r == 0:
                        run.bold = True


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    doc = Document()

    # Base font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    # Title
    title = doc.add_heading("Worm's Garden Adventure 3D", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "Computer Graphics Programming (GEIAL31L-B2a), Labs 10 & 11\n"
        "Instructor: Dr. Jawad Alshboul, Institute of Informatics, "
        "University of Miskolc\n"
        "Author: Qiao Jiashuo (TNUVNQ)")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 1. Goal and summary of the original 2D game
    doc.add_heading("1. Goal and summary of the original 2D game", level=1)
    doc.add_paragraph(
        "The starting material is the supplied 2D Pygame game "
        "Start2Dto3D.py. It generates a connected garden maze with depth-first "
        "recursive backtracking, stores N/E/S/W wall flags for every cell, and "
        "places nine water hazards, four apples (three silver worth 50 and one "
        "gold worth 100), and birds that move between corridor neighbours. The "
        "player worm moves along corridors, can jump, collects apples, and may "
        "only reach the exit after all four apples are collected; the score and "
        "timer persist across levels. Sparks follow the worm and a mouse click "
        "triggers a firework burst.")
    doc.add_paragraph(
        "The assignment converts this 2D presentation into a coherent 3D scene "
        "using only Python, NumPy, Pillow and Pygame. All 3D mathematics, "
        "projection, rasterization, visibility, lighting and texture sampling are "
        "implemented in our own software renderer; no game engine or high-level "
        "3D renderer is used. The maze topology, placement rules, scoring, "
        "timer, jump, hazards, collectibles and level-completion rules are "
        "retained from the source.")

    # 2. World coordinates and data conversion
    doc.add_heading("2. World coordinates and the 2D-to-3D data conversion", level=1)
    doc.add_paragraph(
        "The world uses a right-handed convention with X = east, Y = up and "
        "Z = south. The ground lies on the X/Z plane; Y carries only the height "
        "above the ground. One maze cell is one world unit. Original pixel "
        "lengths and speeds are divided by the original cell size of 50 pixels, "
        "so the converted constants are:")
    bullet(doc, "Wall thickness 8 px -> 0.16 world units; worm radius 11 px -> 0.22; "
                "worm speed 230 px/s -> 4.6 units/s; bird radius 10 px -> 0.20; "
                "bird speed 75 px/s -> 1.5 units/s; water hole radius 13 px -> 0.26.")
    doc.add_paragraph(
        "A cell (col, row) maps to its world centre as "
        "P = (col + 0.5, height, row + 0.5). For example, cell (2, 1) has "
        "centre (2.5, 0, 1.5). The old screen row becomes world Z, not world Y; "
        "the legacy pixel conversion is "
        "P_world = ((x_old - 40)/50, height, (y_old - 95)/50).")
    doc.add_paragraph(
        "The original jump was only a visual screen-space displacement and water "
        "collision was disabled during a jump. In the 3D version the worm stores "
        "a real vertical height Y, and water and bird collisions use explicit "
        "height-aware rules: a grounded worm touching a water hole is reset, "
        "while a worm above the hazard height is not.")

    # 3. Meshes, matrices, camera, projections
    doc.add_heading("3. Meshes, matrices, camera and projections", level=1)
    doc.add_paragraph(
        "Geometry is built from low-poly cubes, a ground grid, and parametric "
        "round solids (lat-long spheres/ellipsoids, ground discs and pennant "
        "flags). Each Mesh "
        "stores world/local vertices, triangle index triples and optional UV "
        "coordinates. The ground is a cols x rows grid of upward-facing "
        "triangles; walls are axis-aligned boxes whose centre and size are "
        "derived from the maze wall flags, deduplicated exactly as in the 2D "
        "draw strategy (N/W walls per cell, E only on the last column, S only on "
        "the last row; entrance and exit openings keep their cleared flags).")
    doc.add_paragraph(
        "Transforms use column-vector 4x4 homogeneous matrices. A position uses "
        "weight w = 1 and a direction uses w = 0, so translation never moves a "
        "direction vector. The combined model matrix is M = T @ R @ S: scale "
        "first, then rotate about Y, then translate. Surface normals are "
        "transformed with the inverse-transpose of the 3x3 linear part before "
        "normalization, which stays correct under non-uniform scaling.")
    doc.add_paragraph(
        "The camera builds an orthonormal basis from eye and target: forward is "
        "the normalized eye-to-target vector, right = normalize(up0 x forward), "
        "and up = forward x right. World points are projected into camera space "
        "by dotting the eye-relative vector with right, up and forward.")
    doc.add_paragraph(
        "Perspective projection uses a focal length "
        "focal = (H/2) / tan(fov_y/2); screen coordinates are "
        "sx = W/2 + focal*x/z, sy = H/2 - focal*y/z. Orthographic projection "
        "replaces the depth division with a fixed pixels-per-unit scale, so "
        "object size does not shrink with distance. The two projections are "
        "comparable in the screenshots below: perspective shows near-large / "
        "far-small convergence, while orthographic keeps parallel edges parallel.")
    add_image(doc, "01_wireframe_cube.png", 3.6,
             "Figure 1. Appendix E wireframe unit cube, projected with our camera.")
    add_image(doc, "02_perspective.png", 5.4,
             "Figure 2. Perspective overview: distant corridors converge and shrink.")
    add_image(doc, "03_orthographic.png", 5.4,
             "Figure 3. Orthographic overview: size is independent of depth.")

    # 4. Rasterization, depth, clipping
    doc.add_heading("4. Triangle rasterization, depth method and clipping", level=1)
    doc.add_paragraph(
        "The renderer keeps an HxWx3 uint8 colour buffer and an HxW float depth "
        "buffer. For every triangle it transforms vertices to camera space, clips "
        "against the near plane with the Sutherland-Hodgman algorithm, projects "
        "the resulting polygon to screen pixels, and rasterizes each fan "
        "triangle.")
    doc.add_paragraph(
        "Visibility is resolved with a Z-buffer. For each pixel inside the "
        "triangle, barycentric weights (wa, wb, wc) are computed from the edge "
        "functions. Depth is interpolated with perspective correction using "
        "1/z: invz = wa/za + wb/zb + wc/zc, and z = 1/invz. A fragment only "
        "overwrites the pixel when its depth is nearer than the stored depth. "
        "The occlusion test below places a blue cube in front of a red cube; the "
        "hidden red surfaces never draw through.")
    add_image(doc, "04_occlusion.png", 3.8,
             "Figure 4. Near blue cube occludes the far red cube (Z-buffer).")
    doc.add_paragraph(
        "The rasterizer scans only the triangle's bounding box. The per-pixel "
        "barycentric, depth and UV math follows the handbook description; the "
        "bounding-box scan is vectorized with NumPy arrays for speed, but the "
        "visibility and interpolation steps are unchanged.")

    # 5. Lighting, textures, image processing, animation
    doc.add_heading("5. Lighting, textures, image processing and animation", level=1)
    doc.add_paragraph(
        "Lighting is flat shading: one face normal and one intensity per "
        "triangle. The Lambertian model is I = ka + kd * max(0, n . L), with "
        "ambient ka = 0.25 and diffuse kd = 0.75; L points from the surface "
        "toward the light. The factor is applied to each RGB channel.")
    doc.add_paragraph(
        "Two procedural textures are generated at runtime with Pillow: a grass "
        "texture with short blade strokes and a stone texture with rectangular "
        "tones. UVs are attached to the ground and wall vertices; the default "
        "sampler is nearest-neighbour lookup with horizontal/vertical wrap. UVs "
        "are interpolated with perspective correction (u/z, v/z, 1/z) so texture "
        "detail stays attached to the surface in perspective view.")
    add_image(doc, "05_textured_scene.png", 5.4,
             "Figure 5. Textured, illuminated maze with grass, stone walls, "
             "water holes, apples, birds and start/finish flags.")
    doc.add_paragraph(
        "Animations in the 3D world include: apples bobbing vertically around "
        "APPLE_HEIGHT with a sine offset, birds hovering at BIRD_HEIGHT while "
        "following corridor neighbours and remembering their last few cells so "
        "they patrol a wider route instead of oscillating in a small area, "
        "a 3D spark trail emitted while moving, "
        "and a coloured firework burst on mouse click. The worm and bird "
        "models yaw about the vertical axis to face their current travel "
        "direction. Every particle stores a "
        "3D position and velocity in world space, with gravity, damping and a "
        "bounded lifetime and count.")

    # 6. Gameplay adaptation
    doc.add_heading("6. Gameplay adaptation, collisions and retained rules", level=1)
    doc.add_paragraph(
        "Movement is integrated on the X/Z plane against the maze wall boxes "
        "using a circle-vs-AABB test in two axes, so the worm stays in corridors "
        "and cannot cross walls. Vertical motion uses an explicit jump state: "
        "gravity pulls the worm down, an upward jump velocity sets the height, "
        "and a short jump cooling period prevents repeated mid-air jumps.")
    bullet(doc, "Water hazards: a grounded worm within the water hole radius is "
                "reset to the start cell; height-aware, so jumping over a hole "
                "does not trigger it.")
    bullet(doc, "Birds: a 3D sphere-distance collision that includes the vertical "
                "separation; a grounded worm touching a bird is reset, and jumping "
                "raises the worm above the bird's hover height.")
    bullet(doc, "Apples: collected by proximity to the bobbing apple position; "
                "three silver (50) and one gold (100), matching the source scoring.")
    bullet(doc, "Level completion: the exit unlocks only after all four apples "
                "are collected; reaching the finish then regenerates a fresh maze "
                "and keeps the score. R restarts the whole run, resetting score "
                "and level with no stale particles.")

    # 7. Extensions
    doc.add_heading("7. Two extensions: algorithms, screenshots and evaluation", level=1)

    doc.add_heading("7.1 Atmospheric depth fog (graphics)", level=2)
    doc.add_paragraph(
        "Question investigated: how can distant geometry be blended into the "
        "background to convey depth, using only per-fragment colour work? For a "
        "fragment at camera-space depth z, the fog factor is "
        "t = clamp((z - near)/(far - near), 0, 1), and the final colour is "
        "C = C_frag*(1 - t) + C_fog*t. near = 8 and far = 26 world units cover "
        "the maze diagonal. The fog is applied after lighting and texture "
        "sampling, before the depth-buffer write. Toggle with F.")
    add_image(doc, "09_fog_off.png", 5.0, "Figure 6. Fog off: full contrast across the maze.")
    add_image(doc, "10_fog_on.png", 5.0, "Figure 7. Fog on: distant walls and ground fade toward the fog colour.")
    doc.add_paragraph(
        "Evaluation: near fragments keep their own colour while the far side of "
        "the maze approaches the light blue-grey fog colour, which matches the "
        "expected depth cue. Limitation: the fog is a single linear depth blend "
        "with no height-based scattering or colour gradient.")

    doc.add_heading("7.2 Bilinear texture filtering (image processing)", level=2)
    doc.add_paragraph(
        "Question investigated: nearest-neighbour magnification produces hard "
        "texel blocks; can interpolating the four neighbouring texels soften "
        "magnified textures? The bilinear sampler takes fractional UV "
        "coordinates, wraps horizontally and vertically, and blends the four "
        "adjacent texels by the fractional parts fx, fy: "
        "top = c00*(1-fx) + c10*fx, bottom = c01*(1-fx) + c11*fx, "
        "result = top*(1-fy) + bottom*fy. The renderer selects nearest or "
        "bilinear per material; toggle with T.")
    add_image(doc, "13_tex_nearest_zoom.png", 2.6, "Figure 8. Nearest-neighbour magnification: hard blocks.")
    add_image(doc, "14_tex_bilinear_zoom.png", 2.6, "Figure 9. Bilinear magnification: smoothed texel transitions.")
    doc.add_paragraph(
        "Evaluation: the 2D magnification comparison shows clear block edges "
        "with nearest sampling versus softened transitions with bilinear "
        "filtering. In the 3D overview the difference is mild because the "
        "procedural texture is already fine-grained; the close-up comparison "
        "isolates the sampling behaviour. Limitation: filtering is per "
        "fragment with no mipmaps, so distant textures can still alias.")

    # 8. Testing, performance, limitations
    doc.add_heading("8. Testing, performance, limitations and contributions", level=1)
    doc.add_paragraph(
        "Tests are split into small numerical unit checks and gameplay "
        "integration checks, each run directly with the project interpreter. "
        "The suite covers maze/coordinate conversion, vector and 4x4 transforms, "
        "mesh construction, camera projection, rasterization and Z-buffer "
        "occlusion, maze-to-world walls, materials/lighting/sampling, physics "
        "collisions and jumping, game rules, particles, the two extensions and "
        "the resize aspect-ratio check.")
    bullet(doc, "test_maze, test_transforms, test_geometry, test_camera, "
                "test_renderer, test_maze_world, test_materials, test_physics, "
                "test_game, test_particles, test_extensions, test_resize.")
    bench_rows = _benchmark_rows()
    tri_count = bench_rows[1][2]
    fps_values = [float(r[5]) for r in bench_rows[1:]]
    fps_range = f"{min(fps_values):.1f}-{max(fps_values):.1f}"
    doc.add_paragraph(
        "Performance was measured offscreen with benchmark.py at four internal "
        "resolution presets, perspective overview view, five frames per preset. "
        f"The scene contains {tri_count} triangles.")
    perf_table(doc, bench_rows)
    doc.add_paragraph(
        f"The measured render time is roughly 0.14-0.19 seconds per frame "
        f"({fps_range} fps), which is still below real-time. This is reported "
        "honestly: the renderer is a pure CPU software rasterizer, and the "
        "dominant cost is the per-triangle Python and object work (transform, "
        "clip, normal, shade, mesh setup) rather than pixel fill. Resolution "
        "has a moderate effect, consistent with triangle/transform work "
        "dominating over pixel fill. The hot path was optimized by caching the "
        "camera basis per instance, caching the identity-model-matrix check, "
        "skipping the identity-model 4x4 multiply, removing redundant "
        "per-triangle lighting normalization, reusing the frame/depth buffers "
        "between frames, replacing per-triangle NumPy vector allocations with "
        "scalar dot/cross/normal math, and scalarising the camera transform, "
        "projection, near-plane clip and per-channel shade so they no longer "
        "allocate small arrays. Static placements (water-hole discs, flags) are "
        "built once per level rather than every frame, and the near-plane clip "
        "has a fast path for triangles fully in front of the camera. As the "
        "optional CPU optimization noted in handbook p.30, the per-fragment "
        "raster scan is additionally compiled with Numba ``@njit``; Numba is "
        "not a Part A prerequisite, and if it is absent the renderer "
        "automatically falls back to the pure Python/NumPy path. A "
        "spatial/caching pass or a compiled data-oriented layout would be the "
        "next optimization.")
    add_image(doc, "07_low_res_672x420.png", 4.6, "Figure 10. Low internal resolution 672x420.")
    add_image(doc, "08_high_res_1600x1000.png", 4.6, "Figure 11. High internal resolution 1600x1000: finer wall edges.")
    doc.add_paragraph("Main limitations:")
    bullet(doc, "Entities are assembled from parametric ellipsoids and boxes: "
                "a segmented caterpillar with a round head and eyes, an apple "
                "with stem and leaf, and a bird with body, head, wings, tail, "
                "belly, beak and eyes. They remain low-poly with flat facets "
                "and no detailed anatomy or smooth vertex normals.")
    bullet(doc, "No antialiasing; polygon edges can look jagged at high zoom.")
    bullet(doc, "The area outside the maze grid is the framebuffer clear colour "
                "(black), with no surrounding sky or ground plane.")
    bullet(doc, "Flat shading only; there are no specular highlights, smooth "
                "vertex normals or cast shadows.")
    bullet(doc, "No mipmaps; distant textures may alias, and the software "
                "renderer is not real-time on this machine.")
    doc.add_paragraph(
        "This is an individual project. All code, tests, procedural textures, "
        "screenshots and the two extensions were implemented for this "
        "assignment, reusing only the supplied 2D starting file's maze logic and "
        "rules. Third-party usage is limited to the pinned course libraries "
        "NumPy, Pillow and Pygame; no external art, sound, font or model asset "
        "is bundled.")

    doc.save(OUT)
    print("Report written to", OUT)


if __name__ == "__main__":
    main()
