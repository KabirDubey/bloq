"""Write the Track A BLOG inputs: Low et al. (arXiv:2605.30455) lattice-surgery gadgets in bloq's own layout (L1).

Every file is a closed block graph built only from bloq's blocks (cubes, walk, pipes, Hadamard pipes), so bloq
prepares and measures each time face itself. Orientation: the first two letters of a cube kind are its x and y
boundaries ("zx" = Z on the x faces, X on the y faces; "xz" = the other way). Basis: the third letter.

    python3 low_gadgets/native/make_blogs.py      # writes low_gadgets/native/inputs/*.blog
"""

import pathlib

OUT = pathlib.Path(__file__).parent / "inputs"
KINDS = {("zx", "z"): "ZXZ", ("zx", "x"): "ZXX", ("xz", "z"): "XZZ", ("xz", "x"): "XZX"}
OTHER = {"z": "x", "x": "z"}
AXES = {"h": (1, 0), "v": (0, 1)}  # h: patches side by side in x, v: one above the other in y


def write(name: str, comment: str, lines: list[str]) -> None:
    body = "\n".join(lines)
    (OUT / f"{name}.blog").write_text(f"BLOG 1.0\n# {comment}\n\nmodule main {{\n{body}\n}}\n")


def g3() -> None:
    """Temporal Hadamard: one cube, a Hadamard pipe, one cube. H exchanges both the boundaries and the basis."""
    for (o, b), kind in KINDS.items():
        after = KINDS[(o[::-1], OTHER[b])]
        write(f"g3n_hadamard_{o}_{b}", f"G3 native: {kind} -H> {after}; prepare {b}, read {OTHER[b]}",
              [f"0: {kind} [0, 0, 0]", f"1: {after} [0, 0, 1]", "0 -H> 1"])


def g4() -> None:
    """Walking: a cube, bloq's walk block one cell along x or y, a cube."""
    for (o, b), kind in KINDS.items():
        for ax, (dx, dy) in AXES.items():
            write(f"g4n_walk_{o}_{ax}_{b}", f"G4 native: walk {kind} one cell along {'x' if ax == 'h' else 'y'}",
                  [f"0: {kind} [0, 0, 0]", f"1: walk {kind} [0, 0, 1] -> [{dx}, {dy}, 2]",
                   f"2: {kind} [{dx}, {dy}, 3]", "[0, 0, 0] -> +Z", f"[{dx}, {dy}, 2] -> +Z"])


def surgery(o: str, b: str, ax: str, k: int, bridge_basis: str) -> list[str]:
    """Two patches for 3 cube layers, joined in the middle layer through k bridge cubes (k = 0: G5)."""
    dx, dy = AXES[ax]
    kind = KINDS[(o, b)]
    far = k + 1
    lines = [f"{i}: {kind} [{x * dx}, {x * dy}, {t}]" for i, (x, t) in enumerate(
        (x, t) for x in (0, far) for t in range(3))]
    lines += [f"{6 + j}: {KINDS[(o, bridge_basis)]} [{x * dx}, {x * dy}, 1]" for j, x in enumerate(range(1, far))]
    lines += ["0 -> 1", "1 -> 2", "3 -> 4", "4 -> 5"]
    chain = [1] + [6 + j for j in range(k)] + [4]
    lines += [f"{a} -> {c}" for a, c in zip(chain, chain[1:])]
    return lines


def g5_g6() -> None:
    """G5: merge and split as a spatial junction. G6: the same merge through k = 1..3 bridge cubes.

    The bridge cubes have no temporal pipe, so bloq prepares and measures them in their time basis. That basis is
    the one that leaves the merged parity deterministic: Z prepared cubes for an XX merge (z-faced seam), X for ZZ.
    """
    for (o, b) in KINDS:
        for ax in AXES:
            seam = o[0] if ax == "h" else o[1]  # boundary type of the faces the pipe joins
            write(f"g5n_surgery_{o}_{ax}_{b}", f"G5 native: merge and split of two {KINDS[(o, b)]} patches",
                  surgery(o, b, ax, 0, b))
            for k in (1, 2, 3):
                write(f"g6n_bridge_{o}_{ax}_{b}_k{k}", f"G6 native: merge through {k} bridge cubes",
                      surgery(o, b, ax, k, seam))


CNOT = """in control_in: data = 0
out control_out: data = 3
in target_in: data = 6
out target_out: data = 9

0: Port [0, 0, 0] <In_Control>
1: ZXZ [0, 0, 1]
2: ZXZ [0, 0, 2]
3: Port [0, 0, 3] <Out_Control>
4: ZXX [0, 1, 1]
5: ZXZ [0, 1, 2]
6: Port [1, 1, 0] <In_Target>
7: ZXZ [1, 1, 1]
8: ZXZ [1, 1, 2]
9: Port [1, 1, 3] <Out_Target>
[0, 0, 0] -> +Z
[0, 0, 1] -> +Z
[0, 0, 1] -> +Y
[0, 0, 2] -> +Z
[0, 1, 1] -> +Z
[0, 1, 2] -> +X
[1, 1, 0] -> +Z
[1, 1, 1] -> +Z
[1, 1, 2] -> +Z"""


def g7() -> None:
    """Lattice-surgery CX: bloq's gallery `cnot` (zx), and its mirror in the x = y plane (xz orientation)."""
    write("g7n_cx_zx", "G7 native: bloq gallery cnot (bloq_graph/assets/cnot.blog), ports closed by --auto-fill",
          CNOT.splitlines())
    swap = str.maketrans({"X": "Z", "Z": "X"})
    mirrored = []
    for line in CNOT.splitlines():
        if "[" in line and ":" in line and "->" not in line:  # block: swap x and y, swap the spatial letters
            head, rest = line.split(": ", 1)
            kind, coords = rest.split(" [", 1)
            x, y, t = coords.split("]")[0].split(", ")
            tail = coords.split("]", 1)[1]
            if kind != "Port":
                kind = kind[:2].translate(swap) + kind[2]
            line = f"{head}: {kind} [{y}, {x}, {t}]{tail}"
        elif "->" in line and "[" in line:  # pipe: swap x and y in the source and the direction
            src, dst = line.split(" -> ")
            x, y, t = src.strip("[]").split(", ")
            dst = {"+X": "+Y", "+Y": "+X"}.get(dst, dst)
            line = f"[{y}, {x}, {t}] -> {dst}"
        mirrored.append(line)
    write("g7n_cx_xz", "G7 native: the cnot mirrored in the x = y plane, ports closed by --auto-fill", mirrored)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.blog"):
        old.unlink()
    g3(), g4(), g5_g6(), g7()
    print(len(list(OUT.glob("*.blog"))), "files in", OUT)
