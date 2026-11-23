#!/usr/bin/env python3
"""Generate golden achievement-map values by transcribing the Java, not the Rust.

Sibling of ``gen_hud_golden.py`` and written the same way, for the same reason: a test that
compares a port against itself proves only that the port is self-consistent. Everything below is
a line-by-line transcription of the decompiled Java and has *never* been run against `mc-ui`:

  * ``GuiAchievements.drawAchievementScreen``  (MCP-919-main/src/minecraft/net/minecraft/client/gui/achievement/GuiAchievements.java:241)
  * ``GuiAchievements.updateScreen`` / ``drawScreen``'s clamps  (.../GuiAchievements.java:104, :212)
  * ``Gui.drawHorizontalLine`` / ``drawVerticalLine`` / ``drawRect``  (.../client/gui/Gui.java:20, :35, :50)
  * ``StatFileWriter.func_150874_c``  (.../stats/StatFileWriter.java:29)

It is a *separate* script rather than a section bolted onto ``gen_hud_golden.py`` so that
regenerating this fixture can never perturb the committed ``hud_debug_golden.json`` that
``tests/vanilla_hud_debug.rs`` depends on — the two goldens stay independent.

Run:  python gen_achievements_golden.py > ../fixtures/achievements_golden.json

The output is committed, so the test needs no Python.
"""

import json
import math
import sys


class JavaRandom:
    """java.util.Random — the constructor scramble, next(bits) and nextInt(bound)."""

    MASK = (1 << 48) - 1

    def __init__(self, seed):
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = (seed ^ 0x5DEECE66D) & self.MASK

    def next(self, bits):
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & self.MASK
        r = self.seed >> (48 - bits)
        if r >= 1 << 31:
            r -= 1 << 32
        return r

    def next_int(self, bound):
        if bound & (bound - 1) == 0:
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            val = bits % bound
            if bits - val + (bound - 1) >= 0:
                return val


def to_int(v):
    """Wrap to a signed 32-bit Java `int`."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def jdiv(a, b):
    """Java integer division: truncates toward zero, unlike Python's floor //."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def jmod(a, b):
    """Java's %: the sign follows the dividend."""
    return a - jdiv(a, b) * b


def jshr(a, n):
    """Java's >> on an int: arithmetic shift, which Python's >> already is."""
    return a >> n


# ------------------------------------------------------------------- the backdrop cascade
BLOCKS = [
    "sand",
    "dirt",
    "stone",
    "coal_ore",
    "iron_ore",
    "diamond_ore",
    "redstone_ore",
    "bedrock",
]


def backdrop_tile(player_id_hash, col, row):
    """`drawAchievementScreen`'s inner tile loop, for one tile.

    java:
        random.setSeed((long)(playerID.hashCode() + k1 + i4 + (l1 + l3) * 16));
        int j4 = random.nextInt(1 + l1 + l3) + (l1 + l3) / 2;
        ... cascade ...
    """
    seed = to_int(to_int(player_id_hash + col) + to_int(row * 16))
    rnd = JavaRandom(seed)
    bound = 1 + row
    if bound < 1:  # Java would throw; the port clamps, so the golden clamps too
        bound = 1
    j4 = rnd.next_int(bound) + jdiv(row, 2)
    sprite = 0  # `textureatlassprite = func_175371_a(Blocks.sand)` before the branch
    if j4 <= 37 and row != 35:
        if j4 == 22:
            sprite = 5 if rnd.next_int(2) == 0 else 6
        elif j4 == 10:
            sprite = 4
        elif j4 == 8:
            sprite = 3
        elif j4 > 4:
            sprite = 2
        elif j4 > 0:
            sprite = 1
    else:
        sprite = 7
    return sprite


def backdrop_layout(scroll_x, scroll_y, zoom, player_id_hash):
    """The whole tile loop: every tile's local (x, y) and which block it shows."""
    k1 = jshr(scroll_x + 288, 4)
    l1 = jshr(scroll_y + 288, 4)
    i2 = jmod(scroll_x + 288, 16)
    j2 = jmod(scroll_y + 288, 16)
    f = 16.0 / zoom
    f1 = 16.0 / zoom
    tiles = []
    l3 = 0
    while l3 * f - j2 < 155.0:
        f2 = 0.6 - (l1 + l3) / 25.0 * 0.3
        i4 = 0
        while i4 * f1 - i2 < 224.0:
            tiles.append(
                {
                    "x": i4 * 16 - i2,
                    "y": l3 * 16 - j2,
                    "block": BLOCKS[backdrop_tile(player_id_hash, k1 + i4, l1 + l3)],
                    "shade": round(f2, 6),
                }
            )
            i4 += 1
        l3 += 1
    return tiles


# --------------------------------------------------------------------- the achievement tree
def locked_depth(nodes, index):
    """StatFileWriter.func_150874_c."""
    by_key = {n["key"]: n for n in nodes}
    node = nodes[index]
    if node["unlocked"]:
        return 0
    i = 0
    cursor = by_key.get(node["parent"]) if node["parent"] else None
    while cursor is not None and not cursor["unlocked"]:
        i += 1
        cursor = by_key.get(cursor["parent"]) if cursor["parent"] else None
    return i


def can_unlock(nodes, index):
    by_key = {n["key"]: n for n in nodes}
    node = nodes[index]
    if not node["parent"]:
        return True
    parent = by_key.get(node["parent"])
    return bool(parent and parent["unlocked"])


def draw_rect(left, top, right, bottom):
    """Gui.drawRect, which normalises its corners. Returns the covered [x0, y0, x1, y1]."""
    if left < right:
        left, right = right, left
    if top < bottom:
        top, bottom = bottom, top
    return [min(left, right), min(top, bottom), max(left, right), max(top, bottom)]


def horizontal_line(start_x, end_x, y):
    if end_x < start_x:
        start_x, end_x = end_x, start_x
    return draw_rect(start_x, y, end_x + 1, y + 1)


def vertical_line(x, start_y, end_y):
    if end_y < start_y:
        start_y, end_y = end_y, start_y
    return draw_rect(x, start_y + 1, x + 1, end_y)


def connections(nodes, scroll_x, scroll_y):
    """The `parentAchievement != null` loop: two lines and an arrow per achievement."""
    by_key = {n["key"]: n for n in nodes}
    out = []
    for index, a in enumerate(nodes):
        if not a["parent"]:
            continue
        parent = by_key.get(a["parent"])
        if parent is None:
            continue
        k5 = a["column"] * 24 - scroll_x + 11
        l5 = a["row"] * 24 - scroll_y + 11
        j6 = parent["column"] * 24 - scroll_x + 11
        k6 = parent["row"] * 24 - scroll_y + 11
        flag = a["unlocked"]
        flag1 = can_unlock(nodes, index)
        k4 = locked_depth(nodes, index)
        if k4 > 4:
            continue
        l4 = -16777216
        if flag:
            l4 = -6250336
        elif flag1:
            l4 = -16711936
        entry = {
            "key": a["key"],
            "color": l4 & 0xFFFFFFFF,
            "horizontal": horizontal_line(k5, j6, l5),
            "vertical": vertical_line(j6, l5, k6),
        }
        if k5 > j6:
            entry["arrow"] = [k5 - 11 - 7, l5 - 5, 114, 234, 7, 11]
        elif k5 < j6:
            entry["arrow"] = [k5 + 11, l5 - 5, 107, 234, 7, 11]
        elif l5 > k6:
            entry["arrow"] = [k5 - 5, l5 - 11 - 7, 96, 234, 11, 7]
        elif l5 < k6:
            entry["arrow"] = [k5 - 5, l5 + 11, 96, 241, 11, 7]
        out.append(entry)
    return out


def cells(nodes, scroll_x, scroll_y, zoom):
    """The visible-node loop: the 26x26 plate, its tint, and the item position."""
    out = []
    for index, a in enumerate(nodes):
        l6 = a["column"] * 24 - scroll_x
        j7 = a["row"] * 24 - scroll_y
        if not (l6 >= -24 and j7 >= -24 and l6 <= 224.0 * zoom and j7 <= 155.0 * zoom):
            continue
        l7 = locked_depth(nodes, index)
        if a["unlocked"]:
            shade = 0.75
        elif can_unlock(nodes, index):
            shade = 1.0
        elif l7 < 3:
            shade = 0.3
        elif l7 == 3:
            shade = 0.2
        elif l7 == 4:
            shade = 0.1
        else:
            continue
        icon_shade = shade if can_unlock(nodes, index) else 0.1
        out.append(
            {
                "key": a["key"],
                "plate": [l6 - 2, j7 - 2, 26 if a["special"] else 0, 202, 26, 26],
                "shade": shade,
                "icon_shade": icon_shade,
                "item": [l6 + 3, j7 + 3],
            }
        )
    return out


def clamp_scroll(value, low, high):
    """drawScreen's four clamps: `< low -> low`, `>= high -> high - 1`."""
    if value < low:
        return float(low)
    if value >= high:
        return float(high - 1)
    return value


def ease(current, target):
    """updateScreen: 85% of the way, or all of it inside two pixels."""
    d0 = target[0] - current[0]
    d1 = target[1] - current[1]
    if d0 * d0 + d1 * d1 < 4.0:
        return (current[0] + d0, current[1] + d1)
    return (current[0] + d0 * 0.85, current[1] + d1 * 0.85)


# --------------------------------------------------------------------------------- main
# A five-node tree with every link direction and every reachability state represented.
TREE = [
    {"key": "achievement.openInventory", "column": 0, "row": 0, "parent": None,
     "unlocked": True, "special": False},
    {"key": "achievement.mineWood", "column": 2, "row": 1, "parent": "achievement.openInventory",
     "unlocked": True, "special": False},
    {"key": "achievement.buildWorkBench", "column": 4, "row": 2, "parent": "achievement.mineWood",
     "unlocked": False, "special": False},
    {"key": "achievement.buildPickaxe", "column": 4, "row": 4, "parent": "achievement.buildWorkBench",
     "unlocked": False, "special": False},
    {"key": "achievement.theEnd2", "column": 1, "row": 4, "parent": "achievement.buildPickaxe",
     "unlocked": False, "special": True},
]

SCROLLS = [(0, 0), (-82, -70), (24, -12), (-120, 35)]
HASHES = [0, 1, -991726233, 2147483647]

out = {
    "note": "generated by tests/javaref/gen_achievements_golden.py from the Java, never from the Rust",
    "tree": TREE,
    "backdrop_tiles": [],
    "backdrop_layout": {},
    "connections": {},
    "cells": {},
    "clamps": [],
    "easing": [],
}

for h in HASHES:
    for row in list(range(0, 40)) + [55, 90]:
        for col in range(-2, 6):
            out["backdrop_tiles"].append(
                {"hash": h, "col": col, "row": row, "block": BLOCKS[backdrop_tile(h, col, row)]}
            )

for sx, sy in SCROLLS:
    key = f"{sx},{sy}"
    out["backdrop_layout"][key] = {
        "scroll": [sx, sy],
        "hash": -991726233,
        "tiles": backdrop_layout(sx, sy, 1.0, -991726233),
    }
    out["connections"][key] = connections(TREE, sx, sy)
    out["cells"][key] = cells(TREE, sx, sy, 1.0)

# minDisplayColumn/Row and their maxima are `static int`s that start at 0 and are only narrowed.
MIN_COL = min([0] + [n["column"] for n in TREE])
MIN_ROW = min([0] + [n["row"] for n in TREE])
MAX_COL = max([0] + [n["column"] for n in TREE])
MAX_ROW = max([0] + [n["row"] for n in TREE])
LOW_X, LOW_Y = MIN_COL * 24 - 112, MIN_ROW * 24 - 112
HIGH_X, HIGH_Y = MAX_COL * 24 - 77, MAX_ROW * 24 - 77
out["bounds"] = [LOW_X, LOW_Y, HIGH_X, HIGH_Y]

for v in [-1000.0, -113.0, -112.0, 0.0, 18.0, 19.0, 20.0, 5000.0]:
    out["clamps"].append({"in": v, "x": clamp_scroll(v, LOW_X, HIGH_X), "y": clamp_scroll(v, LOW_Y, HIGH_Y)})

cur = (0.0, 0.0)
tgt = (100.0, -40.0)
for _ in range(12):
    cur = ease(cur, tgt)
    out["easing"].append([round(cur[0], 9), round(cur[1], 9)])

json.dump(out, sys.stdout, indent=1)
sys.stdout.write("\n")
