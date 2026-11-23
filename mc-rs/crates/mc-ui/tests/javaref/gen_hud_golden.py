#!/usr/bin/env python3
"""Generate golden HUD and F3 values by transcribing the Java, not the Rust.

This is the same trick S09 and S15 used, and for the same reason: asserting a port against
itself proves only that it is self-consistent. Everything below is a line-by-line transcription
of two Java methods, written from the decompiled source and *never* run against `mc-ui`:

  * ``GuiIngame.renderPlayerStats``  (MCP-919-main/src/minecraft/net/minecraft/client/gui/GuiIngame.java:609)
  * ``GuiOverlayDebug.call`` / ``getDebugInfoRight``  (.../GuiOverlayDebug.java:102, :171)

Run:  python gen_hud_golden.py > ../fixtures/hud_debug_golden.json

The output is committed, so the test needs no Python.
"""

import json
import math
import sys


# --------------------------------------------------------------------------- MathHelper
def ceiling_float_int(value):
    """MathHelper.ceiling_float_int: `(int) value` then `+1` when the cast lost anything."""
    i = int(value)  # Java's (int) truncates toward zero
    return i + 1 if value > i else i


def ceiling_double_int(value):
    i = int(value)
    return i + 1 if value > i else i


class JavaRandom:
    """java.util.Random, only the two methods GuiIngame uses."""

    MASK = (1 << 48) - 1

    def __init__(self, seed):
        self.seed = (seed ^ 0x5DEECE66D) & self.MASK

    def next(self, bits):
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & self.MASK
        r = self.seed >> (48 - bits)
        # to signed 32-bit
        if r >= 1 << 31:
            r -= 1 << 32
        return r

    def next_int(self, bound):
        if bound & (bound - 1) == 0:  # power of two
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            val = bits % bound
            if bits - val + (bound - 1) >= 0:
                return val


# --------------------------------------------------------------------- GuiIngame stats
def render_player_stats(width, height, s):
    """Transcription of GuiIngame.renderPlayerStats. Returns the blit list in draw order.

    Each entry is [x, y, u, v] of a 9x9 sprite from icons.png.
    """
    out = []
    i = ceiling_float_int(s["health"])                      # i = ceil(getHealth())
    flag = False                                            # the damage flash; not modelled
    k = s["food"]                                           # foodLevel
    l = s["prev_food"]
    flag1 = False
    i1 = width // 2 - 91                                    # left edge
    j1 = width // 2 + 91                                    # right edge
    k1 = height - 39                                        # baseline
    f = s["max_health"]
    f1 = s["absorption"]
    l1 = ceiling_float_int((f + f1) / 2.0 / 10.0)           # heart rows
    i2 = max(10 - (l1 - 2), 3)                              # row pitch
    j2 = k1 - (l1 - 1) * i2 - 10                            # armour / bubble row
    f2 = f1
    k2 = s["armor"]
    l2 = -1
    if s["regenerating"]:
        l2 = s["update_counter"] % ceiling_float_int(f + 5.0)

    rand = JavaRandom(s["update_counter"] * 312871)

    # ---- armour
    for i3 in range(10):
        if k2 > 0:
            j3 = i1 + i3 * 8
            if i3 * 2 + 1 < k2:
                out.append([j3, j2, 34, 9])
            if i3 * 2 + 1 == k2:
                out.append([j3, j2, 25, 9])
            if i3 * 2 + 1 > k2:
                out.append([j3, j2, 16, 9])

    # ---- health
    for i6 in range(ceiling_float_int((f + f1) / 2.0) - 1, -1, -1):
        j6 = 16
        if s["poisoned"]:
            j6 += 36
        elif s["withered"]:
            j6 += 72
        k3 = 1 if flag else 0
        l3 = ceiling_float_int((i6 + 1) / 10.0) - 1
        i4 = i1 + i6 % 10 * 8
        j4 = k1 - l3 * i2
        if i <= 4:
            j4 += rand.next_int(2)
        if i6 == l2:
            j4 -= 2
        k4 = 5 if s["hardcore"] else 0
        out.append([i4, j4, 16 + k3 * 9, 9 * k4])
        if flag:
            if i6 * 2 + 1 < l:
                out.append([i4, j4, j6 + 54, 9 * k4])
            if i6 * 2 + 1 == l:
                out.append([i4, j4, j6 + 63, 9 * k4])
        if f2 > 0.0:
            if f2 == f1 and f1 % 2.0 == 1.0:
                out.append([i4, j4, j6 + 153, 9 * k4])
            else:
                out.append([i4, j4, j6 + 144, 9 * k4])
            f2 -= 2.0
        else:
            if i6 * 2 + 1 < i:
                out.append([i4, j4, j6 + 36, 9 * k4])
            if i6 * 2 + 1 == i:
                out.append([i4, j4, j6 + 45, 9 * k4])

    # ---- food (no riding entity)
    for k6 in range(10):
        i7 = k1
        l7 = 16
        j8 = 0
        if s["hungry"]:
            l7 += 36
            j8 = 13
        if s["saturation"] <= 0.0 and s["update_counter"] % (k * 3 + 1) == 0:
            i7 = k1 + (rand.next_int(3) - 1)
        if flag1:
            j8 = 1
        i9 = j1 - k6 * 8 - 9
        out.append([i9, i7, 16 + j8 * 9, 27])
        if k6 * 2 + 1 < k:
            out.append([i9, i7, l7 + 36, 27])
        if k6 * 2 + 1 == k:
            out.append([i9, i7, l7 + 45, 27])

    # ---- air
    if s["underwater"]:
        l6 = s["air"]
        k7 = ceiling_double_int((l6 - 2) * 10.0 / 300.0)
        i8 = ceiling_double_int(l6 * 10.0 / 300.0) - k7
        for l8 in range(k7 + i8):
            if l8 < k7:
                out.append([j1 - l8 * 8 - 9, j2, 16, 18])
            else:
                out.append([j1 - l8 * 8 - 9, j2, 25, 18])

    return out


# --------------------------------------------------------------- GuiOverlayDebug strings
def java_floor(v):
    return math.floor(v)


def wrap_angle_to_180(f):
    """MathHelper.wrapAngleTo180_float: `f %= 360; if (f >= 180) f -= 360; if (f < -180) f += 360`."""
    f = math.fmod(f, 360.0)
    if f >= 180.0:
        f -= 360.0
    if f < -180.0:
        f += 360.0
    return f


def facing_of(yaw):
    """EnumFacing.getHorizontal(floor(yaw * 4 / 360 + 0.5) & 3) -> SOUTH, WEST, NORTH, EAST."""
    idx = java_floor(yaw / 90.0 + 0.5) & 3
    return ["south", "west", "north", "east"][idx]


FACING_TEXT = {
    "north": "Towards negative Z",
    "south": "Towards positive Z",
    "west": "Towards negative X",
    "east": "Towards positive X",
}


def debug_left(pos, rot):
    x, y, z = pos
    bx, by, bz = java_floor(x), java_floor(y), java_floor(z)
    facing = facing_of(rot[0])
    return [
        "XYZ: %.3f / %.5f / %.3f" % (x, y, z),
        "Block: %d %d %d" % (bx, by, bz),
        "Chunk: %d %d %d in %d %d %d" % (bx & 15, by & 15, bz & 15, bx >> 4, by >> 4, bz >> 4),
        "Facing: %s (%s) (%.1f / %.1f)"
        % (facing, FACING_TEXT[facing], wrap_angle_to_180(rot[0]), wrap_angle_to_180(rot[1])),
    ]


def debug_right(used, allocated, maximum):
    def mb(b):
        return b // 1024 // 1024

    return [
        "Mem: % 2d%% %03d/%03dMB" % (used * 100 // maximum, mb(used), mb(maximum)),
        "Allocated: % 2d%% %03dMB" % (allocated * 100 // maximum, mb(allocated)),
    ]


# ------------------------------------------------------------------------------- main
STATES = {
    "default": dict(
        health=20.0, max_health=20.0, absorption=0.0, armor=0, food=20, prev_food=20,
        saturation=5.0, air=300, underwater=False, hardcore=False, poisoned=False,
        withered=False, hungry=False, regenerating=False, update_counter=0,
    ),
    "hurt_and_hungry": dict(
        health=7.0, max_health=20.0, absorption=0.0, armor=9, food=6, prev_food=7,
        saturation=2.0, air=300, underwater=False, hardcore=False, poisoned=False,
        withered=False, hungry=True, regenerating=False, update_counter=13,
    ),
    "health_boost": dict(
        health=34.0, max_health=40.0, absorption=0.0, armor=20, food=20, prev_food=20,
        saturation=5.0, air=300, underwater=False, hardcore=False, poisoned=False,
        withered=False, hungry=False, regenerating=False, update_counter=5,
    ),
    "absorption_and_poison": dict(
        health=20.0, max_health=20.0, absorption=6.0, armor=3, food=20, prev_food=20,
        saturation=5.0, air=300, underwater=False, hardcore=True, poisoned=True,
        withered=False, hungry=False, regenerating=False, update_counter=2,
    ),
    "drowning": dict(
        health=12.0, max_health=20.0, absorption=0.0, armor=0, food=20, prev_food=20,
        saturation=5.0, air=152, underwater=True, hardcore=False, poisoned=False,
        withered=True, hungry=False, regenerating=False, update_counter=9,
    ),
    "big_absorption": dict(
        health=20.0, max_health=20.0, absorption=21.0, armor=0, food=20, prev_food=20,
        saturation=5.0, air=300, underwater=False, hardcore=False, poisoned=False,
        withered=False, hungry=False, regenerating=False, update_counter=1,
    ),
}

WIDTH, HEIGHT = 854, 480

out = {
    "note": "generated by tests/javaref/gen_hud_golden.py from the Java, never from the Rust",
    "screen": [WIDTH, HEIGHT],
    "hud": {},
    "debug_left": {},
    "debug_right": {},
}
for name, state in STATES.items():
    out["hud"][name] = {
        "state": state,
        "quads": render_player_stats(WIDTH, HEIGHT, state),
        "heart_rows": ceiling_float_int((state["max_health"] + state["absorption"]) / 2.0 / 10.0),
    }

for name, (pos, rot) in {
    "origin": ((0.0, 64.0, 0.0), (0.0, 0.0)),
    "fractional": ((123.456789, 64.0, -987.654321), (450.0, -12.34)),
    "negative": ((-0.5, 70.25, -17.75), (-100.0, 89.9)),
    "far": ((30000000.5, 255.0, -30000000.5), (179.99, -0.01)),
}.items():
    out["debug_left"][name] = {"position": list(pos), "rotation": list(rot), "lines": debug_left(pos, rot)}

for name, mem in {
    "quarter": (256 * 1024 * 1024, 512 * 1024 * 1024, 1024 * 1024 * 1024),
    "tiny": (5 * 1024 * 1024, 9 * 1024 * 1024, 100 * 1024 * 1024),
    "full": (2047 * 1024 * 1024, 2048 * 1024 * 1024, 2048 * 1024 * 1024),
}.items():
    out["debug_right"][name] = {"memory": list(mem), "lines": debug_right(*mem)}

json.dump(out, sys.stdout, indent=1)
sys.stdout.write("\n")
