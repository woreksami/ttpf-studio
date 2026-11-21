"""Vehicle data: body colours, part models (bumpers, rims, tyres…) and paint lists for every vehicle.

The game keeps its vehicle tables in self-describing metadata resources (resource type 0x7a, `0DRPz` at +0x10) in
packs.rpf. Each carries its own schema, like a PSO file packed into a resource:

  header  +0x28 ptr struct table (32 bytes each), +0x38 ptr instance table (16 bytes each),
          +0x48 u16 struct count, +0x4c u16 instance block count
  struct  u32 name hash, u32 signature, u32 flags, u32 0, u64 ptr members, u32 size, u16 0, u16 member count
  member  u32 name hash, u32 offset, u8 type, u8 sub type, u16 reference (array element member), u32 extra
  block   u32 struct hash, u32 byte size, u64 ptr: size / struct size consecutive objects
  pointer a system pointer (0x5xxxxxxx), or a pool reference: 1-based block index in bits 0-11, offset in bits 20+

Structs and fields are named by joaat hashes. Format and field notes from the SixNG project
(https://github.com/holdmysocks/SixNG); this reader is a separate implementation.
"""
import struct
import zlib

MODEL_INFO = 0xe59aa857      # one per vehicle model: gameName, txdName, wheelScale, wheelType, ...
CONFIG = 0x51019114          # a ready-made vehicle: model, body colour (ARGB), its part models (incl. rim_*, tyre_*)
VARIATION = 0xfc1c7000       # paint colour lists (weighted by repetition) and weighted part groups
GAME_NAME = 0x120e5836
OWNER = 0xfb1f80ff
COLOUR = 0x5bdc13e3
PARTS = 0xe89b8540
PALETTE = 0x8c866829
WHEEL_SCALE = None           # filled from joaat('wheelScale') at import (case-sensitive field name)


def _joaat_cs(s):
    """joaat without lower-casing (these field names are case-sensitive)."""
    h = 0
    for c in s.encode():
        h = (h + c) & 0xffffffff
        h = (h + (h << 10)) & 0xffffffff
        h ^= h >> 6
    h = (h + (h << 3)) & 0xffffffff
    h ^= h >> 11
    return (h + (h << 15)) & 0xffffffff


WHEEL_SCALE, WHEEL_SCALE_REAR = _joaat_cs('wheelScale'), _joaat_cs('wheelScaleRear')
TXD_NAME = _joaat_cs('txdName')


class Meta:
    def __init__(self, v):
        self.v = v
        self.structs = {}
        st, it = self._sys(self.u64(0x28)), self._sys(self.u64(0x38))
        for i in range(self.u16(0x48)):
            o = st + 32 * i
            mp = self._sys(self.u64(o + 0x10))
            members = []
            for k in range(self.u16(o + 0x1e)):
                m = mp + 16 * k
                members.append((self.u32(m), self.u32(m + 4), v[m + 8], v[m + 9], self.u16(m + 10), self.u32(m + 12)))
            self.structs[self.u32(o)] = (self.u32(o + 0x18), members)
        self.blocks = []
        for i in range(self.u16(0x4c)):
            o = it + 16 * i
            p = self._sys(self.u64(o + 8))
            self.blocks.append((self.u32(o), self.u32(o + 4), p))

    # ---- raw reads
    def u16(self, o):
        return struct.unpack_from('<H', self.v, o)[0]

    def u32(self, o):
        return struct.unpack_from('<I', self.v, o)[0]

    def u64(self, o):
        return struct.unpack_from('<Q', self.v, o)[0]

    def f32(self, o):
        return struct.unpack_from('<f', self.v, o)[0]

    def _sys(self, p):
        o = p - 0x50000000 if 0x50000000 <= p < 0x58000000 else None
        return o if o is not None and o < len(self.v) else None

    def ptr(self, val):
        o = self._sys(val)
        if o is not None:
            return o
        bi, off = val & 0xfff, val >> 20
        if not 1 <= bi <= len(self.blocks):
            return None
        _, size, base = self.blocks[bi - 1]
        return base + off if base is not None and off < size else None

    # ---- objects
    def objects(self, h):
        if h not in self.structs:
            return []
        sz = max(1, self.structs[h][0])
        return [base + k * sz for bh, size, base in self.blocks if bh == h and base is not None
                for k in range(size // sz) if base + (k + 1) * sz <= len(self.v)]

    def _esize(self, m):
        t = m[2]
        if t in (0x01, 0x10, 0x11, 0x60):
            return 1
        if t in (0x12, 0x13):
            return 2
        if t in (0x14, 0x15, 0x4a, 0x21):
            return 4
        if t in (0x31, 0x32, 0x33, 0x34):
            return 16
        if t in (0x08, 0x4f, 0x07, 0x4d):
            return 8
        if t == 0x05 and m[5] in self.structs:
            return self.structs[m[5]][0]
        return 0

    def value(self, members, m, p, depth=0):
        t = m[2]
        if t in (0x15, 0x4a):
            return self.u32(p)
        if t == 0x14:
            return struct.unpack_from('<i', self.v, p)[0]
        if t == 0x21:
            return self.f32(p)
        if t in (0x11, 0x60):
            return self.v[p]
        if t == 0x13:
            return self.u16(p)
        if t == 0x01:
            return bool(self.v[p])
        if t in (0x32, 0x33):
            return struct.unpack_from('<3f', self.v, p)
        if t in (0x31, 0x34):
            return struct.unpack_from('<4f', self.v, p)
        if t == 0x05:
            return self.object(m[5], p, depth + 1)
        if t == 0x07:
            o = self.ptr(self.u64(p))
            if o is None:
                return None
            h = m[5] or next((bh for bh, size, base in self.blocks if base is not None and base <= o < base + size), 0)
            return self.object(h, o, depth + 1)
        if t in (0x52, 0x5c, 0x50):
            el = members[m[4]] if m[4] < len(members) else None
            if el is None:
                return []
            es = self._esize(el)
            if t == 0x52:
                n, base = self.u16(p + 8), self.ptr(self.u64(p))
            elif t == 0x5c:
                n, base = self.u32(p), self.ptr(self.u64(p + 8))
            else:
                n, base = m[5], p
            if base is None or not es or n > 100000:
                return []
            return [self.value(members, el, base + i * es, depth) for i in range(n)]
        return None

    def object(self, h, o, depth=0):
        if h not in self.structs or depth > 12:
            return None
        size, members = self.structs[h]
        out = {}
        for m in members:
            if m[0] == 0x100:                         # array element template, not a field
                continue
            try:
                out[m[0]] = self.value(members, m, o + m[1], depth)
            except (struct.error, IndexError):
                out[m[0]] = None
        return out


def linear_to_srgb(c):
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def argb_to_rgb(c):
    """Stored colours are linear (they read dark); shown as sRGB 0-255."""
    return tuple(int(round(255 * linear_to_srgb(((c >> s) & 0xff) / 255))) for s in (16, 8, 0))


def load(m, streams, progress=lambda f: None, cancel=None):
    """{model hash: dict(info, configs=[dict(colour, parts=[hash])], palettes=[[rgb]])} from an archive (packs.rpf)."""
    import models as M
    out = {}
    seen = set()
    for i, off in enumerate(streams):
        if i % 2000 == 0:
            progress(i / max(1, len(streams)))
            if cancel and cancel.is_set():
                break
        try:                                          # cheap peek: is this the start of a 0x7a meta resource?
            head = zlib.decompressobj().decompress(m[off:off + 256], 0x20)
        except zlib.error:
            continue
        if head[0x10:0x15] != b'0DRPz':
            continue
        v = M.inflate(m, off)
        r = M._virtual(m, streams, i)
        if r and len(r[0]) > len(v or b''):
            v = r[0]
        if not v:
            continue
        try:
            meta = Meta(v)
        except (struct.error, TypeError, IndexError):
            continue
        if MODEL_INFO not in meta.structs and CONFIG not in meta.structs:
            continue
        for o in meta.objects(MODEL_INFO):
            mi = meta.object(MODEL_INFO, o) or {}
            h = mi.get(GAME_NAME)
            if h and 'info' not in out.setdefault(h, dict(configs=[], palettes=[])):
                out[h]['info'] = {k: v for k, v in mi.items() if isinstance(v, (int, float))}
        for o in meta.objects(CONFIG):
            c = meta.object(CONFIG, o) or {}
            h, col = c.get(OWNER), c.get(COLOUR)
            parts = tuple(p for p in (c.get(PARTS) or []) if isinstance(p, int))
            key = (h, col, parts)
            if h and key not in seen:
                seen.add(key)
                out.setdefault(h, dict(configs=[], palettes=[]))['configs'].append(
                    dict(colour=argb_to_rgb(col) if isinstance(col, int) else None, parts=list(parts)))
        for o in meta.objects(VARIATION):
            c = meta.object(VARIATION, o) or {}
            h, pal = c.get(OWNER), [x for x in (c.get(PALETTE) or []) if isinstance(x, int)]
            if h and pal and (h, tuple(pal)) not in seen:
                seen.add((h, tuple(pal)))
                out.setdefault(h, dict(configs=[], palettes=[]))['palettes'].append([argb_to_rgb(x) for x in pal])
    progress(1.0)
    return out
