"""3D models from RPF9 archives (RAGE "gen9" resources): find them, decode their meshes, export OBJ.

A resource is a run of consecutive zlib streams: first its "virtual" chunks (structures, names), then
its "physical" chunks (GPU buffers). Pointers 0x50000000+ address the joined virtual data, 0x60000000+
the joined physical data. The root's +0x08 points to a block map whose bytes +8 / +9 hold the number of
virtual / physical chunks.

Drawable (+0x50) -> LOD list -> models (+0x08 array, +0x10 count) -> geometries. A geometry has the
marker 0x1601 at +0x60, its vertex buffer (physical) at +0x68 and its size in 4-byte words at +0x58;
+0x40 leads to its meshlet buffer. Vertex buffers start with stream descriptors (offset / 2 << 8 | type);
positions are packed in 8 bytes: two 24-bit values and one 16-bit value, scaled from an origin (+0x60)
by a step (+0x70); +0x6C says which axis is in which slot. Meshlets store triangles as three 7-bit local
indices per 32-bit word, mapped to vertices through a 16-bit table.

Format knowledge credit: the 6Stars project (github.com/Ahmet-Dedeler/6Stars) documented this layout.
This file is an independent implementation.
"""
import re
import struct
import zlib

GEOM_MARK = 0x1601


def u16(b, o):
    return struct.unpack_from('<H', b, o)[0]


def u32(b, o):
    return struct.unpack_from('<I', b, o)[0]


def u64(b, o):
    return struct.unpack_from('<Q', b, o)[0]


def f32(b, o):
    return struct.unpack_from('<f', b, o)[0]


def vptr(q):
    return q - 0x50000000 if 0x50000000 <= q < 0x58000000 else None


def pptr(q):
    return q - 0x60000000 if 0x60000000 <= q < 0x70000000 else None


def inflate(m, off, limit=64 << 20):
    d = zlib.decompressobj()
    out, q, n = [], off, 0
    while q < len(m):
        chunk = m[q:q + (1 << 16)]
        q += len(chunk)
        try:
            o = d.decompress(chunk)
        except zlib.error:
            return None
        out.append(o)
        n += len(o)
        if d.eof:
            return b''.join(out)
        if n > limit:
            return None
    return None


def stream_offsets(items):
    """Every zlib stream of an archive in file order (an item may span several 64 KB streams)."""
    offs = sorted(off for it in items if it.kind != 'video' for off, _ in it.parts)
    return offs


def _virtual(m, streams, i):
    """Join a resource's virtual chunks starting at stream i -> (virtual, next stream, physical chunk count) or None."""
    if i >= len(streams):
        return None
    v = inflate(m, streams[i])
    if not v or len(v) < 0x80 or u64(v, 0x60) == GEOM_MARK:
        return None
    bm = vptr(u64(v, 8))
    if bm is None or bm > (32 << 20):
        return None
    k = i + 1
    while len(v) <= bm + 10 and k < len(streams) and k - i < 32:
        nxt = inflate(m, streams[k])
        if nxt is None:
            return None
        v += nxt
        k += 1
    if bm + 10 > len(v):
        return None
    v_cnt, p_cnt = v[bm + 8], v[bm + 9]
    if not (1 <= v_cnt <= 32 and 1 <= p_cnt <= 64):
        return None
    while k - i < v_cnt and k < len(streams):
        v += inflate(m, streams[k]) or b''
        k += 1
    return v, k, p_cnt


def read_resource(m, streams, i):
    """Join a resource starting at stream index i -> (virtual, physical, streams used) or None."""
    r = _virtual(m, streams, i)
    if not r:
        return None
    v, k, p_cnt = r
    p = b''
    for _ in range(p_cnt):
        if k >= len(streams):
            break
        p += inflate(m, streams[k]) or b''
        k += 1
    return v, p, k - i


def model_name(v):
    for ext in (b'.#dr', b'.#ft', b'.#dd'):
        j = v.find(ext)
        if j > 1:
            s = j
            while s > 0 and (chr(v[s - 1]).isalnum() or v[s - 1] in b'_-'):
                s -= 1
            if 2 <= j - s <= 96 and v[s:j].isascii():
                return v[s:j].decode('ascii') + ext.decode()
    if len(v) >= 0xb0:
        no = vptr(u64(v, 0xa8))
        if no is not None and no < len(v):
            end = v.find(b'\0', no, no + 100)
            s = v[no:end]
            if 2 <= len(s) <= 96 and all(chr(c).isalnum() or c in b'_-.#' for c in s):
                return s.decode()
    return None


def meshlet_buffer(v, link):
    lo = vptr(link)
    if lo is None or lo + 8 > len(v):
        return None
    mo = vptr(u64(v, lo))
    if mo is None or mo + 0x20 > len(v):
        return None
    mb = pptr(u64(v, mo + 0x18))
    return None if mb is None else mb


def drawable_bases(v):
    """Where drawables start: the root itself, a fragment's drawable (+0x18), and the first entry of a
    drawable dictionary (+0x30 array, u16 count at +0x38; peds and clothing use these)."""
    out = [0]
    if len(v) >= 0x40:
        if vptr(u64(v, 0x18)) is not None:
            out.append(vptr(u64(v, 0x18)))
        arr, n = vptr(u64(v, 0x30)), u16(v, 0x38)
        if arr is not None and 0 < n <= 4096 and arr + 8 <= len(v) and vptr(u64(v, arr)) is not None:
            out.append(vptr(u64(v, arr)))  # shortcut: entries after the first are not shown
    return out


def geometries(v):
    """LOD0 geometries as (vertex buffer offset, vertex buffer bytes, meshlet buffer offset, drawable base, shader index),
    else every geometry found (base / shader None). Shadow-only copies are left out."""
    seen = set()
    for base in drawable_bases(v):
        if base + 0x58 > len(v):
            continue
        lod = vptr(u64(v, base + 0x50))
        if lod is None or lod + 0x10 > len(v):
            continue
        marr, mcnt = vptr(u64(v, lod)), u16(v, lod + 8)
        if marr is None or not (0 < mcnt <= 128) or marr + mcnt * 8 > len(v):
            continue
        found = []
        for mi in range(mcnt):
            mp = vptr(u64(v, marr + mi * 8))
            if mp is None or mp + 0x38 > len(v):
                continue
            garr, gcnt = vptr(u64(v, mp + 8)), u16(v, mp + 0x10)
            if garr is None or not (0 < gcnt <= 256) or garr + gcnt * 8 > len(v):
                continue
            smap = vptr(u64(v, mp + 0x20))
            if smap is not None and smap + gcnt * 2 > len(v):
                smap = None
            mask = v[mp + 0x34]  # bit 0: drawn in the main pass (others are shadow / auxiliary copies)
            for gi in range(gcnt):
                gp = vptr(u64(v, garr + gi * 8))
                if gp is None or gp + 0x70 > len(v) or u64(v, gp + 0x60) != GEOM_MARK:
                    continue
                vb, mb = pptr(u64(v, gp + 0x68)), meshlet_buffer(v, u64(v, gp + 0x40))
                if vb is not None and mb is not None:
                    found.append((vb, u32(v, gp + 0x58) * 4, mb, base, u16(v, smap + gi * 2) if smap is not None else gi, mask))
        if any(f[5] & 1 for f in found):
            found = [f for f in found if f[5] & 1]
        out = []
        for f in found:
            if (f[0], f[2]) not in seen:
                seen.add((f[0], f[2]))
                out.append(f[:5])
        if out:
            return out
    out = []
    for off in range(0x28, len(v) - 0x10, 8):  # fallback: every geometry marker in the resource
        if u64(v, off - 8) == GEOM_MARK:
            vb, mb = pptr(u64(v, off)), meshlet_buffer(v, u64(v, off - 0x28))
            if vb is not None and mb is not None and (vb, mb) not in seen:
                seen.add((vb, mb))
                out.append((vb, u32(v, off - 0x10) * 4, mb, None, None))
    return out


# shader parameter name hashes (joaat) for colour maps, best first
COLOUR_SLOTS = (0xc8e8c282, 0x346394ea, 0x0df47048)  # diffuseTex, vehicle body/detail slot, diffuseTex2
TINT_PALETTE = 0x2c8d7182
GLASS_SHADER = 0xe3ba3129
STREAMED = 0x980  # texture object that refers to a TTPF texture by 64-bit hash


def shader_info(v, base, idx):
    """-> dict(tex=hash or None, alpha=bool, glass=bool, paint=bool) for shader idx of the drawable at base."""
    info = dict(tex=None, alpha=False, glass=False, paint=False)
    if base is None or idx is None or base + 0x18 > len(v):
        return info
    sg = vptr(u64(v, base + 0x10))
    if sg is None or sg + 0x20 > len(v):
        return info
    sp, n = vptr(u64(v, sg + 0x10)), u16(v, sg + 0x18)
    if sp is None or idx >= n or sp + n * 8 > len(v):
        return info
    sh = vptr(u64(v, sp + idx * 8))
    if sh is None or sh + 0x40 > len(v) or v[sh + 4:sh + 8] != b'atem':
        return info
    pass_ = v[sh + 0x3c]
    info['alpha'] = pass_ in (2, 4, 8)
    info['hidden'] = pass_ >= 16                  # water-exclusion / proxy surfaces (boat hulls), never seen in game
    # 0xbd71bdbf: vehicle window glass; 0xfa9cc069: the dirt / wiper-sweep layer drawn over the windows
    info['glass'] = pass_ == 2 and u32(v, sh) in (GLASS_SHADER, 0xbd71bdbf, 0xfa9cc069)
    params, slots = vptr(u64(v, sh + 0x20)), vptr(u64(v, sh + 0x10))
    if params is None or slots is None or params + 8 > len(v):
        return info
    found, names = {}, set()
    for k in range(v[params + 1]):
        e = params + 8 + k * 8
        if e + 8 > len(v):
            break
        name, code = u32(v, e), u32(v, e + 4)
        names.add(name)
        if name not in COLOUR_SLOTS or code % 4 or slots + code * 4 + 8 > len(v):
            continue
        tp = vptr(u64(v, slots + code * 4))   # slot = 16-byte record per 4 code units
        if tp is None or tp + 0x38 > len(v) or u64(v, tp + 0x10) != STREAMED:
            continue
        ref = vptr(u64(v, tp + 0x30))
        if ref is not None and ref + 16 <= len(v):
            found[name] = u64(v, ref + 8)
    info['slot'] = next((n for n in COLOUR_SLOTS if n in found), None)
    info['tex'] = found.get(info['slot'])
    # most props carry the tint palette too; car paint shaders also have both the diffuse and the body slot
    info['paint'] = {TINT_PALETTE, 0xc8e8c282, 0x346394ea} <= names
    return info


def texture_index(folder, progress=lambda f: None):
    """{64-bit texture hash: (pack path, texture number)} for every .ttpf under folder.
    A pack's ETPF table is followed by its textures' hashes, sorted, in texture order."""
    import os
    packs = []
    for d, _, fs in os.walk(folder):
        packs += [os.path.join(d, f) for f in fs if f.lower().endswith('.ttpf')]
    out = {}
    for k, path in enumerate(sorted(packs)):
        progress(k / max(1, len(packs)))
        try:
            with open(path, 'rb') as f:
                head = f.read(0x400000)
        except OSError:
            continue
        at = head.find(b'ETPF', 0x18)
        if at < 0 or at + 16 > len(head):
            continue
        n = u32(head, at + 8)
        hb = at + 16 + n * 16
        if not 0 < n < 200000 or hb + n * 8 > len(head):
            continue
        hs = struct.unpack_from(f'<{n}Q', head, hb)
        if any(hs[j] > hs[j + 1] for j in range(n - 1)):
            continue  # not a sorted hash list: some other layout
        for j, h in enumerate(hs):
            out.setdefault(h, (path, j))
    return out


AXIS_SLOTS = {0: (2, 1, 0), 1: (2, 0, 1)}  # which world axis each packed slot holds (default x, y, z)


def decode_geometry(p, vb, vb_size, mb, verts, uvs, tris):
    """Append one geometry's vertices / UVs / triangles. Returns the number of triangles added."""
    if mb + 0x10 > len(p) or vb + 0x80 > len(p):
        return 0
    n_meshlets, n_tris, n_refs = u32(p, mb) & 0xFFFF, u32(p, mb + 4), u32(p, mb + 8)
    if not (n_meshlets and n_tris and n_refs):
        return 0
    tri_base = mb + 0x0C + n_meshlets * 16
    remap = tri_base + n_tris * 4
    if remap + n_refs * 2 > len(p):
        return 0
    desc = [u32(p, vb + 4 * k) for k in range(8)]
    pos_off = (desc[0] >> 8) << 1
    raw = desc[0] & 0xFF == 0x63  # plain float3 positions (glass and other small parts), not packed
    stride = 12 if raw else 8
    pos_end, uv = vb_size, None
    for h in desc[1:]:
        if not h:
            continue
        so, kind = (h >> 8) << 1, h & 0xFF
        if pos_off < so < pos_end:
            pos_end = so
        if so > pos_off and uv is None and kind in (0x28, 0x42):
            uv = (so, 4 if kind == 0x28 else 8)
    max_v = min((pos_end - pos_off) // stride, (len(p) - vb - pos_off) // stride)
    if max_v <= 0:
        return 0
    org = struct.unpack_from('<3f', p, vb + 0x60)
    step = struct.unpack_from('<3f', p, vb + 0x70)
    slots = AXIS_SLOTS.get(u32(p, vb + 0x6C), (0, 1, 2))
    base = len(verts)
    used, added, new_tris = -1, 0, []
    for mi in range(n_meshlets):
        mo = mb + 0x0C + mi * 16
        v0, t0 = u16(p, mo + 4), u16(p, mo + 6)
        t1 = u16(p, mo + 16 + 6) if mi + 1 < n_meshlets else n_tris
        for ti in range(t0, min(t1, n_tris)):
            w = u32(p, tri_base + ti * 4)
            r = (v0 + (w & 0x7F), v0 + ((w >> 7) & 0x7F), v0 + ((w >> 14) & 0x7F))
            if max(r) >= n_refs:
                continue
            i = [u16(p, remap + x * 2) for x in r]
            if max(i) < max_v and len(set(i)) == 3:
                new_tris.append((base + i[0], base + i[1], base + i[2]))
                used = max(used, *i)
                added += 1
    if not added:
        return 0
    start = vb + pos_off
    import numpy as np
    n = used + 1
    if raw:
        xyz = np.frombuffer(p, '<f4', n * 3, start).reshape(n, 3).astype(np.float64)
    else:
        rec = np.frombuffer(p, np.uint8, n * 8, start).reshape(n, 8).astype(np.int64)
        vals = np.stack([org[0] + (rec[:, 1] | rec[:, 2] << 8 | rec[:, 3] << 16) * step[0],   # 24-bit slot
                         org[1] + (rec[:, 0] | rec[:, 4] << 8 | rec[:, 5] << 16) * step[1],   # 24-bit slot
                         org[2] + (rec[:, 6] | rec[:, 7] << 8) * step[2]], 1)                 # 16-bit slot
        xyz = np.empty_like(vals)
        xyz[:, list(slots)] = vals
    if not (np.abs(xyz) < 1e5).all():
        return 0  # a layout we don't read right: leave the part out rather than wreck the whole model
    verts.extend(xyz.tolist())
    tris.extend(new_tris)
    uv_xy = np.zeros((n, 2))
    if uv:
        ok = max(0, min(n, (len(p) - vb - uv[0]) // uv[1]))
        if ok and uv[1] == 4:
            uv_xy[:ok] = np.frombuffer(p, '<i2', ok * 2, vb + uv[0]).reshape(ok, 2) / 8192
        elif ok:
            uv_xy[:ok] = np.frombuffer(p, '<f4', ok * 2, vb + uv[0]).reshape(ok, 2)
    uvs.extend(uv_xy.tolist())
    return added


def geometry_bones(p, vb, vb_size, n):
    """First (dominant) skin bone of each of the n vertices of a geometry, or None. Stream 8 of the mesh header's
    stream table (slot k at +4k: offset in 2-byte units << 8 | format): 0x25 = 4 x u8; format 0 is sized per vertex
    (1 = u8, 2 = u16, 4 = 4 x u8, 8 = 4 x u16). Layout from the SixNG project's notes."""
    import numpy as np
    h = u32(p, vb + 0x20)
    if not h or n <= 0:
        return None
    o, fmt = (h >> 8) << 1, h & 0xff
    offs = [(u32(p, vb + 4 * k) >> 8) << 1 for k in range(24) if u32(p, vb + 4 * k)]
    nxt = min([x for x in offs if x > o] + [vb_size])
    width = (nxt - o) // n
    a = vb + o
    if fmt == 0x25 or (fmt == 0 and 4 <= width < 8):
        w_, dt = 4, np.uint8
    elif fmt == 0 and 2 <= width < 4:
        w_, dt = 2, np.uint16
    elif fmt == 0 and 8 <= width < 16:
        w_, dt = 8, np.uint16
    elif fmt == 0 and width == 1:
        w_, dt = 1, np.uint8
    else:
        return None
    if a + w_ * n > len(p):
        return None
    arr = np.frombuffer(p, dt, n * w_ // np.dtype(dt).itemsize, a).reshape(n, -1)
    return arr[:, 0].astype(np.int32)


def decode_model(m, streams, i):
    """-> dict(name, verts, uvs, tris, geometries, streams) or None."""
    r = read_resource(m, streams, i)
    if not r:
        return None
    v, p, used = r
    geoms = geometries(v)
    if not geoms or len(p) < 0x90:
        return None
    verts, uvs, tris, tri_mat, mats, mat_of, bones = [], [], [], [], [], {}, []
    for vb, vb_size, mb, base, sidx in geoms:
        before = len(verts)
        n = decode_geometry(p, vb, vb_size, mb, verts, uvs, tris)
        if not n:
            continue
        if (base, sidx) not in mat_of:
            mat_of[(base, sidx)] = len(mats)
            mats.append(shader_info(v, base, sidx))
        tri_mat += [mat_of[(base, sidx)]] * n
        try:
            b = geometry_bones(p, vb, vb_size, len(verts) - before)
        except (struct.error, ValueError):
            b = None
        bones.append(b if b is not None else [-1] * (len(verts) - before))
    hidden = {k for k, mat in enumerate(mats) if mat.get('hidden')}
    if hidden:                                      # water-exclusion / proxy surfaces are not drawn
        keep = [k for k, mi in enumerate(tri_mat) if mi not in hidden]
        tris, tri_mat = [tris[k] for k in keep], [tri_mat[k] for k in keep]
    if not tris:
        return None
    import numpy as np
    return dict(name=model_name(v), verts=verts, uvs=uvs, tris=tris, tri_mat=tri_mat, materials=mats,
                geometries=len(geoms), streams=used, bones=np.concatenate([np.asarray(b) for b in bones]))


PAINT = (176, 180, 186)     # neutral metallic silver until the game's colour table is decoded (paint is applied in game, not a texture)
GLASS = (40, 55, 65)
PLAIN = (200, 200, 205)


class Textures:
    """Loads the TTPF textures a model's materials point to (packs stay open; images are cached)."""

    def __init__(self, index):
        self.index, self.packs, self.full, self.small, self.mid, self.clear = index or {}, {}, {}, {}, {}, {}

    def mostly_clear(self, h):
        """True when most of a texture is transparent (decals, wiper sweeps): the software preview leaves those out."""
        if h not in self.clear:
            import numpy as np
            img = self.image(h)
            self.clear[h] = bool(img is not None and img.mode == 'RGBA'
                                 and np.asarray(img.getchannel('A').resize((32, 32))).mean() < 128)
        return self.clear[h]

    def texels(self, h, limit=1024):
        """RGB numpy array of a texture (at most limit pixels a side) for per-pixel shading, or None."""
        if h not in self.mid:
            import numpy as np
            img = self.image(h)
            if img is not None and max(img.size) > limit:
                img = img.copy()
                img.thumbnail((limit, limit))
            self.mid[h] = np.asarray(img.convert('RGB'), float) if img is not None else None
        return self.mid[h]

    def image(self, h):
        if h not in self.full:
            img = None
            if h in self.index:
                path, k = self.index[h]
                try:
                    import ttpf_tool
                    pk = self.packs.get(path) or self.packs.setdefault(path, ttpf_tool.Pack(path))
                    img = pk.image(pk.textures[k])[0]
                    img = img.convert('RGBA') if img else None
                except Exception:
                    img = None
            self.full[h] = img
        return self.full[h]

    def sampler(self, h):
        """Small RGB numpy array for colouring the preview, or None."""
        if h not in self.small:
            import numpy as np
            img = self.image(h)
            self.small[h] = np.asarray(img.convert('RGB').resize((128, 128))) if img else None
        return self.small[h]

    def close(self):
        for pk in self.packs.values():
            try:
                pk.close()
            except Exception:
                pass
        self.packs.clear()


def material_colour(mat, paint=PAINT):
    if mat.get('rgb'):                                # a fixed colour set when a vehicle is assembled (rubber, rim metal)
        return mat['rgb']
    return GLASS if mat.get('glass') else paint if mat.get('paint') else PLAIN


TYRE_RUBBER = (28, 28, 30)
RIM_METAL = (178, 182, 188)


def skeleton(v):
    """{bone name: 4x4 bone->model matrix} from a fragment's skeleton (inverse bind matrices, inverted).
    Bone records are 0x50 bytes: rotation, translation, scale, +0x30 name pointer, +0x40 u16 fields
    (flags, ?, next sibling, parent, index, ...); the skeleton header points at the first record and, at +0x08,
    at one 4x4 inverse bind matrix per bone. Layout from the SixNG project's notes."""
    import numpy as np

    def name(q):
        o = vptr(u64(v, q + 0x30))
        if o is None or o >= len(v):
            return None
        s_ = v[o:v.find(b'\x00', o, o + 96)]
        return s_.decode() if s_ and all(c == 95 or 48 <= c <= 57 or 65 <= c <= 90 or 97 <= c <= 122 for c in s_) else None
    for p in range(0, len(v) - 0x50, 16):
        if not (u16(v, p + 0x48) == 0 and u16(v, p + 0x46) == 0xffff and name(p)):
            continue
        bones = []
        while p + (len(bones) + 1) * 0x50 <= len(v):
            q = p + len(bones) * 0x50
            n = name(q)
            if not n or u16(v, q + 0x48) != len(bones):
                break
            bones.append(n)
        if len(bones) < 2:
            continue
        target = (0x50000000 | p).to_bytes(8, 'little')
        hdr = next((o for o in range(0, len(v) - 0x20, 8) if v[o:o + 8] == target), None)
        ibm = vptr(u64(v, hdr + 8)) if hdr is not None else None
        if ibm is None or ibm + 64 * len(bones) > len(v):
            return {}
        out = {}
        for k, n in enumerate(bones):
            m_ = np.array(struct.unpack_from('<16f', v, ibm + 64 * k), np.float64).reshape(4, 4).T
            m_[3] = (0, 0, 0, 1)
            try:
                out[n] = np.linalg.inv(m_)
            except np.linalg.LinAlgError:
                pass
        return out
    return {}


def _append(dst, src, xf=None, rgb=None, force=False):
    import numpy as np
    v = np.asarray(src['verts'], np.float64)
    if xf is not None:
        v = v @ xf[:3, :3].T + xf[:3, 3]
    base, mb = len(dst['verts']), len(dst['materials'])
    dst['verts'] += v.tolist()
    dst['uvs'] += list(src['uvs'])
    dst['tris'] += [[x + base for x in t] for t in src['tris']]
    dst['tri_mat'] += [mi + mb for mi in (src.get('tri_mat') or [0] * len(src['tris']))]
    for mat in src['materials']:
        mat = dict(mat)
        if rgb and (force or mat.get('paint') or not mat.get('tex') or mat.get('slot') != 0xc8e8c282):
            mat['rgb'], mat['paint'], mat['tex'] = rgb, False, None
        dst['materials'].append(mat)


NOT_BODY_BONES = ('interior', 'steering', 'doorcard', 'seat', 'sunvisor', 'dash', 'pedal', 'handbrake', 'gear', 'headrest',
                  'brake', 'gas', 'clutch', 'winder', 'light', 'siren', 'indicator', 'lamp', 'engine', 'exhaust',
                  'suspension', 'axle')   # bones whose parts are trim, not body paint (rule from the SixNG project)
VEH_DATA_SLOTS = (0x346394ea, 0x7d4f12e2)  # layer data maps on vehicle paint shaders: when one exists the part is trim
TRIM = (40, 40, 43)
CHROME = (205, 208, 214)                   # lamp reflectors behind the lamp glass
LAMP_BONES = ('light', 'lamp', 'indicator', 'siren')


def vehicle_trim(model, bone_names, texture_index):
    """Which paint-shader materials are trim (interior, lamps, mechanics) rather than body paint, as the game does:
    a layer data texture exists for it, or at least 70% of the bones its vertices hang on are not body panels."""
    import numpy as np
    tm = np.asarray(model.get('tri_mat') or [0] * len(model['tris']))
    t = np.asarray(model['tris'])
    vb = np.asarray(model.get('bones') if model.get('bones') is not None else [-1] * len(model['verts']))
    out = set()
    for mi, mat in enumerate(model['materials']):
        if not mat.get('paint'):
            continue
        if mat.get('tex') and mat.get('slot') in VEH_DATA_SLOTS and mat['tex'] in texture_index:
            out.add(mi)
            continue
        idx = np.unique(vb[np.unique(t[tm == mi])])
        names = [bone_names[k].lower() for k in idx if 0 <= k < len(bone_names)]
        if names and sum(any(w in b for w in NOT_BODY_BONES) for b in names) * 10 >= len(names) * 7:
            out.add(mi)
    return out


def vehicle_lamps(model, bone_names, materials):
    """Of the given materials, those whose vertices hang mostly on lamp bones (headlights, indicators...)."""
    import numpy as np
    tm = np.asarray(model.get('tri_mat') or [0] * len(model['tris']))
    t = np.asarray(model['tris'])
    vb = np.asarray(model.get('bones') if model.get('bones') is not None else [-1] * len(model['verts']))
    out = set()
    for mi in materials:
        idx = np.unique(vb[np.unique(t[tm == mi])])
        names = [bone_names[k].lower() for k in idx if 0 <= k < len(bone_names)]
        if names and sum(any(w in b for w in LAMP_BONES) for b in names) * 2 > len(names):
            out.add(mi)
    return out


def _hidden_bone(b):
    b = b.lower()        # motion-blur stand-ins, damaged variants, break-off pieces / effect helpers, wiper sweeps
    return ('fast' in b or ('dmg' in b and 'non_dmg' not in b) or 'wiper' in b
            or b.endswith('_frag') or '_frag_' in b or b.endswith('_vfx') or '_vfx_' in b)


def drop_motion_blur(model, bone_names):
    """Remove geometry the game only shows in some states: blurred rotor / wheel discs at speed (*_fast bones) and the
    damaged versions of parts (*_dmg_* bones), which would otherwise sit as smudges over the real parts."""
    import numpy as np
    if model.get('bones') is None or not bone_names:
        return
    fast = np.array([_hidden_bone(b) for b in bone_names] + [False])
    vb = np.asarray(model['bones'])
    hide = fast[np.where((vb >= 0) & (vb < len(bone_names)), vb, len(bone_names))]
    t = np.asarray(model['tris'])
    keep = ~hide[t].any(1)
    if keep.all():
        return
    model['tris'] = t[keep].tolist()
    model['tri_mat'] = np.asarray(model['tri_mat'])[keep].tolist()


def _wheel_sizes(m, streams, index, name):
    """{wheel bone: (radius, width)} measured on the vehicle's lowest-detail model, whose wheels are built in."""
    import numpy as np
    stem = (name or '').split('.')[0]
    stem = stem[:-3] if stem.endswith('_hi') else stem
    if stem not in index:
        return {}
    try:
        mdl = decode_model(m, streams, index[stem])
        r = read_resource(m, streams, index[stem])
        bones = skeleton(r[0]) if r else {}
    except (struct.error, ValueError, IndexError):
        return {}
    if not mdl or mdl.get('bones') is None or not bones:
        return {}
    names = list(bones)
    vv, vb = np.asarray(mdl['verts']), np.asarray(mdl['bones'])
    out = {}
    for w in names:
        if not re.fullmatch(r'wheel_[lr][fmr]\d?', w.lower()):
            continue
        sel = vb == names.index(w)
        if sel.sum() < 12:
            continue
        d = (vv[sel] - bones[w][:3, 3]) @ bones[w][:3, :3]          # into the wheel's own axes (x = axle)
        radius, width = float(np.hypot(d[:, 1], d[:, 2]).max()), float(np.ptp(d[:, 0]))
        if 0.05 < radius < 3 and 0.01 < width < 3:
            out[w] = (radius, width)
    return out


def _lod_wheels(m, streams, index, name):
    """The wheels of a vehicle's lowest-detail model (triangles on its wheel bones), in vehicle space."""
    import numpy as np
    stem = (name or '').split('.')[0]
    stem = stem[:-3] if stem.endswith('_hi') else stem
    if stem not in index:
        return None
    try:
        mdl = decode_model(m, streams, index[stem])
        r = read_resource(m, streams, index[stem])
        bones = skeleton(r[0]) if r else {}
    except (struct.error, ValueError, IndexError):
        return None
    if not mdl or mdl.get('bones') is None or not bones:
        return None
    wheel = np.array([bool(re.fullmatch(r'wheel_[lr][fmr]\d?', b.lower())) for b in bones] + [False])
    vb = np.asarray(mdl['bones'])
    on = wheel[np.where((vb >= 0) & (vb < len(bones)), vb, len(bones))]
    t = np.asarray(mdl['tris'])
    keep = on[t].all(1)
    if not keep.any():
        return None
    return dict(mdl, tris=t[keep].tolist(), tri_mat=np.asarray(mdl['tri_mat'])[keep].tolist())


def _real_colour(c):
    """A configuration colour, or None for the magenta "untinted" marker the vehicle data uses (255, 0, 255)."""
    if not c:
        return None
    c = tuple(c)
    return None if c[0] > 200 and c[2] > 200 and c[1] < 120 else c


def _closest_part(prefix, other, index):
    """A rim_ / tyre_ model for a wheel with only the other half configured: the name sharing most words with it."""
    words = set(other.split('_')[1:])
    names = sorted({re.sub(r'_(hi|mid)$', '', n) for n in index if n.startswith(prefix)})
    return max(names, key=lambda t: (len(words & set(t.split('_'))), -len(t)), default=None)


def _closest_tyre(rim, index):
    return _closest_part('tyre_', rim, index)


def assemble_vehicle(m, streams, index, i, data, config=0, texture_index=()):
    """A complete vehicle as the game spawns it: the body (stream i), painted with the configuration's body
    colour, with that configuration's rim and tyre on the four wheel bones, sized by wheelScale.
    index: {model name without extension: stream index} of the same archive; data: vehmeta.load()[joaat(name)].
    -> model dict (with 'paint' set to the body colour), or the plain body if anything is missing."""
    import numpy as np
    import vehmeta
    body = decode_model(m, streams, i)
    if not body:
        return body
    data = data or {}
    cfgs = data.get('configs') or [dict(colour=None, parts=[])]
    cfg = cfgs[min(config, len(cfgs) - 1)]
    colour = _real_colour(cfg.get('colour'))
    if colour is None:                              # marker colour: the next real configuration colour, then the paint list
        colour = next((_real_colour(c.get('colour')) for c in cfgs if _real_colour(c.get('colour'))), None)
    if colour is None and data.get('palettes'):
        import collections
        colour = collections.Counter(tuple(c) for c in data['palettes'][0]).most_common(1)[0][0]
    body['paint'] = tuple(colour) if colour is not None else PAINT     # nothing real in the data: neutral silver
    r = read_resource(m, streams, i)
    bones = skeleton(r[0]) if r else {}
    trim = vehicle_trim(body, list(bones), texture_index)
    lamps = vehicle_lamps(body, list(bones), trim)
    for mi in trim:                                 # interior, mechanics: dark trim; lamp housings: chrome
        body['materials'][mi] = dict(body['materials'][mi], rgb=CHROME if mi in lamps else TRIM, paint=False, tex=None)
    drop_motion_blur(body, list(bones))
    names = {joaat(n): n for n in cfg.get('_names', ())} or None
    by_hash = names or {joaat(n): n for n in index}
    rim = next((by_hash[p] for p in cfg['parts'] if p in by_hash and by_hash[p].startswith('rim_')), None)
    tyre = next((by_hash[p] for p in cfg['parts'] if p in by_hash and by_hash[p].startswith('tyre_')), None)
    info = {int(k): val for k, val in (data.get('info') or {}).items()}     # keys come back as strings from JSON
    ws = info.get(vehmeta.WHEEL_SCALE) or 0.3
    wsr = info.get(vehmeta.WHEEL_SCALE_REAR) or ws

    def piece(n):
        for k in (n + '_hi', n, n + '_mid'):
            if k in index:
                pm = decode_model(m, streams, index[k])
                if pm:
                    rr = read_resource(m, streams, index[k])
                    drop_motion_blur(pm, list(skeleton(rr[0])) if rr else [])
                return pm
        return None
    if tyre and not rim:                            # some configurations list no rim: one from another configuration
        rim = (next((by_hash[p] for c in cfgs for p in c['parts'] if p in by_hash and by_hash[p].startswith('rim_')), None)
               or _closest_part('rim_', tyre, index))
    if rim and not tyre:                            # some configurations list no tyre: take the closest match
        tyre = (next((by_hash[p] for c in cfgs for p in c['parts'] if p in by_hash and by_hash[p].startswith('tyre_')), None)
                or _closest_tyre(rim, index))
    pieces = [(tyre and piece(tyre), TYRE_RUBBER), (rim and piece(rim), RIM_METAL)]
    # the real wheel size per axle: the vehicle's lowest-detail model carries its own wheels, at the right size, on
    # the same wheel bones. Rims and tyres are scaled to match it (radius and width).
    measured = _wheel_sizes(m, streams, index, body.get('name'))
    tyre_v = np.asarray(pieces[0][0]['verts']) if pieces[0][0] else None
    tyre_r = float(np.ptp(tyre_v[:, 1:], 0).max()) / 2 if tyre_v is not None and len(tyre_v) else None
    tyre_w = float(np.ptp(tyre_v[:, 0])) if tyre_v is not None and len(tyre_v) else None
    out = dict(name=body['name'], verts=[], uvs=[], tris=[], tri_mat=[], materials=[], paint=body.get('paint'),
               geometries=body.get('geometries'), streams=body.get('streams'))
    _append(out, body)
    if not (rim or tyre):                           # nothing configured: the low-detail model's own wheels, as they are
        lw = _lod_wheels(m, streams, index, body.get('name'))
        if lw:
            _append(out, lw, None, TYRE_RUBBER, force=True)
    # every axle: wheel_lf / rf / lr / rr, plus extra axles on buses and trucks (wheel_lm1, wheel_rr2, ...)
    for w in [b for b in bones if (rim or tyre) and re.fullmatch(r'wheel_[lr][fmr]\d?', b.lower())]:
        xf = bones[w].copy()
        # rims and tyres are modelled for one side: turn them 180 degrees about the vertical on the other side
        flip = np.diag([-1.0, -1.0, 1.0]) if xf[0, 3] > 0 else np.eye(3)
        size = measured.get(w)
        if size and tyre_r and tyre_w:
            sr, sx = size[0] / tyre_r, size[1] / tyre_w
        else:                                       # no low-detail wheels to measure: SixNG's fitted factor
            sr = 1.625 * (ws if w.lower()[7] == 'f' else wsr)
            sx = sr
        xf[:3, :3] = xf[:3, :3] @ flip @ np.diag([sx, sr, sr])
        for p_, rgb in pieces:
            if p_:
                _append(out, p_, xf, rgb, force=True)
    # pickups and trucks: the bed / rear body is its own model, mounted on the mod_bed_attach bone
    bed = None
    if 'mod_bed_attach' in bones:
        stem = (body.get('name') or '').split('.')[0]
        stem = stem[:-3] if stem.endswith('_hi') else stem
        bed = next((n for n in (stem + '_pickup', stem + '_truckbed', stem + '_bed', stem + '_pickup1') if
                    any(k in index for k in (n + '_hi', n, n + '_mid'))), None)
        bp = piece(bed) if bed else None
        if bp:
            for mi in vehicle_trim(bp, [], texture_index):
                bp['materials'][mi] = dict(bp['materials'][mi], rgb=TRIM, paint=False, tex=None)
            _append(out, bp, bones['mod_bed_attach'])
    out['parts'] = dict(rim=rim, tyre=tyre, colour=body.get('paint'), bed=bed)
    return out


def colourise(model, textures, paint=PAINT):
    """Per-triangle colours from each material's texture (sampled at the triangle's UV centre)."""
    import numpy as np
    t = np.asarray(model['tris'], dtype=np.int64)
    uv = np.asarray(model['uvs'], dtype=np.float64)
    tm = np.asarray(model.get('tri_mat') or [0] * len(t))
    out = np.empty((len(t), 3), dtype=np.float64)
    found = 0
    for mi, mat in enumerate(model.get('materials') or [{}]):
        sel = tm == mi
        if not sel.any():
            continue
        # body paint shaders: the colour is picked in game; their texture slots hold dirt / detail maps
        tex = textures.sampler(mat['tex']) if mat.get('tex') and not mat.get('paint') and not mat.get('glass') else None
        if tex is None:
            out[sel] = material_colour(mat, paint)
            continue
        found += 1
        c = uv[t[sel]].mean(1) % 1.0
        ys = np.clip((c[:, 1] * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
        xs = np.clip((c[:, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
        out[sel] = tex[ys, xs]
    model['colours'] = out
    model['textured'] = found
    return found


def write_obj(model, path, textures=None, paint=PAINT):
    """OBJ (+ .mtl and PNG textures next to it when textures are given)."""
    import os
    base = os.path.splitext(path)[0]
    stem = os.path.basename(base)
    mats = model.get('materials') or [{}]
    tm = model.get('tri_mat') or [0] * len(model['tris'])
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {model["name"] or "model"} - exported by TTPF Studio\n')
        if textures is not None:
            f.write(f'mtllib {stem}.mtl\n')
        f.writelines(f'v {x:.5f} {y:.5f} {z:.5f}\n' for x, y, z in model['verts'])
        f.writelines(f'vt {u:.5f} {1 - v:.5f}\n' for u, v in model['uvs'])
        for mi in range(len(mats)):
            faces = [tri for tri, m in zip(model['tris'], tm) if m == mi]
            if not faces:
                continue
            if textures is not None:
                f.write(f'usemtl m{mi}\n')
            f.writelines(f'f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n' for a, b, c in faces)
    if textures is None:
        return
    with open(base + '.mtl', 'w', encoding='utf-8') as f:
        for mi, mat in enumerate(mats):
            r, g, b = (x / 255 for x in material_colour(mat, paint))
            f.write(f'newmtl m{mi}\nKd {r:.3f} {g:.3f} {b:.3f}\n')
            if mat.get('glass'):
                f.write('d 0.35\n')
            img = textures.image(mat['tex']) if mat.get('tex') and not mat.get('paint') and not mat.get('glass') else None
            if img:
                png = f'tex_{mat["tex"]:016x}.png'  # by hash: models exported together share them
                if not os.path.exists(os.path.join(os.path.dirname(path), png)):
                    img.save(os.path.join(os.path.dirname(path), png))
                f.write(f'Kd 1 1 1\nmap_Kd {png}\n')
            f.write('\n')


def is_root(m, off):
    """Cheap check on the first bytes of a stream: does it look like the start of a resource?"""
    try:
        h = zlib.decompressobj().decompress(m[off:off + 512], 0x70)
    except zlib.error:
        return False
    return len(h) >= 0x68 and vptr(u64(h, 8)) is not None and u64(h, 0x60) != GEOM_MARK


def has_paint(v):
    """Does any material of this drawable / fragment use car paint? (Every vehicle body does; props never.)"""
    for base in drawable_bases(v):
        sg = vptr(u64(v, base + 0x10)) if base + 0x18 <= len(v) else None
        if sg is None or sg + 0x20 > len(v):
            continue
        n = u16(v, sg + 0x18)
        if 0 < n <= 256 and any(shader_info(v, base, k)['paint'] for k in range(n)):
            return True
    return False


def find_models(m, streams, progress=lambda f: None, cancel=None, painted=None):
    """[(stream index, name)] for every named drawable / fragment in the archive.
    painted: a set to collect the stream indexes of models with car paint (vehicles)."""
    out, i, tick = [], 0, 0
    while i < len(streams):
        if i >= tick:
            tick = i + 2000
            progress(i / max(1, len(streams)))
            if cancel and cancel.is_set():
                break
        if not is_root(m, streams[i]):
            i += 1
            continue
        r = _virtual(m, streams, i)
        n = model_name(r[0]) if r else None
        if not n:
            i += 1
            continue
        out.append((i, n))
        if painted is not None:
            try:
                if has_paint(r[0]):
                    painted.add(i)
            except (IndexError, struct.error):
                pass
        i = r[1] + r[2]  # skip this resource's own chunks: a later one can look like a resource start too
    return out


CATEGORIES = ('Animals', 'People', 'Clothing', 'Vehicles', 'Wheels', 'Weapons', 'Props', 'Plants', 'Interiors',
              'Terrain', 'Buildings & map')
_GARMENTS = ('shirt', 'tshirt', 'tanktop', 'hoodie', 'jacket', 'coat', 'vest', 'sweater', 'pants', 'jeans', 'shorts',
             'skirt', 'dress', 'shoes', 'boots', 'sneakers', 'sandals', 'socks', 'tights', 'hat', 'cap', 'mask', 'glasses',
             'necklace', 'earring', 'piercing', 'bracelet', 'watch', 'ring', 'gloves', 'bag', 'belt', 'bottoms', 'tops',
             'underwear', 'bra', 'swim', 'uniform', 'suit', 'scarf', 'helmet', 'hair', 'beard', 'eyebrow')


def category(name, painted=False):
    """A plain-language group for a model name, for browsing (one of CATEGORIES).
    painted: the model has car-paint materials (from find_models), which marks every vehicle and vehicle part."""
    n = name.lower()
    first = n.split('_')[0].split('.')[0]
    if re.match(r'cr[mbfri]_', n):
        return 'Animals'
    if n.startswith('w_'):
        return 'Weapons'
    if n.startswith(('rim_', 'wheel_')) or '_wheel' in n:
        return 'Wheels'
    if painted or n.startswith(('veh_', 'truckbeds')) or (n.endswith('.#ft') and '_' not in n.split('.')[0]):
        return 'Vehicles'
    if TILE_RE.match(n) or first == 'maptile':
        return 'Terrain'
    if n.startswith(('head_', 'char_', 'cs_', 'hc_', 'player_', 'body', 'u_', 'a_m_', 'a_f_')):
        return 'People'
    if '_' not in first and first.startswith(_GARMENTS):
        return 'Clothing'
    if n.startswith(('prop_', 'prefab_', 'gore_')):
        return 'Props'
    if n.startswith('veg_') or '_veg_' in n or '_tree' in n or '_bush' in n:
        return 'Plants'
    if first in ('int', 'mint') or '_int_' in n or n.endswith('_int.#dr'):
        return 'Interiors'
    return 'Buildings & map'


def whole_vehicle(name):
    """A complete vehicle at high detail: <vehicle>_hi.#ft (e.g. banshee_hi.#ft)."""
    return bool(re.fullmatch(r'[a-z0-9]+_hi\.#ft', name.lower()))


_SHARED = {'cmn', 'veh', 'prop', 'prefab', 'int', 'mint', 'tr', 'w', 'rim', 'a', 'b', 'u', 'v'}


def categorise(entries, painted=()):
    """{entry: category} for a whole model list. A vehicle's name (comet7, bison1800, …) is learned from its painted
    parts, so its unpainted glass, interior and trim pieces land in Vehicles too."""
    families = {n.lower().split('_')[0].split('.')[0] for i, n in entries if i in painted}
    families = {f for f in families if len(f) > 2 and f not in _SHARED and not re.match(r'[ns]\d\d$', f)}
    out = {}
    for e in entries:
        i, n = e[0], e[1]
        c = category(n, i in painted)
        if c in ('Buildings & map', 'Interiors', 'Props', 'Clothing') and n.lower().split('_')[0].split('.')[0] in families:
            c = 'Vehicles'
        out[e] = c
    return out


def joaat(s):
    """RAGE's name hash (Jenkins one-at-a-time on the lower-cased name): how placement refers to models."""
    h = 0
    for c in s.lower().encode():
        h = (h + c) & 0xffffffff
        h = (h + (h << 10)) & 0xffffffff
        h ^= h >> 6
    h = (h + (h << 3)) & 0xffffffff
    h ^= h >> 11
    return (h + (h << 15)) & 0xffffffff


# Map placement resources: a table of (field hash, byte size, pointer) rows at +0x38, row count at +0x4c.
# The placed-objects field holds 0x90-byte records. Layout from the RageForge project's notes.
ENTITIES = 0xce501483
ROOMS = 0x0b1811f1          # only interiors have rooms; their objects are in the interior's own coordinates
ENT_SIZE = 0x90


def placement_fields(v):
    t, n = vptr(u64(v, 0x38)) if len(v) >= 0x50 else None, u16(v, 0x4c) if len(v) >= 0x50 else 0
    if t is None or not 0 < n <= 1024 or t + n * 16 > len(v):
        return None
    out = {}
    for k in range(n):
        row = t + 16 * k
        o, size = vptr(u64(v, row + 8)), u32(v, row + 4)
        if o is not None and 0 < size and o + size <= len(v):
            out.setdefault(u32(v, row), (o, size))
    return out


def parse_placement(v):
    """-> [(model hash, (x, y, z), (qx, qy, qz, qw), scale xy, scale z, is_leaf)] or None.
    is_leaf: no other object in this file names it as its parent, i.e. it is the most detailed version."""
    fl = placement_fields(v)
    if not fl or ENTITIES not in fl or fl[ENTITIES][1] % ENT_SIZE or ROOMS in fl:
        return None
    o, size = fl[ENTITIES]
    rows = []
    for b in range(o, o + size, ENT_SIZE):
        rows.append((u32(v, b + 0x44), struct.unpack_from('<3f', v, b + 0x38), struct.unpack_from('<4f', v, b + 0x10),
                     f32(v, b + 0x54), f32(v, b + 0x58), u32(v, b + 0x5c)))
    parents = {r[5] for r in rows}
    return [(h, pos, q, sxy, sz, k not in parents) for k, (h, pos, q, sxy, sz, _) in enumerate(rows)]


def find_placements(m, streams, progress=lambda f: None, cancel=None):
    """Every placed object in an archive's placement files, as parse_placement rows."""
    out, i, tick = [], 0, 0
    while i < len(streams):
        if i >= tick:
            tick = i + 2000
            progress(i / max(1, len(streams)))
            if cancel and cancel.is_set():
                break
        rows = None
        if is_root(m, streams[i]):
            v = inflate(m, streams[i])      # most placement files fit in one stream; their page map is not a model's
            rows = parse_placement(v) if v else None
            if not rows:
                r = _virtual(m, streams, i)
                rows = parse_placement(r[0]) if r else None
        if rows:
            out += rows
        i += 1
    return out


def _fragments(sx, sy, sz, t, W, H):
    """Rasterise triangles -> (pixel index, triangle, barycentric b1, b2) of the nearest surface at each covered pixel."""
    import numpy as np
    x, y, z = sx[t], sy[t], sz[t]
    parent = np.arange(len(t))
    corner = np.broadcast_to(np.eye(3), (len(t), 3, 3)).copy()   # each corner as barycentrics of the parent
    for _ in range(12):                         # split big triangles in four until every one is small on screen
        big = np.maximum(np.ptp(x, 1), np.ptp(y, 1)) > 32
        if not big.any():
            break
        keep = ~big

        def mid(a, i, j):
            return (a[big][:, i] + a[big][:, j]) / 2

        parts = []
        for arr in (x, y, z, corner):
            a0, a1, a2 = arr[big][:, 0], arr[big][:, 1], arr[big][:, 2]
            m01, m12, m20 = mid(arr, 0, 1), mid(arr, 1, 2), mid(arr, 2, 0)
            subs = [np.stack(s, 1) for s in ((a0, m01, m20), (m01, a1, m12), (m20, m12, a2), (m01, m12, m20))]
            parts.append(np.concatenate([arr[keep]] + subs))
        x, y, z, corner = parts
        parent = np.concatenate([parent[keep]] + [parent[big]] * 4)
    x0, x1 = np.floor(x.min(1)).astype(np.int64), np.ceil(x.max(1)).astype(np.int64)
    y0, y1 = np.floor(y.min(1)).astype(np.int64), np.ceil(y.max(1)).astype(np.int64)
    ok = (x1 >= 0) & (x0 < W) & (y1 >= 0) & (y0 < H)
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    ok &= np.abs(area) > 1e-9
    ext = np.maximum(x1 - x0, y1 - y0) + 1
    pix, tid, bb1, bb2, dep = [], [], [], [], []
    f4 = np.float32
    for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 1 << 30)):
        sel = np.nonzero(ok & (ext > lo) & (ext <= hi))[0]
        if not len(sel):
            continue
        B = int(ext[sel].max())
        g = np.arange(B, dtype=f4)
        step = max(1, 4_000_000 // (B * B))      # bound memory per chunk
        for c0 in range(0, len(sel), step):
            s = sel[c0:c0 + step]
            px = (x0[s, None, None] + 0.5).astype(f4) + g[None, None, :]     # (n, 1, B): columns
            py = (y0[s, None, None] + 0.5).astype(f4) + g[None, :, None]     # (n, B, 1): rows
            ax, ay = x[s, 0, None, None].astype(f4), y[s, 0, None, None].astype(f4)
            ar = area[s, None, None].astype(f4)
            dx, dy = px - ax, py - ay
            b1 = (dx * (y[s, 2, None, None] - ay).astype(f4) - (x[s, 2, None, None] - ax).astype(f4) * dy) / ar
            b2 = ((x[s, 1, None, None] - ax).astype(f4) * dy - dx * (y[s, 1, None, None] - ay).astype(f4)) / ar
            inside = (b1 >= -1e-5) & (b2 >= -1e-5) & (b1 + b2 <= 1 + 1e-5) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            k, r, c = np.nonzero(inside)
            if not len(k):
                continue
            B1, B2 = b1[k, r, c].astype(np.float64), b2[k, r, c].astype(np.float64)
            tt_ = s[k]
            pix.append((y0[tt_] + r) * W + x0[tt_] + c)
            tid.append(tt_)
            bb1.append(B1)
            bb2.append(B2)
            dep.append(z[tt_, 0] * (1 - B1 - B2) + z[tt_, 1] * B1 + z[tt_, 2] * B2)
    if not pix:
        return None
    pix, tid, bb1, bb2, dep = (np.concatenate(a) for a in (pix, tid, bb1, bb2, dep))
    order = np.lexsort((dep, pix))              # nearest first within each pixel
    pix, keep = np.unique(pix[order], return_index=True)
    sel = order[keep]
    s, B1, B2 = tid[sel], bb1[sel], bb2[sel]
    pb = corner[s, 0] * (1 - B1 - B2)[:, None] + corner[s, 1] * B1[:, None] + corner[s, 2] * B2[:, None]
    return pix, parent[s], pb[:, 1], pb[:, 2]


def _studio(W, H, bg):
    """Soft vertical gradient behind the model."""
    import numpy as np
    top = np.array(bg, float) + 26
    bot = np.array(bg, float) * 0.7
    f = np.linspace(0, 1, H)[:, None, None]
    return np.broadcast_to(top * (1 - f) + bot * f, (H, W, 3)).copy()


def _find_models_range(path, streams, base):
    """find_models over a slice of an archive's streams (a worker process): [(index, name, painted)]."""
    import mmap
    with open(path, 'rb') as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            painted = set()
            found = find_models(m, streams, painted=painted)
        finally:
            m.close()
    return [(base + i, n, int(i in painted)) for i, n in found]


def find_models_parallel(path, streams, progress=lambda f: None, workers=None):
    """find_models on all CPU cores -> [(stream index, name, painted)]. A worker starting mid-resource may misread
    its first streams, so each reads past its slice and the lists are joined where two workers agree (same result as
    one front-to-back pass)."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    n = len(streams)
    workers = workers or max(1, min(16, (os.cpu_count() or 2) - 1))
    step = max(20000, -(-n // (workers * 4)))
    bounds = [(a, min(n, a + step)) for a in range(0, n, step)]
    parts = [None] * len(bounds)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_find_models_range, path, streams[a:min(n, b + 400)], a): k for k, (a, b) in enumerate(bounds)}
        for done, fu in enumerate(as_completed(futs)):
            parts[futs[fu]] = fu.result()
            progress((done + 1) / len(bounds))
    out, cur = [], parts[0]
    for k in range(1, len(parts)):
        nxt = parts[k]
        starts = {r[0]: j for j, r in enumerate(nxt)}
        j = next((j for j, r in enumerate(cur) if r[0] >= bounds[k][0] and r[0] in starts), None)
        if j is None:
            out += [r for r in cur if r[0] < bounds[k][1]]
            cur = [r for r in nxt if not out or r[0] > out[-1][0]]
            continue
        out += cur[:j]
        cur = nxt[starts[cur[j][0]]:]
    return out + cur


def render(model, size=640, yaw=0.8, pitch=0.45, zoom=1.0, bg=(24, 26, 31), textures=None, paint=PAINT, smooth=True, pan=(0, 0)):
    """Preview as a PIL image (game is Z-up; shown Y-up). With textures, every pixel samples its material's
    colour map; otherwise model["colours"] (per triangle) or plain grey. Smooth lighting, 2x anti-aliasing."""
    import numpy as np
    from PIL import Image
    v = np.asarray(model['verts'], dtype=np.float64)
    t = np.asarray(model['tris'], dtype=np.int64)
    size = (size, size) if isinstance(size, int) else tuple(size)     # an int, or (width, height)
    if not len(v) or not len(t):
        return Image.new('RGB', size, bg)
    ss = 2 if smooth else 1
    SW, SH = size[0] * ss, size[1] * ss
    mats_ = model.get('materials') or []
    if textures is not None and mats_:              # no alpha blending here: leave out mostly-transparent decals
        skip = [mi for mi, mat in enumerate(mats_) if mat.get('alpha') and mat.get('tex') and textures.mostly_clear(mat['tex'])]
        if skip:
            keep = ~np.isin(np.asarray(model.get('tri_mat') or [0] * len(t)), skip)
            t = t[keep]
            model = dict(model, tris=t, tri_mat=list(np.asarray(model.get('tri_mat') or [0] * len(keep))[keep]),
                         colours=None if model.get('colours') is None else np.asarray(model['colours'])[keep])
            if not len(t):
                return Image.new('RGB', size, bg)
    # frame the bulk of the model: a few stray far-off vertices must not shrink it to a dot in a corner
    lo, hi = np.percentile(v, 0.5, axis=0), np.percentile(v, 99.5, axis=0)
    v = (v - (lo + hi) / 2)[:, [0, 2, 1]] * np.array([1, 1, -1])
    cy, sy_, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    w = v @ ry.T @ rx.T
    # fit the footprint, not a bounding sphere: the widest the model gets while spinning (its ground diagonal) across,
    # and its height plus the part of its top the tilt shows, up and down. Stable while turning, and a long car fills the view.
    ext = (hi - lo)[[0, 2, 1]]                      # x, up, depth in view axes
    across = float(np.hypot(ext[0], ext[2])) or 1.0
    updown = float(ext[1] * abs(np.cos(pitch)) + across * abs(np.sin(pitch))) or 1.0
    px_per = min(SW * 0.9 / across, SH * 0.86 / updown) * zoom
    sx = w[:, 0] * px_per + SW / 2 + pan[0] * SH         # pan: camera shift in view heights (right-drag)
    sy = SH / 2 - w[:, 1] * px_per - pan[1] * SH
    img = _studio(SW, SH, bg)
    fr = _fragments(sx, sy, -w[:, 2], t, SW, SH)
    if fr is None:
        return Image.fromarray(img.astype(np.uint8)).resize(size, Image.LANCZOS)
    pix, tid, b1, b2 = fr
    b0 = 1 - b1 - b2
    # smooth normals: face normals summed onto their vertices
    fn = np.cross(w[t[:, 1]] - w[t[:, 0]], w[t[:, 2]] - w[t[:, 0]])
    vn = np.zeros_like(w)
    for k in range(3):
        np.add.at(vn, t[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    tv = t[tid]
    nrm = vn[tv[:, 0]] * b0[:, None] + vn[tv[:, 1]] * b1[:, None] + vn[tv[:, 2]] * b2[:, None]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    nrm *= np.where(nrm[:, 2:3] < 0, -1, 1)    # two-sided: light the side we see
    key = np.array([-0.45, 0.6, 0.66]) / np.linalg.norm([-0.45, 0.6, 0.66])
    fill = np.array([0.6, 0.1, 0.8]) / np.linalg.norm([0.6, 0.1, 0.8])
    half = (key + [0, 0, 1]) / np.linalg.norm(key + [0, 0, 1])
    light = 0.32 + 0.62 * np.clip(nrm @ key, 0, 1) + 0.18 * np.clip(nrm @ fill, 0, 1)
    spec = 0.12 * np.clip(nrm @ half, 0, 1) ** 24
    rim = 0.10 * (1 - np.abs(nrm[:, 2])) ** 3
    # base colour per pixel
    base = np.full((len(pix), 3), 175.0)
    tc = model.get('colours')
    if tc is not None and len(tc) == len(t):
        base = np.asarray(tc, float)[tid]
    mats = model.get('materials') or []
    if textures is not None and mats and model.get('uvs') is not None and len(model['uvs']) == len(v):
        uv = np.asarray(model['uvs'], float)
        tm = np.asarray(model.get('tri_mat') or [0] * len(t))[tid]
        puv = uv[tv[:, 0]] * b0[:, None] + uv[tv[:, 1]] * b1[:, None] + uv[tv[:, 2]] * b2[:, None]
        for mi, mat in enumerate(mats):
            sel = tm == mi
            if not sel.any():
                continue
            arr = textures.texels(mat['tex']) if mat.get('tex') and not mat.get('paint') and not mat.get('glass') else None
            if arr is None:
                base[sel] = material_colour(mat, paint) if tc is None else base[sel]
                continue
            th, tw = arr.shape[:2]
            uu, vv = puv[sel, 0] % 1.0, puv[sel, 1] % 1.0
            base[sel] = arr[np.clip((vv * th).astype(int), 0, th - 1), np.clip((uu * tw).astype(int), 0, tw - 1)]
    col = base * light[:, None] + 255 * (spec + rim)[:, None]
    flat = img.reshape(-1, 3)
    flat[pix] = col
    out = Image.fromarray(np.clip(flat.reshape(SH, SW, 3), 0, 255).astype(np.uint8))
    return out.resize(size, Image.LANCZOS) if ss > 1 else out


# ---------------------------------------------------------------- whole map (terrain tiles)
# Terrain is cut into 512 m cells named sSS_wWW (s grows south = -y, w grows west = -x). A cell holds
# 8 x 8 tiles of 64 m, numbered 1..64 row by row (+1 = east, +8 = north), in three detail levels:
# _hd (full mesh), _lod (light mesh + a baked top-down colour map) and _slod (coarser, 4 x 4 of 128 m).
# Tiles are stored around their own centre with their height offset removed, so neighbours are lifted
# until their shared edges meet (they agree to well under a millimetre).
TILE_RE = re.compile(r'([ns]\d\d)_([ew]\d\d)(?:_([ns]\d\d)_([ew]\d\d))?_(?:(\d+)_)?(hd|lod|slod\d?)\.')
CELL = 512.0


def _cell(c):
    """Cell name part -> its low edge in metres: n01 / e01 start at 0, s01 / w01 end at 0."""
    return (int(c[1:]) - 1) * CELL if c[0] in 'ne' else -int(c[1:]) * CELL


def area_rect(text):
    """'s13_w19' or two corner cells 's13_w19 n02_e03' -> (x0, y0, x1, y1) in metres covering them, or None."""
    cells = re.findall(r'([ns]\d\d)_([ew]\d\d)', text.lower())
    if not cells:
        return None
    xs = [_cell(ew) for _, ew in cells]
    ys = [_cell(ns) for ns, _ in cells]
    return min(xs), min(ys), max(xs) + CELL, max(ys) + CELL


def tile_rect(name):
    """-> (kind, x0, y0, size) of a terrain tile in map metres, or None."""
    g = TILE_RE.match(name)
    if not g:
        return None
    ns, ew, ns2, ew2, k, kind = g.groups()
    xa, xb, ya, yb = _cell(ew), _cell(ew2 or ew), _cell(ns), _cell(ns2 or ns)
    x0, y0 = min(xa, xb), min(ya, yb)
    if not k:                                  # a whole cell, or a block of cells (slod2..4)
        if kind in ('hd', 'lod'):
            return None
        return kind, x0, y0, abs(xb - xa) + CELL
    k = int(k) - 1
    n = 8 if kind in ('hd', 'lod') else 4
    if not 0 <= k < n * n:
        return None
    size = CELL / n
    return kind, x0 + k % n * size, y0 + k // n * size, size


def uv_fit(mdl):
    """The baked colour maps are flat projections: uv = [x, y, 1] @ sol. -> sol (3 x 2) or None."""
    import numpy as np
    v, uv = np.asarray(mdl['verts'])[:, :2], np.asarray(mdl['uvs'])
    if len(v) < 3 or len(uv) != len(v):
        return None
    A = np.c_[v, np.ones(len(v))]
    sol = np.linalg.lstsq(A, uv, rcond=None)[0]
    return sol if np.abs(A @ sol - uv).max() < 0.01 else None


def placed(rect, sol):
    """Where a lod / slod tile really is. Its name only picks the colour map it shares with its neighbours
    (2 x 2 tiles for _lod, a whole cell for _slod); the spot inside that map comes from its UVs."""
    if sol is None or rect[0] == 'hd' or sol[0, 0] <= 0 or sol[1, 1] >= 0:
        return rect
    kind, x0, y0, size = rect
    span = 1 / sol[0, 0]                        # metres one colour map covers
    if not size * 0.99 <= span <= 8 * CELL:
        return rect
    ax, ay = (x0 + size / 2) // span * span, (y0 + size / 2) // span * span
    cx, cy = ax + sol[2, 0] * span, ay + span - sol[2, 1] * span
    return kind, round((cx - size / 2) / size) * size, round((cy - size / 2) / size) * size, size


def _edge(v, side, half):
    """Heights along one edge of a tile, sampled at 32 points (None if the edge has too few vertices)."""
    import numpy as np
    ax, sgn = (0 if side[1] == 'x' else 1), (1 if side[0] == '+' else -1)
    sel = np.abs(v[:, ax] - half * sgn) < half / 50
    if sel.sum() < 8:
        return None
    e = v[sel]
    o = e[:, 1 - ax]
    k = np.argsort(o)
    return np.interp(np.linspace(-half * 0.97, half * 0.97, 32), o[k], e[k, 2])


def lift_tiles(tiles):
    """tiles: {(x0, y0): (verts, size)} -> {(x0, y0): z offset} so shared edges meet (0 for the first of each island)."""
    import numpy as np
    links = {t: [] for t in tiles}
    for (x0, y0), (v, size) in tiles.items():
        for dx, dy, side, opp in ((size, 0, '+x', '-x'), (0, size, '+y', '-y')):
            nb = (x0 + dx, y0 + dy)
            if nb not in tiles or tiles[nb][1] != size:
                continue
            a, b = _edge(v, side, size / 2), _edge(tiles[nb][0], opp, size / 2)
            if a is None or b is None or np.std(a - b) > 0.5:   # not a real shared edge: leave unlinked
                continue
            d = float(np.median(a - b))
            links[(x0, y0)].append((nb, d))
            links[nb].append(((x0, y0), -d))
    z = {}
    for start in sorted(tiles, key=lambda t: -len(links[t])):
        if start in z:
            continue
        z[start], todo = 0.0, [start]
        while todo:
            t = todo.pop()
            for nb, d in links[t]:
                if nb not in z:
                    z[nb] = z[t] + d
                    todo.append(nb)
    return z


def colour_map(m, streams, entries, rect, px_per_m, textures, progress=lambda f: None, cancel=None):
    """Top-down RGB image of the map's ground, painted from the baked colour maps of the lod / slod tiles.

    entries: [(stream index, name)], rect: (x0, y0, x1, y1) in metres. Row 0 of the image is the north edge."""
    import numpy as np
    x0, y0, x1, y1 = rect
    W, H = max(1, int((x1 - x0) * px_per_m)), max(1, int((y1 - y0) * px_per_m))
    out = np.full((H, W, 3), (70, 74, 80), np.uint8)
    painters = [(tile_rect(n), i) for i, n in entries]
    painters = [(r, i) for r, i in painters if r and r[0] != 'hd']
    painters.sort(key=lambda p: -p[0][3])           # coarse first, finer on top
    for k, (rect_, i) in enumerate(painters):
        if cancel and cancel.is_set():
            break
        progress(k / max(1, len(painters)))
        mdl = decode_model(m, streams, i)
        if not mdl or not mdl.get('materials') or not mdl['materials'][0].get('tex'):
            continue
        sol = uv_fit(mdl)
        img = textures.image(mdl['materials'][0]['tex']) if sol is not None else None
        if img is None:
            continue
        tex = np.asarray(img.convert('RGB'))
        kind, tx, ty, size = placed(rect_, sol)
        c0, c1 = int((tx - x0) * px_per_m), int((tx + size - x0) * px_per_m)
        r0, r1 = int((y1 - ty - size) * px_per_m), int((y1 - ty) * px_per_m)
        c0, c1, r0, r1 = max(c0, 0), min(c1, W), max(r0, 0), min(r1, H)
        if c0 >= c1 or r0 >= r1:
            continue
        lx = x0 + (np.arange(c0, c1) + 0.5) / px_per_m - (tx + size / 2)
        ly = y1 - (np.arange(r0, r1) + 0.5) / px_per_m - (ty + size / 2)
        gx, gy = np.meshgrid(lx, ly)
        u = gx * sol[0, 0] + gy * sol[1, 0] + sol[2, 0]
        w = gx * sol[0, 1] + gy * sol[1, 1] + sol[2, 1]
        px = tex[np.clip((w % 1.0 * tex.shape[0]).astype(int), 0, tex.shape[0] - 1),
                 np.clip((u % 1.0 * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)]
        has = px.astype(np.int32).sum(2) > 12           # pure black = nothing baked there: keep the coarser layer
        out[r0:r1, c0:c1][has] = px[has]
    return out


def quat_matrix(q):
    """3x3 rotation for a placement quaternion (x, y, z, w). The file stores the inverse rotation."""
    import numpy as np
    x, y, z, w = -q[0], -q[1], -q[2], q[3]
    n = x * x + y * y + z * z + w * w
    if not 0.5 < n < 2:
        return np.eye(3)
    s = 2 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def write_glb(path, tiles, placed_rows, objects, textures, ground, rect, progress=lambda f: None, cancel=None):
    """Binary glTF of a map: terrain tiles [(name, local verts, tris, (x, y, z))] painted with the ground image,
    and placed objects (find_placements rows; objects = {hash: (m, streams, i)}). Each model is stored once and
    every placed copy is a node pointing at it, so repeated props cost almost nothing. Z-up is turned to glTF's Y-up
    on the root node. -> dict(objects, models, verts, tris)."""
    import io
    import json
    import numpy as np
    blob, views, accs, meshes, nodes, mats, images, texs = bytearray(), [], [], [], [], [], [], []
    mat_of, mesh_of = {}, {}

    def view(data, target=None):
        blob.extend(b'\0' * (-len(blob) % 4))
        views.append(dict(buffer=0, byteOffset=len(blob), byteLength=len(data), **({'target': target} if target else {})))
        blob.extend(data)
        return len(views) - 1

    def acc(a, typ, ctype, target):
        a = np.ascontiguousarray(a)
        d = dict(bufferView=view(a.tobytes(), target), componentType=ctype, count=len(a), type=typ)
        if typ == 'VEC3':
            d.update(min=a.min(0).tolist(), max=a.max(0).tolist())
        accs.append(d)
        return len(accs) - 1

    def material(key, colour, img=None, glass=False):
        if key not in mat_of:
            pbr = {'baseColorFactor': [c / 255 for c in colour] + [0.35 if glass else 1.0], 'metallicFactor': 0.0, 'roughnessFactor': 0.9}
            if img is not None:
                buf = io.BytesIO()
                img.save(buf, 'PNG')
                images.append(dict(bufferView=view(buf.getvalue()), mimeType='image/png'))
                texs.append(dict(source=len(images) - 1, sampler=0))
                pbr.update(baseColorTexture={'index': len(texs) - 1}, baseColorFactor=[1.0, 1.0, 1.0, 1.0])
            mats.append(dict(name=key, pbrMetallicRoughness=pbr, doubleSided=True, **({'alphaMode': 'BLEND'} if glass else {})))
            mat_of[key] = len(mats) - 1
        return mat_of[key]

    def mesh(name, v, uv, groups):
        attrs = {'POSITION': acc(v.astype('<f4'), 'VEC3', 5126, 34962)}
        if uv is not None:
            attrs['TEXCOORD_0'] = acc(uv.astype('<f4'), 'VEC2', 5126, 34962)
        prims = [dict(attributes=attrs, indices=acc(idx.astype('<u4').ravel(), 'SCALAR', 5125, 34963),
                      **({'material': mi} if mi is not None else {})) for idx, mi in groups]
        meshes.append(dict(name=name, primitives=prims))
        return len(meshes) - 1

    nt = nv = n_obj = 0
    gm = material('ground', PLAIN, ground) if ground is not None else None
    for k, (name, v, t, (tx, ty, tz)) in enumerate(tiles):
        if cancel and cancel.is_set():
            break
        progress(0.7 + 0.1 * k / max(1, len(tiles)))
        uv = None
        if ground is not None:      # image row 0 is the north edge; glTF puts v = 0 at the top
            uv = np.c_[(v[:, 0] + tx - rect[0]) / (rect[2] - rect[0]), (rect[3] - v[:, 1] - ty) / (rect[3] - rect[1])]
        nodes.append(dict(name=name, mesh=mesh(name, v, uv, [(t, gm)]), translation=[float(tx), float(ty), float(tz)]))
        nt += len(t)
        nv += len(v)
    for k, (h, pos, q, sxy, sz, _) in enumerate(placed_rows):
        if cancel and cancel.is_set():
            break
        progress(0.8 + 0.2 * k / max(1, len(placed_rows)))
        if h not in mesh_of:
            mesh_of[h] = None
            mdl = decode_model(*objects[h])
            if mdl:
                t, tm = np.asarray(mdl['tris'], np.int64), np.asarray(mdl['tri_mat'] or [0] * len(mdl['tris']))
                groups = []
                for mi, mat in enumerate(mdl['materials'] or [{}]):
                    sel = t[tm == mi]
                    if not len(sel):
                        continue
                    if textures is None:
                        groups.append((sel, None))
                        continue
                    tex = mat.get('tex') if not mat.get('paint') else None
                    img = textures.image(tex) if tex else None
                    key = f't{tex:016x}' if img is not None else 'c%02x%02x%02x' % material_colour(mat) + ('g' if mat.get('glass') else '')
                    groups.append((sel, material(key, material_colour(mat), img, bool(mat.get('glass')))))
                mesh_of[h] = (mesh((mdl['name'] or 'model').split('.')[0], np.asarray(mdl['verts']), np.asarray(mdl['uvs']), groups),
                              len(t), len(mdl['verts']))
        if not mesh_of[h]:
            continue
        mi, ntri, nvert = mesh_of[h]
        node = dict(mesh=mi, translation=[float(c) for c in pos])
        if abs(q[3]) < 0.99999:
            node['rotation'] = [-float(q[0]), -float(q[1]), -float(q[2]), float(q[3])]   # stored as the inverse
        if sxy != 1 or sz != 1:
            node['scale'] = [float(sxy), float(sxy), float(sz)]
        nodes.append(node)
        nt += ntri
        nv += nvert
        n_obj += 1
    if len(blob) >= 1 << 32:
        raise ValueError('more than 4 GB of mesh data: export a smaller area')
    s = 0.5 ** 0.5
    nodes.append(dict(name='map', rotation=[-s, 0.0, 0.0, s], children=list(range(len(nodes)))))  # Z-up -> Y-up
    doc = dict(asset=dict(version='2.0', generator='TTPF Studio'), scene=0, scenes=[dict(nodes=[len(nodes) - 1])],
               nodes=nodes, meshes=meshes, accessors=accs, bufferViews=views, buffers=[dict(byteLength=len(blob))])
    if mats:
        doc['materials'] = mats
    if images:
        doc.update(images=images, textures=texs, samplers=[dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)])
    js = json.dumps(doc, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    blob.extend(b'\0' * (-len(blob) % 4))
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack('<I4s', len(js), b'JSON') + js)
        f.write(struct.pack('<I4s', len(blob), b'BIN\0'))
        f.write(blob)
    progress(1.0)
    return dict(objects=n_obj, models=len(meshes), verts=nv, tris=nt)


def export_map(m, streams, entries, path, textures=None, px_per_m=2.0, progress=lambda f: None, cancel=None,
               placements=None, objects=None, area=None):
    """Stitch every terrain tile into one OBJ in map coordinates (+ .mtl and a top-down colour PNG).

    entries: [(stream index, name)] from find_models. Uses the _hd meshes, and _lod ones where a tile has no _hd.
    area: (x0, y0, x1, y1) in metres to export only part of the map (see area_rect).
    placements: rows from find_placements; they give the tiles' real heights, and with
    objects ({model hash: (m, streams, stream index)}, from any archive) the buildings and props are added.
    Returns dict(tiles, objects, verts, tris, size=(metres x, metres y), png) or None when there are no terrain tiles."""
    import os
    import numpy as np
    want = ('hd', 'lod')
    best = {}
    for i, n in entries:
        r, mdl = tile_rect(n), None
        if not r or r[0] not in want:
            continue
        if area and not (area[0] - 128 < r[1] < area[2] + 128 and area[1] - 128 < r[2] < area[3] + 128):
            continue                            # (a lod tile can sit up to one tile off its name)
        if r[0] == 'lod':                       # small; decoded now because its UVs say where it goes
            mdl = decode_model(m, streams, i)
            if not mdl:
                continue
            r = placed(r, uv_fit(mdl))
        if area and not (area[0] <= r[1] < area[2] and area[1] <= r[2] < area[3]):
            continue
        key = r[1:]
        if key not in best or want.index(r[0]) < want.index(best[key][0]):
            best[key] = (r[0], i, mdl, n)
    if not best:
        return None
    where = {h: pos for h, pos, *_ in placements or ()}
    tiles, meshes, real = {}, {}, {}
    for k, ((x0, y0, size), (kind, i, mdl, n)) in enumerate(sorted(best.items())):
        if cancel and cancel.is_set():
            return None
        progress(0.5 * k / len(best))
        mdl = mdl or decode_model(m, streams, i)
        if not mdl:
            continue
        v = np.asarray(mdl['verts'], np.float32)
        meshes[(x0, y0)] = (v, np.asarray(mdl['tris'], np.int32), size)
        tiles[(x0, y0)] = (v, size)
        pos = where.get(joaat(n.split('.')[0]))
        if pos:
            real[(x0, y0)] = pos
    z = lift_tiles(tiles)
    if real:                                    # placed tiles: exact spot and height; the rest follow their neighbours
        shift = float(np.median([p[2] - z.get(t, 0.0) for t, p in real.items()]))
        z = {t: zz + shift for t, zz in z.items()}
    xs = [x for x, y in meshes] + [x + s for (x, y), (_, _, s) in meshes.items()]
    ys = [y for x, y in meshes] + [y + s for (x, y), (_, _, s) in meshes.items()]
    rect = (min(xs), min(ys), max(xs), max(ys))
    base = os.path.splitext(path)[0]
    stem = os.path.basename(base)
    png = ground = None
    glb = path.lower().endswith('.glb')
    if textures is not None:
        px = min(px_per_m, 8192 / max(rect[2] - rect[0], rect[3] - rect[1]))
        img = colour_map(m, streams, entries, rect, px, textures, lambda f: progress(0.5 + 0.2 * f), cancel)
        from PIL import Image
        ground = Image.fromarray(img)
        if not glb:
            png = stem + '_colour.png'
            ground.save(os.path.join(os.path.dirname(path) or '.', png))
    # buildings and props: the most detailed placed version of every non-terrain model we can find, inside the map
    todo = [r for r in placements or () if r[5] and objects and r[0] in objects
            and rect[0] <= r[1][0] <= rect[2] and rect[1] <= r[1][1] <= rect[3]]
    if glb:
        tiles = [(f'tile_{int(x0)}_{int(y0)}', v, t, (*real.get((x0, y0), (x0 + size / 2, y0 + size / 2, 0))[:2], z.get((x0, y0), 0.0)))
                 for (x0, y0), (v, t, size) in meshes.items()]
        r = write_glb(path, tiles, todo, objects, textures, ground, rect, progress, cancel)
        return dict(tiles=len(meshes), size=(rect[2] - rect[0], rect[3] - rect[1]), png=None, **r)
    mtl = {'ground': f'Kd 1 1 1\nmap_Kd {png}\n'} if png else {}
    nv = nvt = nt = n_obj = 0
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {stem} - map exported by TTPF Studio ({len(meshes)} terrain tiles, Z up, metres)\n')
        if textures is not None:
            f.write(f'mtllib {stem}.mtl\n')
        if png:
            f.write('usemtl ground\n')
        for k, ((x0, y0), (v, t, size)) in enumerate(meshes.items()):
            if cancel and cancel.is_set():
                break
            progress(0.7 + 0.1 * k / len(meshes))
            px_, py_, _ = real.get((x0, y0), (x0 + size / 2, y0 + size / 2, 0))
            w = v.astype(np.float64) + (px_, py_, z.get((x0, y0), 0.0))
            f.write(f'o tile_{int(x0)}_{int(y0)}\n')
            f.write(''.join(f'v {a:.3f} {b:.3f} {c:.3f}\n' for a, b, c in w))
            if png:
                uu = (w[:, 0] - rect[0]) / (rect[2] - rect[0])
                vv = (w[:, 1] - rect[1]) / (rect[3] - rect[1])
                f.write(''.join(f'vt {a:.6f} {b:.6f}\n' for a, b in zip(uu, vv)))
                f.write(''.join(f'f {a}/{b} {c}/{d} {e}/{g}\n' for a, c, e, b, d, g in
                                np.c_[t + nv + 1, t + nvt + 1].tolist()))
                nvt += len(v)
            else:
                f.write(''.join(f'f {a} {b} {c}\n' for a, b, c in (t + nv + 1).tolist()))
            nv += len(v)
            nt += len(t)
        cache = {}
        for k, (h, pos, q, sxy, sz, _) in enumerate(todo):
            if cancel and cancel.is_set():
                break
            progress(0.8 + 0.2 * k / len(todo))
            if h not in cache:
                om, ost, oi = objects[h]
                mdl = decode_model(om, ost, oi)
                cache[h] = mdl and (np.asarray(mdl['verts']), np.asarray(mdl['uvs']), np.asarray(mdl['tris'], np.int64),
                                    np.asarray(mdl['tri_mat'] or [0] * len(mdl['tris'])), mdl['materials'], mdl['name'])
            if not cache[h]:
                continue
            v, uv, t, tm, mats, name = cache[h]
            w = (v * (sxy, sxy, sz)) @ quat_matrix(q).T + pos
            f.write(f'o {(name or "object").split(".")[0]}_{n_obj}\n')
            f.write(''.join(f'v {a:.3f} {b:.3f} {c:.3f}\n' for a, b, c in w))
            f.write(''.join(f'vt {a:.5f} {1 - b:.5f}\n' for a, b in uv))
            for mi, mat in enumerate(mats or [{}]):
                sel = t[tm == mi]
                if not len(sel):
                    continue
                if textures is not None:
                    mn = f't{mat["tex"]:016x}' if mat.get('tex') and not mat.get('paint') and not mat.get('glass') else 'c%02x%02x%02x' % material_colour(mat)
                    if mn not in mtl:
                        r_, g_, b_ = (x / 255 for x in material_colour(mat))
                        mtl[mn] = f'Kd {r_:.3f} {g_:.3f} {b_:.3f}\n' + ('d 0.35\n' if mat.get('glass') else '')
                        img = textures.image(mat['tex']) if mn[0] == 't' else None
                        if img:
                            img.save(os.path.join(os.path.dirname(path) or '.', f'tex_{mat["tex"]:016x}.png'))
                            mtl[mn] = f'Kd 1 1 1\nmap_Kd tex_{mat["tex"]:016x}.png\n'
                    f.write(f'usemtl {mn}\n')
                f.write(''.join(f'f {a}/{b} {c}/{d} {e}/{g}\n' for a, c, e, b, d, g in
                                np.c_[sel + nv + 1, sel + nvt + 1].tolist()))
            nv += len(v)
            nvt += len(uv)
            nt += len(t)
            n_obj += 1
    if textures is not None:
        with open(base + '.mtl', 'w', encoding='utf-8') as f:
            f.writelines(f'newmtl {k}\n{body}\n' for k, body in mtl.items())
    progress(1.0)
    return dict(tiles=len(meshes), objects=n_obj, verts=nv, tris=nt, size=(rect[2] - rect[0], rect[3] - rect[1]), png=png)
