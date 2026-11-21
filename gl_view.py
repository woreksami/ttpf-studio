"""GPU model viewer for the 3D panel: OpenGL 3.3 inside a Tk frame (pyopengltk).

Every material's colour map is sampled per pixel with mipmaps and anisotropic filtering, lit with smooth normals,
and the frame is drawn 4x multisampled. Redraws only happen when something changes (drag, zoom, new model), so it
is as smooth as the screen and idle otherwise. If OpenGL is not available, `available()` is False and the caller
keeps using the software renderer in models.render.
"""
import ctypes

import numpy as np

try:
    from OpenGL import GL
    from pyopengltk import OpenGLFrame
except Exception:            # missing packages or no OpenGL: the software preview is used instead
    GL = OpenGLFrame = None

VS = """#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 nrm;
layout(location = 2) in vec2 uv;
uniform mat4 mvp;
uniform mat4 model;
out vec3 vN;
out vec2 vUV;
out vec3 vP;
void main() {
    vN = mat3(model) * nrm;
    vP = (model * vec4(pos, 1.0)).xyz;
    vUV = uv;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""
FS = """#version 330 core
in vec3 vN;
in vec2 vUV;
in vec3 vP;
uniform sampler2D tex;
uniform int useTex;
uniform vec3 base;
uniform float alpha;
uniform vec3 eye;
uniform int cutout;
uniform float gloss;                                   // 0 matte (trim, rubber, cloth) .. 1 clear-coated paint
out vec4 frag;
void main() {
    vec3 n = normalize(vN);
    vec3 v = normalize(eye - vP);
    if (dot(n, v) < 0.0) n = -n;                       // two-sided: light the side we see
    vec4 c = useTex == 1 ? texture(tex, vUV) : vec4(base, 1.0);
    if (cutout == 1 && c.a < 0.35) discard;           // transparent parts of decals, blades, light glass
    vec3 key = normalize(vec3(-0.45, 0.75, 0.55));
    vec3 fill = normalize(vec3(0.6, 0.2, -0.75));
    vec3 h = normalize(key + v);
    float lit = 0.30 + 0.70 * max(dot(n, key), 0.0) + 0.22 * max(dot(n, fill), 0.0);
    float spec = (0.18 + 0.9 * gloss) * pow(max(dot(n, h), 0.0), mix(40.0, 160.0, gloss));
    float rim = 0.12 * pow(1.0 - max(dot(n, v), 0.0), 3.0);
    // glossy surfaces reflect a soft studio sky: bright above, dark floor, stronger at grazing angles (Fresnel)
    vec3 r = reflect(-v, n);
    vec3 env = mix(vec3(0.06, 0.065, 0.075), vec3(0.78, 0.82, 0.88), smoothstep(-0.15, 0.55, r.y));
    float fres = 0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    vec3 col = c.rgb * lit * (1.0 - 0.25 * gloss) + env * gloss * (0.12 + 0.6 * fres) + vec3(spec + rim * (1.0 - gloss));
    frag = vec4(pow(col, vec3(1.0)), alpha * (useTex == 1 ? max(c.a, 0.0) : 1.0));
}
"""
BG_VS = """#version 330 core
const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out float t;
void main() { gl_Position = vec4(p[gl_VertexID], 0.0, 1.0); t = p[gl_VertexID].y * 0.5 + 0.5; }
"""
BG_FS = """#version 330 core
in float t;
uniform vec3 top;
uniform vec3 bot;
out vec4 frag;
void main() { frag = vec4(mix(bot, top, clamp(t, 0.0, 1.0)), 1.0); }
"""


def available():
    return GL is not None


def _perspective(fovy, aspect, near, far):
    f = 1 / np.tan(np.radians(fovy) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]], np.float32)


def _rot(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], np.float32)
    rx = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], np.float32)
    return rx @ ry


def smooth_normals(v, t):
    fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, t[:, k], fn)
    return (vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def prepare(model, textures, paint, material_colour):
    """CPU side of loading a model (safe on a worker thread): vertex arrays, triangles per material, texture images."""
    v = np.asarray(model['verts'], np.float32)
    t = np.asarray(model['tris'], np.uint32)
    lo, hi = np.percentile(v, 0.5, axis=0), np.percentile(v, 99.5, axis=0)
    v = (v - (lo + hi) / 2)[:, [0, 2, 1]] * np.array([1, 1, -1], np.float32)      # game is Z-up, the view is Y-up
    ext = (hi - lo)[[0, 2, 1]]
    uv = np.asarray(model.get('uvs') or np.zeros((len(v), 2)), np.float32)
    if len(uv) != len(v):
        uv = np.zeros((len(v), 2), np.float32)
    n = smooth_normals(v, t.astype(np.int64))
    tm = np.asarray(model.get('tri_mat') or [0] * len(t))
    parts = []
    for mi, mat in enumerate(model.get('materials') or [{}]):
        sel = t[tm == mi]
        if not len(sel):
            continue
        img = None
        if textures is not None and mat.get('tex') and not mat.get('paint') and not mat.get('glass'):
            img = textures.image(mat['tex'])
            if img is not None:
                img = img.convert('RGBA')
                if max(img.size) > 2048:
                    img.thumbnail((2048, 2048))
        colour = np.array(material_colour(mat, paint), np.float32) / 255
        gloss = 1.0 if mat.get('paint') else 0.8 if mat.get('rgb') in ((178, 182, 188), (205, 208, 214)) else 0.0  # paint, metal
        parts.append(dict(idx=np.ascontiguousarray(sel.ravel()), img=img, colour=colour, gloss=gloss,
                          glass=bool(mat.get('glass')), alpha=bool(mat.get('alpha')), tex=mat.get('tex')))
    inter = np.ascontiguousarray(np.hstack([v, n, uv]).astype(np.float32))
    return dict(inter=inter, parts=parts, ext=ext)


class ModelView(OpenGLFrame if OpenGLFrame else object):
    """Orbit view of one model. set_scene(prepared) to show a model, set_camera(yaw, pitch, zoom) to move."""

    def __init__(self, master, bg=(24, 26, 31), **kw):
        super().__init__(master, **kw)
        self.bg = np.array(bg, np.float32) / 255
        self.yaw, self.pitch, self.zoom = 0.8, 0.35, 1.0
        self.pan = (0.0, 0.0)                       # camera shift, in view heights (right-drag)
        self.scene = self.pending = None
        self.ready = False
        self.queued = False
        self.texcache = {}
        self.msaa = None

    # ---- GL setup
    def _program(self, vs, fs):
        def comp(src, kind):
            s = GL.glCreateShader(kind)
            GL.glShaderSource(s, src)
            GL.glCompileShader(s)
            if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
            return s
        p = GL.glCreateProgram()
        GL.glAttachShader(p, comp(vs, GL.GL_VERTEX_SHADER))
        GL.glAttachShader(p, comp(fs, GL.GL_FRAGMENT_SHADER))
        GL.glLinkProgram(p)
        if not GL.glGetProgramiv(p, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(p).decode())
        return p

    def initgl(self):
        try:
            self._init()
        except Exception:                               # old driver / no OpenGL 3.3: the caller falls back
            self.failed = True

    def _init(self):
        if not self.ready:
            self.prog = self._program(VS, FS)
            self.bgprog = self._program(BG_VS, BG_FS)
            self.bgvao = GL.glGenVertexArrays(1)
            self.u = {k: GL.glGetUniformLocation(self.prog, k) for k in ('mvp', 'model', 'tex', 'useTex', 'base', 'alpha', 'eye', 'cutout', 'gloss')}
            self.ready = True
        self._make_msaa()

    def _make_msaa(self):
        """4x multisampled off-screen target, resolved to the window each frame (smooth edges, no jaggies)."""
        w, h = max(1, self.winfo_width()), max(1, self.winfo_height())
        if self.msaa and self.msaa[3:] == (w, h):
            return
        if self.msaa:
            GL.glDeleteFramebuffers(1, [self.msaa[0]])
            GL.glDeleteRenderbuffers(2, [self.msaa[1], self.msaa[2]])
        fb = GL.glGenFramebuffers(1)
        col, dep = GL.glGenRenderbuffers(2)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, col)
        GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, 4, GL.GL_RGBA8, w, h)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, dep)
        GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, 4, GL.GL_DEPTH_COMPONENT24, w, h)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, col)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, dep)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.msaa = (fb, col, dep, w, h)

    # ---- scene
    def set_scene(self, prepared):
        self.pending = prepared
        self.request()

    def set_camera(self, yaw=None, pitch=None, zoom=None, pan=None):
        self.pan = self.pan if pan is None else pan
        self.yaw = self.yaw if yaw is None else yaw
        self.pitch = self.pitch if pitch is None else pitch
        self.zoom = self.zoom if zoom is None else zoom
        self.request()

    def request(self):
        """Draw once at the next idle moment (many requests in a row become one frame)."""
        if not self.queued:
            self.queued = True
            self.after_idle(self._frame)

    def _frame(self):
        self.queued = False
        if self.winfo_ismapped():
            self._display()

    def _texture(self, key, img):
        if key in self.texcache:
            return self.texcache[key]
        tid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
        data = img.tobytes()
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, img.width, img.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        try:
            GL.glTexParameterf(GL.GL_TEXTURE_2D, 0x84FE, 8.0)    # GL_TEXTURE_MAX_ANISOTROPY: sharp at grazing angles
        except Exception:
            pass
        self.texcache[key] = tid
        return tid

    def _upload(self, prep):
        if self.scene:
            GL.glDeleteVertexArrays(1, [self.scene['vao']])
            GL.glDeleteBuffers(1 + len(self.scene['parts']), [self.scene['vbo']] + [p['ebo'] for p in self.scene['parts']])
        if len(self.texcache) > 64:                      # keep GPU memory in check across many models
            GL.glDeleteTextures(len(self.texcache), list(self.texcache.values()))
            self.texcache.clear()
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, prep['inter'].nbytes, prep['inter'], GL.GL_STATIC_DRAW)
        stride = 8 * 4
        for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 2, 24)):
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, stride, ctypes.c_void_p(off))
        parts = []
        for p in prep['parts']:
            ebo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, p['idx'].nbytes, p['idx'], GL.GL_STATIC_DRAW)
            tid = self._texture(p['tex'], p['img']) if p['img'] is not None else None
            parts.append(dict(ebo=ebo, count=len(p['idx']), tex=tid, colour=p['colour'], glass=p['glass'],
                              alpha=p.get('alpha', False), gloss=p.get('gloss', 0.0)))
        GL.glBindVertexArray(0)
        self.scene = dict(vao=vao, vbo=vbo, parts=parts, ext=prep['ext'])

    # ---- frame
    def redraw(self):
        if getattr(self, 'failed', False) or not self.ready:
            return
        try:
            self._redraw()
        except Exception:
            self.failed = True

    def _redraw(self):
        self._make_msaa()
        if self.pending is not None:
            self._upload(self.pending)
            self.pending = None
        w, h = self.msaa[3], self.msaa[4]
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.msaa[0])
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.bgprog)
        GL.glUniform3f(GL.glGetUniformLocation(self.bgprog, 'top'), *(self.bg + 0.10))
        GL.glUniform3f(GL.glGetUniformLocation(self.bgprog, 'bot'), *(self.bg * 0.7))
        GL.glBindVertexArray(self.bgvao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        if self.scene:
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            ext = self.scene['ext']
            across = float(np.hypot(ext[0], ext[2])) or 1.0
            updown = float(ext[1] * abs(np.cos(self.pitch)) + across * abs(np.sin(self.pitch))) or 1.0
            aspect = w / h
            fov = 30.0
            # distance at which the model's footprint fills ~88% of the view, like the software preview
            half_h = max(updown / 2 / 0.88, across / 2 / 0.92 / aspect)
            dist = half_h / np.tan(np.radians(fov) / 2) / self.zoom + across / 2
            proj = _perspective(fov, aspect, max(0.01, dist / 100), dist * 4 + across)
            view = np.eye(4, dtype=np.float32)
            view[2, 3] = -dist
            vis_h = 2 * (dist - across / 2) * np.tan(np.radians(fov) / 2)     # visible height at the model
            view[0, 3], view[1, 3] = self.pan[0] * vis_h, self.pan[1] * vis_h
            model = _rot(self.yaw, self.pitch)
            mvp = proj @ view @ model
            GL.glUseProgram(self.prog)
            GL.glUniformMatrix4fv(self.u['mvp'], 1, True, mvp)
            GL.glUniformMatrix4fv(self.u['model'], 1, True, model)
            GL.glUniform3f(self.u['eye'], 0.0, 0.0, float(dist))    # lighting is in rotated-model space; the camera sits on +z
            GL.glUniform1i(self.u['tex'], 0)
            GL.glBindVertexArray(self.scene['vao'])
            GL.glActiveTexture(GL.GL_TEXTURE0)
            for glass_pass in (False, True):             # opaque first, then see-through glass on top
                if glass_pass:
                    GL.glEnable(GL.GL_BLEND)
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                    GL.glDepthMask(False)
                for p in self.scene['parts']:
                    if (p['glass'] or (p['alpha'] and p['tex'] is not None)) != glass_pass:
                        continue
                    GL.glUniform1i(self.u['useTex'], 1 if p['tex'] else 0)
                    GL.glUniform1i(self.u['cutout'], 1 if p['alpha'] and p['tex'] else 0)
                    GL.glUniform1f(self.u['gloss'], 0.9 if p['glass'] else 0.8 if p['alpha'] else p['gloss'])   # glass and lenses reflect
                    GL.glUniform3f(self.u['base'], *p['colour'])
                    GL.glUniform1f(self.u['alpha'], 0.55 if p['glass'] else 1.0)
                    if p['tex']:
                        GL.glBindTexture(GL.GL_TEXTURE_2D, p['tex'])
                    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, p['ebo'])
                    GL.glDrawElements(GL.GL_TRIANGLES, p['count'], GL.GL_UNSIGNED_INT, None)
            GL.glDepthMask(True)
            GL.glDisable(GL.GL_BLEND)
            GL.glBindVertexArray(0)
        # resolve the multisampled frame to the window
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.msaa[0])
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
        GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def snapshot(self):
        """The last frame as a PIL image (for tests and 'save picture')."""
        from PIL import Image
        w, h = self.msaa[3], self.msaa[4]
        self.tkMakeCurrent()
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glReadBuffer(GL.GL_FRONT)
        data = GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        return Image.frombytes('RGB', (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)


def check():
    """Open a GPU view, draw a test triangle and read it back. -> 0 if the OpenGL path works, else 1.
    (`ttpf_tool.py --glcheck`, used to test builds.)"""
    if not available():
        return 1
    import tkinter as tk
    from PIL import Image
    root = tk.Tk()
    root.geometry('240x180+40+40')
    view = ModelView(root, width=240, height=180)
    view.pack(fill='both', expand=True)
    tri = dict(verts=[[-1, 0, -1], [1, 0, -1], [0, 0, 1]], tris=[[0, 1, 2]], uvs=[[0, 0], [1, 0], [0.5, 1]],
               materials=[{}], tri_mat=[0])
    result = []

    def go():
        try:
            view.set_scene(prepare(tri, None, (220, 40, 40), lambda m, p: p))
            view._display()
            view._display()
            img = view.snapshot()
            reds = sum(1 for px in img.convert('RGB').resize((48, 36), Image.NEAREST).getdata() if px[0] > 60 and px[0] > 2.5 * px[1])
            result.append(0 if not getattr(view, 'failed', False) and view.scene and reds > 3 else 1)
        except Exception:
            result.append(1)
        root.destroy()
    root.after(600, go)
    root.mainloop()
    return result[0] if result else 1
