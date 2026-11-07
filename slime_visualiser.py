#slime visualiser (pygame)
#only DRAWS things, it never changes your slimes, brains or world.
#
#what your world needs to give it:
#  slime  -> .x .y .radius  (required)
#            .angle (radians, where it is heading)  .turn (-1..1)  .energy (0..1)
#            .hue (0..1 colour)  .gulping (True/False)  .brain (feed_forward_network)
#  food    -> (x, y) tuples or objects with .x .y
#  hazards -> (x, y, radius) tuples or objects with .x .y .radius
#the arena is a rectangle at vis.arena, world coords == screen coords
#run this file directly to see a preview with placeholder random brains

import math
import random
import colorsys
import numpy as np
import pygame
from pygame import gfxdraw

#---------------- palette ----------------
BG = (246, 240, 255)
ARENA = (255, 252, 250)
ARENA_BORDER = (222, 206, 246)
INK = (70, 40, 90)
SOFT_INK = (140, 115, 165)
CARD = (255, 255, 255)
CARD_BORDER = (228, 216, 248)
SPARKLES = [(255, 138, 101), (255, 213, 79), (100, 181, 246), (102, 217, 142), (255, 128, 171)]
BOW = (232, 64, 64)
BOW_DARK = (178, 38, 52)
EYE = (68, 28, 78)

#---------------- small helpers ----------------
def lerp(a, b, t):
    return a + (b - a) * t

def mix(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))

def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c[:3])

def hue_to_rgb(h, s=0.58, v=0.95):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def ease_out_back(x):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2

def xy(o):
    return (o.x, o.y) if hasattr(o, "x") else (o[0], o[1])

def fill_poly(surf, pts, colour):
    pts = [(int(x), int(y)) for x, y in pts]
    gfxdraw.filled_polygon(surf, pts, colour)
    gfxdraw.aapolygon(surf, pts, colour)

def ellipse_pts(cx, cy, rx, ry, rot=0.0, n=22):
    cr, sr = math.cos(rot), math.sin(rot)
    pts = []
    for k in range(n):
        t = 2 * math.pi * k / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        pts.append((cx + x * cr - y * sr, cy + x * sr + y * cr))
    return pts

def sparkle_pts(cx, cy, size, rot=0.0, n=36):
    #four pointed curvy star, like the sparkles in the reference picture
    cr, sr = math.cos(rot), math.sin(rot)
    pts = []
    for k in range(n):
        t = 2 * math.pi * k / n
        c, s = math.cos(t), math.sin(t)
        x = size * math.copysign(abs(c) ** 3.0, c)
        y = size * math.copysign(abs(s) ** 3.0, s)
        pts.append((cx + x * cr - y * sr, cy + x * sr + y * cr))
    return pts

def blob_pts(cx, cy, r, sx, sy, wobble, phase, lean, n=44):
    #a little dome of jelly: round on top, flatter and wider at the bottom
    pts = []
    for k in range(n):
        t = 2 * math.pi * k / n
        c, s = math.cos(t), math.sin(t)
        rad = 1 + wobble * math.sin(3 * t + phase) + wobble * 0.6 * math.sin(5 * t - phase * 1.3)
        x = r * c * (1 - 0.08 * s) * rad * sx + r * lean * max(0.0, s)
        up = r * s * (0.72 + 0.28 * s) * rad * sy
        pts.append((cx + x, cy + 0.28 * r - up))
    return pts


class SlimeVisualiser:
    def __init__(self, size=(1180, 720)):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Slime Lab")
        self.clock = pygame.time.Clock()
        self.f_small = pygame.font.SysFont(None, 20)
        self.f_med = pygame.font.SysFont(None, 26)
        self.f_big = pygame.font.SysFont(None, 40)
        self.arena = pygame.Rect(20, 20, 760, 680)
        self.states = {}      #per slime animation state (spring, blink, pop in...)
        self.particles = []   #little sparkles that fly out when something is eaten
        self.twinkles = [
            (random.randint(self.arena.left + 20, self.arena.right - 20),
             random.randint(self.arena.top + 20, self.arena.bottom - 20),
             random.randint(6, 12), random.random() * 6.28, random.choice(SPARKLES))
            for _ in range(16)
        ]

    #---------------- public ----------------
    def burst(self, x, y, n=7):
        for _ in range(n):
            a = random.random() * 2 * math.pi
            sp = random.uniform(0.8, 2.4)
            self.particles.append({"x": x, "y": y, "vx": math.cos(a) * sp, "vy": math.sin(a) * sp - 1.0,
                                   "life": 38, "max": 38, "size": random.uniform(4, 8),
                                   "col": random.choice(SPARKLES), "rot": random.random()})

    def frame(self, slimes, food, hazards, champion=None, champion_state=None, input_labels=None,
              generation=None, best_hist=(), avg_hist=(), stats=()):
        t = pygame.time.get_ticks() / 1000.0
        scr = self.screen
        scr.fill(BG)
        pygame.draw.rect(scr, ARENA, self.arena, border_radius=26)
        pygame.draw.rect(scr, ARENA_BORDER, self.arena, width=3, border_radius=26)

        for (x, y, size, ph, col) in self.twinkles:
            a = int(50 + 40 * math.sin(t * 1.5 + ph))
            fill_poly(scr, sparkle_pts(x, y, size, 0.0), col + (a,))

        for f in food:
            x, y = xy(f)
            pulse = 1 + 0.12 * math.sin(t * 4 + x * 0.05)
            col = SPARKLES[int(x + y) % len(SPARKLES)]
            fill_poly(scr, sparkle_pts(x, y, 15 * pulse, t * 0.4), col)

        for h in hazards:
            x, y = xy(h)
            rad = h.radius if hasattr(h, "radius") else (h[2] if len(h) > 2 else 18)
            self._draw_hazard(x, y, rad, t)

        alive = {id(s) for s in slimes}
        self.states = {k: v for k, v in self.states.items() if k in alive}
        for s in sorted(slimes, key=lambda s: s.y):
            self._draw_slime(s, t, s is champion)

        for p in self.particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]; p["vy"] += 0.04; p["life"] -= 1; p["rot"] += 0.08
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            k = p["life"] / p["max"]
            fill_poly(scr, sparkle_pts(p["x"], p["y"], p["size"] * (0.4 + 0.6 * k), p["rot"]), p["col"] + (int(255 * k),))

        self._side_panel(champion, champion_state, input_labels, generation, best_hist, avg_hist, stats)
        pygame.display.flip()
        self.clock.tick(60)

    #---------------- things in the arena ----------------
    def _draw_hazard(self, x, y, rad, t):
        scr = self.screen
        pulse = 1 + 0.05 * math.sin(t * 5 + x)
        pts = []
        for k in range(24):
            a = 2 * math.pi * k / 24 + t * 0.3
            rr = rad * pulse * (1.0 if k % 2 == 0 else 0.68)
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
        fill_poly(scr, pts, (92, 78, 130))
        gfxdraw.filled_circle(scr, int(x), int(y), int(rad * 0.6), (110, 94, 150))
        for sgn in (-1, 1):
            gfxdraw.filled_ellipse(scr, int(x + sgn * rad * 0.22), int(y - rad * 0.05), max(2, int(rad * 0.11)), max(3, int(rad * 0.16)), (255, 255, 255))
        pygame.draw.line(scr, (255, 255, 255), (x - rad * 0.3, y - rad * 0.28), (x - rad * 0.08, y - rad * 0.18), 2)
        pygame.draw.line(scr, (255, 255, 255), (x + rad * 0.3, y - rad * 0.28), (x + rad * 0.08, y - rad * 0.18), 2)

    def _state(self, s):
        k = id(s)
        if k not in self.states:
            self.states[k] = {"r": s.radius, "seen": s.radius, "squash": 0.0, "vel": 0.0, "pop": 0.0,
                              "gulp": 0.0, "phase": random.random() * 6.28}
        return self.states[k]

    def _draw_slime(self, s, t, champion):
        scr = self.screen
        st = self._state(s)

        #animation state: smooth growth, jelly spring, pop in, detect eating by growth
        st["r"] += (s.radius - st["r"]) * 0.2
        st["pop"] = min(1.0, st["pop"] + 0.05)
        if s.radius > st["seen"] + 0.3:
            st["vel"] += 0.06
            self.burst(s.x, s.y - s.radius * 0.3, 5)
        st["seen"] = s.radius
        st["vel"] += -0.12 * st["squash"] - 0.10 * st["vel"]
        st["squash"] += st["vel"]
        st["gulp"] += ((1.0 if getattr(s, "gulping", False) else 0.0) - st["gulp"]) * 0.35

        r = st["r"] * ease_out_back(st["pop"])
        if r < 1:
            return
        cx, cy = float(s.x), float(s.y)
        sq = st["squash"] + 0.015 * math.sin(t * 3 + st["phase"])
        sx, sy = 1 + sq * 0.9, 1 - sq
        turn = float(getattr(s, "turn", 0.0))
        angle = float(getattr(s, "angle", 0.0))
        energy = float(getattr(s, "energy", 1.0))
        base = hue_to_rgb(getattr(s, "hue", 0.76))
        wob = 0.012 + 0.03 * abs(turn)

        #shadow
        gfxdraw.filled_ellipse(scr, int(cx), int(cy + 0.74 * r), int(0.85 * r * sx), max(2, int(0.17 * r)), (90, 60, 120, 45))

        #body: darker rim underneath, lighter body on top
        fill_poly(scr, blob_pts(cx, cy, r, sx, sy, wob, t * 4 + st["phase"], turn * 0.14), shade(base, 0.78))
        fill_poly(scr, blob_pts(cx, cy - 0.04 * r, r * 0.95, sx, sy, wob, t * 4 + st["phase"], turn * 0.14), base)

        #cheeks
        cheek = mix(base, (255, 150, 178), 0.45 + 0.3 * st["gulp"])
        for sgn in (-1, 1):
            gfxdraw.filled_ellipse(scr, int(cx + sgn * 0.48 * r * sx), int(cy + 0.14 * r * sy), max(2, int(0.1 * r)), max(2, int(0.07 * r)), cheek)

        #face looks where the slime is heading
        fx = cx + math.cos(angle) * 0.12 * r + turn * 0.06 * r
        fy = cy + math.sin(angle) * 0.07 * r
        blinking = (t * 0.9 + st["phase"] * 7) % 4.0 < 0.13
        ex, ew = 0.30 * r * sx, max(2, int(0.075 * r))
        eh = max(3, int(0.12 * r))
        if blinking:
            eh = max(1, int(eh * 0.2))
        for sgn in (-1, 1):
            gfxdraw.filled_ellipse(scr, int(fx + sgn * ex), int(fy - 0.05 * r * sy), ew, eh, EYE)
            gfxdraw.aaellipse(scr, int(fx + sgn * ex), int(fy - 0.05 * r * sy), ew, eh, EYE)

        my = fy + 0.20 * r * sy
        w = 0.17 * r * sx
        thick = max(2, int(r * 0.075))
        if st["gulp"] > 0.3:
            #mouth wide open with a little tongue
            ry = int((0.06 + 0.16 * st["gulp"]) * r)
            gfxdraw.filled_ellipse(scr, int(fx), int(my + ry * 0.4), max(3, int(0.15 * r)), max(2, ry), EYE)
            gfxdraw.filled_ellipse(scr, int(fx), int(my + ry * 0.9), max(2, int(0.08 * r)), max(1, int(ry * 0.45)), (255, 120, 150))
        else:
            depth = (0.10 if energy > 0.25 else -0.07) * r
            pts = [(fx + (k / 4 - 1) * w, my + depth * (1 - (k / 4 - 1) ** 2) * (1 if energy > 0.25 else 1)) for k in range(9)]
            pygame.draw.lines(scr, EYE, False, pts, thick)
            for p in (pts[0], pts[-1]):
                gfxdraw.filled_circle(scr, int(p[0]), int(p[1]), thick // 2, EYE)

        #glossy highlight
        fill_poly(scr, ellipse_pts(cx - 0.36 * r * sx, cy - 0.40 * r * sy, 0.20 * r, 0.10 * r, -0.6), (255, 255, 255, 190))
        gfxdraw.filled_circle(scr, int(cx - 0.08 * r * sx), int(cy - 0.52 * r * sy), max(1, int(0.035 * r)), (255, 255, 255, 190))

        #the current champion wears the bow
        if champion:
            bx, by = cx + 0.44 * r * sx + turn * 0.06 * r, cy - 0.62 * r * sy
            for sgn in (-1, 1):
                pts = ellipse_pts(bx + sgn * 0.30 * r, by + sgn * 0.05 * r, 0.31 * r, 0.17 * r, sgn * 0.5)
                fill_poly(scr, pts, BOW)
                pygame.draw.polygon(scr, BOW_DARK, [(int(a), int(b)) for a, b in pts], 1)
            gfxdraw.filled_circle(scr, int(bx), int(by), max(2, int(0.1 * r)), BOW_DARK)

    #---------------- side panel ----------------
    def _text(self, text, pos, font=None, colour=INK, center=False, right=False):
        img = (font or self.f_small).render(str(text), True, colour)
        rect = img.get_rect()
        if center: rect.center = pos
        elif right: rect.midright = pos
        else: rect.topleft = pos
        self.screen.blit(img, rect)

    def _card(self, rect, title):
        pygame.draw.rect(self.screen, CARD, rect, border_radius=18)
        pygame.draw.rect(self.screen, CARD_BORDER, rect, width=2, border_radius=18)
        self._text(title, (rect.x + 16, rect.y + 12), self.f_med, SOFT_INK)

    def _side_panel(self, champion, champion_state, input_labels, generation, best_hist, avg_hist, stats):
        x, w = 800, 360
        top = pygame.Rect(x, 20, w, 120)
        self._card(top, "Slime Lab")
        if generation is not None:
            self._text(f"Generation {generation}", (x + 16, 50), self.f_big)
        for i, (label, value) in enumerate(list(stats)[:3]):
            self._text(f"{label}: {value}", (x + 16, 88 + i * 15), self.f_small, SOFT_INK)
        self._draw_network(pygame.Rect(x, 155, w, 270), champion, champion_state, input_labels)
        self._draw_fitness(pygame.Rect(x, 440, w, 260), best_hist, avg_hist)

    def _draw_network(self, rect, champion, state, input_labels):
        self._card(rect, "Champion's brain")
        brain = getattr(champion, "brain", None) if champion is not None else None
        if brain is None or state is None:
            self._text("waiting for a champion...", rect.center, self.f_small, SOFT_INK, center=True)
            return
        out, acts, _ = brain.calculation(state)
        layers = [a.flatten() for a in acts] + [out.flatten()]
        pad_l, pad_r, pad_t, pad_b = 92, 100, 48, 16
        iw, ih = rect.w - pad_l - pad_r, rect.h - pad_t - pad_b
        cols = [rect.x + pad_l + i * iw / (len(layers) - 1) for i in range(len(layers))]
        pos = [[(cols[i], rect.y + pad_t + (j + 1) * ih / (len(a) + 1)) for j in range(len(a))] for i, a in enumerate(layers)]

        for i, W in enumerate(brain.weights):
            flow = np.abs(W * layers[i][np.newaxis, :])
            peak = flow.max() if flow.max() > 1e-9 else 1.0
            for j, a in enumerate(pos[i]):
                for k, b in enumerate(pos[i + 1]):
                    sgn = flow[k][j] / peak
                    col = mix(CARD, (70, 200, 130) if W[k][j] > 0 else (245, 105, 115), 0.12 + 0.88 * sgn)
                    pygame.draw.line(self.screen, col, a, b, 2 if sgn > 0.6 else 1)

        for i, layer in enumerate(pos):
            vals = np.abs(layers[i])
            peak = vals.max() if vals.max() > 1e-9 else 1.0
            for j, p in enumerate(layer):
                strength = vals[j] / peak
                fill = mix((240, 232, 252), (125, 62, 195), strength)
                if i == len(pos) - 1 and j == 1 and layers[i][1] > 0.5:
                    fill = (90, 200, 130)   #gulping
                rad = 7 if len(layer) <= 10 else 5
                gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), rad, fill)
                gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), rad, (160, 130, 200))

        labels = input_labels or [f"in {j}" for j in range(len(pos[0]))]
        for j, p in enumerate(pos[0]):
            self._text(labels[j] if j < len(labels) else f"in {j}", (p[0] - 12, p[1]), self.f_small, INK, right=True)
        names = ["Turn", "Gulp"]
        for j, p in enumerate(pos[-1]):
            v = layers[-1][j]
            self._text(f"{names[j] if j < 2 else 'out'} {v:+.2f}" if j == 0 else f"{names[j] if j < 2 else 'out'} {v:.2f}",
                       (p[0] + 14, p[1] - 17), self.f_small, INK)
            bar = pygame.Rect(int(p[0] + 14), int(p[1] + 1), 62, 6)
            pygame.draw.rect(self.screen, (238, 230, 250), bar, border_radius=3)
            if j == 0:   #turn bar grows left or right from the middle
                mid = bar.centerx
                wd = int(abs(v) * bar.w / 2)
                pygame.draw.rect(self.screen, (125, 62, 195), (mid if v >= 0 else mid - wd, bar.y, max(1, wd), bar.h), border_radius=3)
            else:
                pygame.draw.rect(self.screen, (90, 200, 130), (bar.x, bar.y, max(1, int(v * bar.w)), bar.h), border_radius=3)

    def _draw_fitness(self, rect, best, avg):
        self._card(rect, "Fitness per generation")
        inner = pygame.Rect(rect.x + 20, rect.y + 52, rect.w - 40, rect.h - 100)
        if len(best) < 2:
            self._text("waiting for generation 2...", rect.center, self.f_small, SOFT_INK, center=True)
            return
        lo = min(min(best), min(avg) if len(avg) else min(best))
        hi = max(max(best), max(avg) if len(avg) else max(best))
        if hi - lo < 1e-9:
            hi = lo + 1
        pygame.draw.line(self.screen, CARD_BORDER, inner.bottomleft, inner.bottomright, 2)
        pygame.draw.line(self.screen, CARD_BORDER, inner.bottomleft, inner.topleft, 2)

        def pts(series):
            n = len(series)
            return [(inner.x + i * inner.w / (n - 1), inner.bottom - (v - lo) / (hi - lo) * inner.h) for i, v in enumerate(series)]
        for series, col in ((avg, (70, 200, 130)), (best, (150, 80, 215))):
            if len(series) >= 2:
                p = pts(series)
                pygame.draw.lines(self.screen, col, False, p, 3)
                gfxdraw.filled_circle(self.screen, int(p[-1][0]), int(p[-1][1]), 5, col)
        self._text(f"best {best[-1]:.1f}", (rect.x + 20, rect.bottom - 34), self.f_small, (150, 80, 215))
        self._text(f"avg {avg[-1]:.1f}" if len(avg) else "", (rect.x + 120, rect.bottom - 34), self.f_small, (70, 170, 110))
        self._text(f"{lo:.0f} to {hi:.0f}", (rect.right - 16, rect.bottom - 26), self.f_small, SOFT_INK, right=True)


#---------------- preview with placeholder world ----------------
#everything below is ONLY so you can see the visuals: random brains, no evolution,
#placeholder sensors. replace it with your real world.

LABELS = ["Food angle", "Food dist", "Hazard angle", "Hazard dist", "Energy", "Wall dist"]

class DemoSlime:
    def __init__(self, brain, arena):
        self.brain = brain
        self.x = random.randint(arena.left + 60, arena.right - 60)
        self.y = random.randint(arena.top + 60, arena.bottom - 60)
        self.angle = random.random() * 2 * math.pi
        self.energy = 0.55
        self.hue = random.choice([0.76, 0.40, 0.92, 0.55, 0.08, 0.15])
        self.gulping = False
        self.turn = 0.0
        self.state = np.zeros((6, 1))
    @property
    def radius(self):
        return 16 + 18 * self.energy

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def sense(s, food, hazards, arena):
    def nearest(items):
        best, bd = (0.0, 1.0), 1e9
        for it in items:
            dx, dy = it[0] - s.x, it[1] - s.y
            d = math.hypot(dx, dy)
            if d < bd:
                bd, best = d, (wrap(math.atan2(dy, dx) - s.angle) / math.pi, min(d / 400, 1.0))
        return best
    fa, fd = nearest(food)
    ha, hd = nearest(hazards)
    wall = min(s.x - arena.left, arena.right - s.x, s.y - arena.top, arena.bottom - s.y) / 200
    return np.array([[fa], [fd], [ha], [hd], [s.energy], [min(wall, 1.0)]])

def main(frames=None, shot=None):
    import slime_brain
    vis = SlimeVisualiser()
    A = vis.arena
    def new_brain():
        b = slime_brain.feed_forward_network(2, 6, 2, [8, 8]); b.initialise_weights(); return b
    slimes = [DemoSlime(new_brain(), A) for _ in range(12)]
    rnd = lambda: (random.randint(A.left + 30, A.right - 30), random.randint(A.top + 30, A.bottom - 30))
    food = [rnd() for _ in range(16)]
    hazards = [rnd() + (random.randint(16, 24),) for _ in range(5)]
    gen, best, avg, n = 1, [], [], 0
    running = True
    while running and (frames is None or n < frames):
        n += 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        for i, s in enumerate(slimes):
            s.state = sense(s, food, hazards, A)
            turn, gulp = s.brain.action(s.state)
            s.turn, s.gulping = turn, gulp > 0.5
            s.angle += turn * 0.12
            s.x += math.cos(s.angle) * 1.6; s.y += math.sin(s.angle) * 1.6
            s.x = min(max(s.x, A.left + 25), A.right - 25); s.y = min(max(s.y, A.top + 25), A.bottom - 25)
            s.energy -= 0.0006 + (0.0003 if s.gulping else 0)
            for k, f in enumerate(food):
                if s.gulping and math.hypot(f[0] - s.x, f[1] - s.y) < s.radius + 10:
                    s.energy = min(1.0, s.energy + 0.18); food[k] = rnd()
            for h in hazards:
                if math.hypot(h[0] - s.x, h[1] - s.y) < s.radius + h[2]:
                    s.energy -= 0.003
            if s.energy <= 0:
                slimes[i] = DemoSlime(new_brain(), A)
        if n % 300 == 0:   #fake history just to preview the chart
            gen += 1
            best.append((best[-1] if best else 5) + random.uniform(0.5, 3)); avg.append((avg[-1] if avg else 1) + random.uniform(0, 2))
        champ = max(slimes, key=lambda s: s.energy)
        vis.frame(slimes, food, hazards, champion=champ, champion_state=champ.state, input_labels=LABELS,
                  generation=gen, best_hist=best, avg_hist=avg,
                  stats=[("Slimes alive", len(slimes)), ("Champion energy", f"{champ.energy:.2f}"), ("Food left", len(food))])
    if shot:
        pygame.image.save(vis.screen, shot)
    pygame.quit()

if __name__ == "__main__":
    main()
