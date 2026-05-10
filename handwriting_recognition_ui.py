import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, sys, math, time, threading, colorsys, random, string
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np

# ─── Optional PyTorch ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─── Colour Palette ───────────────────────────────────────────────────────────
C = {
    "bg":         "#0F1117",
    "bg2":        "#161A24",
    "bg3":        "#1E2333",
    "bg4":        "#252A3D",
    "card":       "#1A1F2E",
    "border":     "#2A3050",
    "border2":    "#384070",
    "accent":     "#4F8EF7",
    "accent2":    "#7B5CF0",
    "accent3":    "#36C7A0",
    "accent_dim": "#1A3A7A",
    "success":    "#36C7A0",
    "warning":    "#F59E0B",
    "error":      "#EF4444",
    "text":       "#E8EAF6",
    "text2":      "#9BA3C4",
    "text3":      "#5C6490",
    "white":      "#FFFFFF",
    "bar_bg":     "#1A2040",
    "canvas_bg":  "#FFFFFF",
    "tab_active": "#1E2333",
    "tab_idle":   "#161A24",
}

# ─── Model ────────────────────────────────────────────────────────────────────
CLASSES         = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best_model.pth"

if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.shortcut = (
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch))
                if stride != 1 or in_ch != out_ch else nn.Identity()
            )
        def forward(self, x):
            return F.relu(self.body(x) + self.shortcut(x))

    class HandwritingResNet(nn.Module):
        def __init__(self, num_classes=62):
            super().__init__()
            self.stem    = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                          nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.layer1  = nn.Sequential(ResidualBlock(32,  64,  2), ResidualBlock(64,  64))
            self.layer2  = nn.Sequential(ResidualBlock(64,  128, 2), ResidualBlock(128, 128))
            self.layer3  = nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256))
            self.pool    = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))
        def forward(self, x):
            return self.classifier(
                self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).view(x.size(0), -1))

def load_model():
    if not TORCH_AVAILABLE:
        return None, "PyTorch not installed — running in demo mode"
    if not CHECKPOINT_PATH.exists():
        return None, f"Checkpoint not found at {CHECKPOINT_PATH}"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = HandwritingResNet(62).to(device)
        state  = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_pil(pil_img):
    img = pil_img.convert("L")
    arr = np.array(img, dtype=np.float32)
    if arr.mean() > 128:
        arr = 255 - arr
    binary = arr > 30
    if binary.any():
        rows = np.any(binary, axis=1); cols = np.any(binary, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        arr = arr[r0:r1+1, c0:c1+1]
    crop   = Image.fromarray(arr.astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = np.array(crop, dtype=np.float32)
    canvas = (canvas / 255.0 - 0.1307) / 0.3081
    return torch.tensor(canvas).unsqueeze(0).unsqueeze(0)

def run_predict(model, pil_img):
    if model is None:
        idx  = random.randint(0, 61)
        p    = np.random.dirichlet(np.ones(62) * 0.3)
        p[idx] = random.uniform(0.5, 0.9); p /= p.sum()
        top5 = sorted(enumerate(p), key=lambda x: -x[1])[:5]
        return [(CLASSES[i], float(v)) for i, v in top5], 42.0
    device = next(model.parameters()).device
    t0     = time.perf_counter()
    with torch.no_grad():
        logits = model(preprocess_pil(pil_img).to(device))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    ms   = (time.perf_counter() - t0) * 1000
    top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
    return [(CLASSES[i], float(v)) for i, v in top5], ms


# ═════════════════════════════════════════════════════════════════════════════
# Custom Widgets
# ═════════════════════════════════════════════════════════════════════════════

class AnimatedBar(tk.Canvas):
    def __init__(self, parent, label, value, rank, color, **kw):
        super().__init__(parent, bg=C["card"], highlightthickness=0, height=54, **kw)
        self.label   = label
        self.target  = value
        self.current = 0.0
        self.rank    = rank
        self.color   = color
        self.bind("<Configure>", lambda e: self._draw(self.current))
        self.after(40 + rank * 70, self._animate)

    def _animate(self):
        step = (self.target - self.current) * 0.15
        self.current = min(self.current + max(step, 0.003), self.target)
        self._draw(self.current)
        if abs(self.current - self.target) > 0.001:
            self.after(16, self._animate)

    def _draw(self, val):
        self.delete("all")
        W = self.winfo_width() or 400
        H = self.winfo_height() or 54
        self.create_rectangle(0, 0, W, H,
                               fill=C["bg3"] if self.rank == 0 else C["card"],
                               outline="")
        bw = 36
        self.create_rectangle(0, 0, bw, H,
                               fill=self.color if self.rank == 0 else C["bg4"],
                               outline="")
        self.create_text(bw // 2, H // 2, text=f"#{self.rank+1}",
                          fill=C["white"] if self.rank == 0 else C["text3"],
                          font=("Helvetica", 10, "bold"))
        cx = bw + 16
        self.create_text(cx, H // 2, text=f'"{self.label}"',
                          fill=C["white"] if self.rank == 0 else C["text2"],
                          font=("Helvetica", 16, "bold"), anchor="w")
        bx  = cx + 44
        by  = H // 2 - 6
        bh  = 12
        bnd = W - 70
        bwi = bnd - bx
        self.create_rectangle(bx, by, bnd, by + bh, fill=C["bar_bg"], outline="")
        fw = int(bwi * val)
        if fw > 4:
            self.create_rectangle(bx, by, bx + fw, by + bh, fill=self.color, outline="")
            self.create_rectangle(bx, by, bx + fw, by + bh // 2,
                                   fill=self._lighten(self.color, 0.28), outline="")
        self.create_text(W - 8, H // 2, text=f"{val*100:.1f}%",
                          fill=self.color if self.rank == 0 else C["text2"],
                          font=("Helvetica", 11, "bold"), anchor="e")
        self.create_line(bw, H - 1, W, H - 1, fill=C["border"], width=1)

    @staticmethod
    def _lighten(hex_color, amt):
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
        hh, s, v = colorsys.rgb_to_hsv(r, g, b)
        r2, g2, b2 = colorsys.hsv_to_rgb(hh, s, min(1.0, v + amt))
        return "#{:02x}{:02x}{:02x}".format(int(r2*255), int(g2*255), int(b2*255))


class MetricCard(tk.Frame):
    def __init__(self, parent, title, value="—", subtitle="", color=None, **kw):
        super().__init__(parent, bg=C["card"], **kw)
        color = color or C["accent"]
        tk.Frame(self, bg=color, height=3).pack(fill="x")
        inner = tk.Frame(self, bg=C["card"], padx=14, pady=10)
        inner.pack(fill="both", expand=True)
        tk.Label(inner, text=title.upper(), bg=C["card"], fg=C["text3"],
                 font=("Helvetica", 9, "bold")).pack(anchor="w")
        self._val = tk.Label(inner, text=value, bg=C["card"], fg=color,
                              font=("Helvetica", 24, "bold"))
        self._val.pack(anchor="w", pady=(3, 0))
        self._sub = tk.Label(inner, text=subtitle, bg=C["card"], fg=C["text3"],
                              font=("Helvetica", 10))
        self._sub.pack(anchor="w")

    def update(self, value, subtitle=""):
        self._val.config(text=value)
        if subtitle:
            self._sub.config(text=subtitle)


class StatusBar(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["bg2"], height=28, **kw)
        self.pack_propagate(False)
        self._dot = tk.Label(self, text="●", bg=C["bg2"], fg=C["success"],
                              font=("Helvetica", 10))
        self._dot.pack(side="left", padx=(12, 4))
        self._lbl = tk.Label(self, text="Ready", bg=C["bg2"], fg=C["text2"],
                              font=("Helvetica", 10))
        self._lbl.pack(side="left")
        self._rgt = tk.Label(self, text="", bg=C["bg2"], fg=C["text3"],
                              font=("Helvetica", 10))
        self._rgt.pack(side="right", padx=12)

    def set(self, msg, color=None, right=""):
        self._lbl.config(text=msg)
        self._dot.config(fg=color or C["success"])
        self._rgt.config(text=right)


class ImageDropZone(tk.Canvas):
    def __init__(self, parent, on_image, **kw):
        super().__init__(parent, bg=C["bg3"], highlightthickness=0,
                         cursor="hand2", **kw)
        self.on_image = on_image
        self._hover   = False
        self._pil     = None
        self._tk      = None
        self.bind("<Configure>",  self._redraw)
        self.bind("<Button-1>",   self._click)
        self.bind("<Enter>",      lambda e: self._set_hover(True))
        self.bind("<Leave>",      lambda e: self._set_hover(False))

    def _set_hover(self, val):
        self._hover = val; self._redraw()

    def _redraw(self, _=None):
        self.delete("all")
        W = self.winfo_width()  or 400
        H = self.winfo_height() or 320
        if self._pil:
            self._draw_image(W, H)
        else:
            self._draw_placeholder(W, H)

    def _draw_placeholder(self, W, H):
        dc = C["accent"] if self._hover else C["border2"]
        for i in range(16, W - 16, 18):
            self.create_line(i, 3, i + 10, 3, fill=dc, width=2)
            self.create_line(i, H - 3, i + 10, H - 3, fill=dc, width=2)
        for i in range(16, H - 16, 18):
            self.create_line(3, i, 3, i + 10, fill=dc, width=2)
            self.create_line(W - 3, i, W - 3, i + 10, fill=dc, width=2)
        cx, cy = W // 2, H // 2 - 28
        r = 34
        ic = C["accent"] if self._hover else C["text3"]
        self.create_oval(cx - r, cy - r, cx + r, cy + r,
                          outline=ic, width=2,
                          fill=C["bg4"] if self._hover else "")
        self.create_line(cx, cy + 14, cx, cy - 10,
                          fill=ic, width=3, arrow=tk.LAST, arrowshape=(9, 11, 4))
        self.create_line(cx - 12, cy + 14, cx + 12, cy + 14, fill=ic, width=3)
        self.create_line(cx - 12, cy + 14, cx - 12, cy + 18, fill=ic, width=3)
        self.create_line(cx + 12, cy + 14, cx + 12, cy + 18, fill=ic, width=3)
        self.create_text(cx, cy + r + 20,
                          text="Drop Image Here" if not self._hover else "Click to Browse",
                          fill=C["accent"] if self._hover else C["text2"],
                          font=("Helvetica", 13, "bold"))
        self.create_text(cx, cy + r + 40,
                          text="PNG · JPG · BMP · GIF · TIFF · WebP · ICO · PPM",
                          fill=C["text3"], font=("Helvetica", 10))

    def _draw_image(self, W, H):
        img = self._pil.copy()
        img.thumbnail((W - 24, H - 24), Image.LANCZOS)
        iW, iH = img.size
        x, y   = (W - iW) // 2, (H - iH) // 2
        self.create_rectangle(x + 4, y + 4, x + iW + 4, y + iH + 4,
                               fill="#050810", outline="")
        self._tk = ImageTk.PhotoImage(img)
        self.create_image(x, y, anchor="nw", image=self._tk)
        self.create_rectangle(x, y, x + iW, y + iH, outline=C["accent"], width=1)
        if self._hover:
            self.create_rectangle(x, y + iH - 28, x + iW, y + iH,
                                   fill="#00000099", outline="")
            self.create_text(x + iW // 2, y + iH - 14,
                              text="Click to change", fill=C["white"],
                              font=("Helvetica", 10))

    def _click(self, _=None):
        ftypes = [
            ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp *.ico *.ppm *.pgm"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select Image", filetypes=ftypes)
        if path:
            self._load(path)

    def _load(self, path):
        try:
            img = Image.open(path)
            self._pil = img; self._redraw()
            self.on_image(img, path)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")

    def clear(self):
        self._pil = None; self._tk = None; self._redraw()


class DrawCanvas(tk.Frame):
    """Mouse drawing canvas — brush size, colour picker, undo, save, erase."""

    BRUSH_SIZES = [6, 10, 16, 24]

    def __init__(self, parent, on_image, **kw):
        super().__init__(parent, bg=C["bg3"], **kw)
        self.on_image    = on_image
        self._last_x     = None
        self._last_y     = None
        self._brush_size = 14
        self._ink_color  = "#111111"
        self._bg_color   = "#FFFFFF"
        self._history    = []
        self._pil_canvas = None
        self._hint_id    = None
        self._tk_buf     = None

        self._build()
        self.after(120, self._reset_canvas)

    def _build(self):
        # ── Toolbar ────────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg=C["bg4"], pady=6, padx=10)
        toolbar.pack(fill="x")

        tk.Label(toolbar, text="Brush:", bg=C["bg4"], fg=C["text3"],
                 font=("Helvetica", 10)).pack(side="left", padx=(0, 4))

        self._size_var = tk.IntVar(value=self._brush_size)
        for sz in self.BRUSH_SIZES:
            tk.Radiobutton(
                toolbar, text=str(sz), variable=self._size_var, value=sz,
                command=lambda s=sz: self._set_brush(s),
                bg=C["bg4"], fg=C["text2"], selectcolor=C["bg3"],
                activebackground=C["bg4"], font=("Helvetica", 10),
                relief="flat", cursor="hand2",
            ).pack(side="left", padx=2)

        tk.Frame(toolbar, bg=C["border"], width=1).pack(side="left", fill="y", padx=8)

        # Ink colour
        tk.Label(toolbar, text="Ink:", bg=C["bg4"], fg=C["text3"],
                 font=("Helvetica", 10)).pack(side="left", padx=(0, 4))
        self._ink_btn = tk.Button(
            toolbar, text="   ", bg=self._ink_color, relief="flat",
            width=3, cursor="hand2",
            command=lambda: self._pick_color("ink"))
        self._ink_btn.pack(side="left", padx=(0, 4))

        tk.Label(toolbar, text="Bg:", bg=C["bg4"], fg=C["text3"],
                 font=("Helvetica", 10)).pack(side="left", padx=(4, 4))
        self._bg_btn = tk.Button(
            toolbar, text="   ", bg=self._bg_color, relief="flat",
            width=3, cursor="hand2",
            command=lambda: self._pick_color("bg"))
        self._bg_btn.pack(side="left", padx=(0, 8))

        tk.Frame(toolbar, bg=C["border"], width=1).pack(side="left", fill="y", padx=6)

        # Brush preview dot
        self._preview = tk.Canvas(toolbar, width=34, height=34,
                                   bg=C["bg4"], highlightthickness=0)
        self._preview.pack(side="left", padx=(0, 10))
        self._update_preview()

        # Action buttons
        bk = dict(bg=C["bg3"], fg=C["text2"], relief="flat",
                   font=("Helvetica", 10), padx=10, pady=4,
                   cursor="hand2", bd=0,
                   activebackground=C["border2"],
                   activeforeground=C["white"])
        tk.Button(toolbar, text="↩ Undo",  command=self._undo,         **bk).pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="✕ Clear", command=self._reset_canvas, **bk).pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="💾 Save",  command=self._save_canvas,  **bk).pack(side="left")

        # Coord label
        self._coord_lbl = tk.Label(toolbar, text="x:—  y:—", bg=C["bg4"],
                                    fg=C["text3"], font=("Helvetica", 9))
        self._coord_lbl.pack(side="right")

        # ── Canvas ──────────────────────────────────────────────────────────
        outer = tk.Frame(self, bg=C["border"])
        outer.pack(fill="both", expand=True, padx=10, pady=(6, 6))

        self.canvas = tk.Canvas(outer, bg=self._bg_color,
                                 highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>",   self._on_press_erase)
        self.canvas.bind("<B3-Motion>",       self._on_drag_erase)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)
        self.canvas.bind("<Motion>",          self._on_hover)
        self.canvas.bind("<Configure>",       self._on_resize)

        # ── Bottom info ──────────────────────────────────────────────────────
        bottom = tk.Frame(self, bg=C["bg3"], pady=5, padx=10)
        bottom.pack(fill="x")
        self._info_lbl = tk.Label(bottom, text="Canvas ready  ·  right-click = erase  ·  Ctrl+Z = undo",
                                   bg=C["bg3"], fg=C["text3"], font=("Helvetica", 10))
        self._info_lbl.pack(side="left")

        self.bind_all("<Control-z>", lambda e: self._undo())

    # ── Canvas management ─────────────────────────────────────────────────────
    def _reset_canvas(self):
        W = max(self.canvas.winfo_width(), 4)
        H = max(self.canvas.winfo_height(), 4)
        self._pil_canvas = Image.new("RGB", (W, H), self._bg_color)
        self._history.clear()
        self.canvas.delete("all")
        self.canvas.config(bg=self._bg_color)
        cx, cy = W // 2, H // 2
        self._hint_id = self.canvas.create_text(
            cx, cy,
            text="Draw here  ·  Right-click = Erase  ·  Ctrl+Z = Undo",
            fill=C["text3"], font=("Helvetica", 11))
        self._info_lbl.config(text="Canvas empty  ·  right-click = erase  ·  Ctrl+Z = undo")

    def _on_resize(self, event):
        if self._pil_canvas is None:
            self._reset_canvas()
            return
        old = self._pil_canvas
        new = Image.new("RGB", (event.width, event.height), self._bg_color)
        new.paste(old, (0, 0))
        self._pil_canvas = new

    def _push_history(self):
        if self._pil_canvas:
            self._history.append(self._pil_canvas.copy())
            if len(self._history) > 30:
                self._history.pop(0)

    def _undo(self):
        if not self._history:
            return
        self._pil_canvas = self._history.pop()
        self._refresh_from_pil()
        self._info_lbl.config(text="Undo applied")

    def _refresh_from_pil(self):
        self.canvas.delete("all")
        self._tk_buf = ImageTk.PhotoImage(self._pil_canvas)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_buf)

    # ── Stroke ────────────────────────────────────────────────────────────────
    def _stroke(self, x0, y0, x1, y1, color):
        if self._pil_canvas is None:
            return
        r    = self._brush_size // 2
        draw = ImageDraw.Draw(self._pil_canvas)
        if x0 is not None:
            draw.line([x0, y0, x1, y1], fill=color, width=self._brush_size)
            draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=color)
            self.canvas.create_line(x0, y0, x1, y1, fill=color,
                                     width=self._brush_size,
                                     capstyle=tk.ROUND, smooth=True)
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=color)
        self.canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r,
                                 fill=color, outline="")

    def _on_press(self, e):
        self._push_history()
        if self._hint_id:
            self.canvas.delete(self._hint_id)
            self._hint_id = None
        self._last_x, self._last_y = e.x, e.y
        self._stroke(None, None, e.x, e.y, self._ink_color)
        self._info_lbl.config(text=f"Drawing  ·  brush {self._brush_size}px")

    def _on_drag(self, e):
        self._stroke(self._last_x, self._last_y, e.x, e.y, self._ink_color)
        self._last_x, self._last_y = e.x, e.y

    def _on_press_erase(self, e):
        self._push_history()
        if self._hint_id:
            self.canvas.delete(self._hint_id); self._hint_id = None
        self._last_x, self._last_y = e.x, e.y
        self._stroke(None, None, e.x, e.y, self._bg_color)
        self._info_lbl.config(text="Erasing…")

    def _on_drag_erase(self, e):
        self._stroke(self._last_x, self._last_y, e.x, e.y, self._bg_color)
        self._last_x, self._last_y = e.x, e.y

    def _on_release(self, e):
        self._last_x = self._last_y = None
        if self._pil_canvas:
            W, H = self._pil_canvas.size
            self._info_lbl.config(text=f"Ready  ·  {W}×{H}px")
            self.on_image(self._pil_canvas.copy(), "canvas")

    def _on_hover(self, e):
        self._coord_lbl.config(text=f"x:{e.x}  y:{e.y}")

    # ── Brush / colour ────────────────────────────────────────────────────────
    def _set_brush(self, sz):
        self._brush_size = sz
        self._update_preview()

    def _pick_color(self, which):
        from tkinter.colorchooser import askcolor
        init = self._ink_color if which == "ink" else self._bg_color
        result = askcolor(color=init, title="Choose colour")
        if result and result[1]:
            hex_c = result[1]
            if which == "ink":
                self._ink_color = hex_c
                self._ink_btn.config(bg=hex_c)
            else:
                self._bg_color = hex_c
                self._bg_btn.config(bg=hex_c)
                self.canvas.config(bg=hex_c)
                if self._pil_canvas:
                    tmp = Image.new("RGB", self._pil_canvas.size, hex_c)
                    tmp.paste(self._pil_canvas)
                    self._pil_canvas = tmp
            self._update_preview()

    def _update_preview(self):
        self._preview.delete("all")
        cx = cy = 17
        r = max(2, self._brush_size // 2)
        self._preview.create_oval(cx - r, cy - r, cx + r, cy + r,
                                   fill=self._ink_color, outline="")

    def _save_canvas(self):
        if not self._pil_canvas:
            return
        path = filedialog.asksaveasfilename(
            title="Save canvas", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")])
        if path:
            self._pil_canvas.save(path)
            messagebox.showinfo("Saved", f"Canvas saved:\n{path}")

    def get_pil(self):
        return self._pil_canvas.copy() if self._pil_canvas else None


# ═════════════════════════════════════════════════════════════════════════════
# Main Application
# ═════════════════════════════════════════════════════════════════════════════

class HandwritingRecognitionApp(tk.Tk):
    BAR_COLORS = [C["accent"], C["accent3"], C["accent2"], C["warning"], C["text3"]]

    def __init__(self):
        super().__init__()
        self.title("Handwritten Character Recognition  |  ITM-390 · AUPP")
        self.configure(bg=C["bg"])
        self.geometry("1320x860")
        self.minsize(1050, 720)

        self._model        = None
        self._upload_img   = None
        self._canvas_img   = None
        self._history      = []
        self._bar_widgets  = []
        self._active_src   = "upload"

        self._build_ui()
        self._load_model_async()

    # ── Model ─────────────────────────────────────────────────────────────────
    def _load_model_async(self):
        self.status.set("Loading model…", C["warning"])
        threading.Thread(target=self._bg_load, daemon=True).start()

    def _bg_load(self):
        model, err = load_model()
        self.after(0, lambda: self._model_ready(model, err))

    def _model_ready(self, model, err):
        self._model = model
        if err:
            self.status.set(f"Demo mode — {err}", C["warning"],
                            "Install PyTorch + checkpoint for real predictions")
            self._badge.config(text="DEMO MODE", fg=C["warning"])
        else:
            dev = "GPU" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "CPU"
            self.status.set("Model ready", C["success"],
                            f"HandwritingResNet · 2.77M params · {dev}")
            self._badge.config(text="● MODEL READY", fg=C["success"])

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self.status = StatusBar(self)
        self.status.pack(side="bottom", fill="x")
        self._build_toolbar()
        self._build_body()

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["bg2"], height=58)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Frame(hdr, bg=C["accent"], width=4).pack(side="left", fill="y")
        lf = tk.Frame(hdr, bg=C["bg2"])
        lf.pack(side="left", padx=18)
        tk.Label(lf, text="✍", bg=C["bg2"], fg=C["accent"],
                 font=("Helvetica", 20)).pack(side="left", padx=(0, 10))
        tk.Label(lf, text="Handwritten Character Recognition",
                 bg=C["bg2"], fg=C["text"],
                 font=("Helvetica", 14, "bold")).pack(side="left")
        tk.Label(lf, text="  ·  ITM-390 Machine Learning  ·  AUPP",
                 bg=C["bg2"], fg=C["text3"],
                 font=("Helvetica", 11)).pack(side="left")
        rf = tk.Frame(hdr, bg=C["bg2"])
        rf.pack(side="right", padx=18)
        self._badge = tk.Label(rf, text="● LOADING…", bg=C["bg2"],
                                fg=C["warning"], font=("Helvetica", 10, "bold"))
        self._badge.pack(side="right")

    def _build_toolbar(self):
        tb = tk.Frame(self, bg=C["bg"], pady=8)
        tb.pack(fill="x", padx=16)
        bk = dict(bg=C["bg4"], fg=C["text"], relief="flat",
                   font=("Helvetica", 11), padx=14, pady=6,
                   activebackground=C["accent_dim"],
                   activeforeground=C["white"], cursor="hand2", bd=0)
        self._btn_predict = tk.Button(
            tb, text="▶  Run Prediction", command=self._on_predict,
            state="disabled", bg=C["accent"], fg=C["white"], relief="flat",
            font=("Helvetica", 11, "bold"), padx=16, pady=6,
            activebackground=C["accent2"], cursor="hand2", bd=0)
        self._btn_predict.pack(side="left", padx=(0, 8))
        tk.Button(tb, text="✕  Clear",      command=self._on_clear,        **bk).pack(side="left", padx=(0, 8))
        tk.Button(tb, text="📋  History",    command=self._show_history,    **bk).pack(side="left", padx=(0, 8))
        tk.Button(tb, text="ℹ  Model Info", command=self._show_model_info, **bk).pack(side="left")
        tk.Label(tb, text="Enter = Predict  ·  Esc = Clear  ·  Ctrl+Z = Undo (draw)",
                 bg=C["bg"], fg=C["text3"], font=("Helvetica", 10)).pack(side="right")
        self.bind("<Return>", lambda e: self._on_predict())
        self.bind("<Escape>", lambda e: self._on_clear())

    def _build_body(self):
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ── Left: tabbed input ────────────────────────────────────────────────
        left = tk.Frame(body, bg=C["bg"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Tab buttons
        tab_bar = tk.Frame(left, bg=C["bg"])
        tab_bar.grid(row=0, column=0, sticky="ew")

        tab_kw = dict(relief="flat", font=("Helvetica", 11),
                       padx=20, pady=8, cursor="hand2", bd=0)

        self._tb_upload = tk.Button(
            tab_bar, text="☁  Upload Image",
            command=lambda: self._switch_tab("upload"),
            bg=C["tab_active"], fg=C["accent"],
            activebackground=C["tab_active"], activeforeground=C["accent"],
            **tab_kw)
        self._tb_upload.pack(side="left")

        self._tb_draw = tk.Button(
            tab_bar, text="✏  Draw Canvas",
            command=lambda: self._switch_tab("draw"),
            bg=C["tab_idle"], fg=C["text3"],
            activebackground=C["tab_active"], activeforeground=C["text"],
            **tab_kw)
        self._tb_draw.pack(side="left", padx=(2, 0))

        # Sliding underline indicator
        self._indicator = tk.Frame(tab_bar, bg=C["accent"], height=2)
        self._indicator.place(in_=self._tb_upload, relwidth=1.0, rely=1.0, y=-2)

        # Tab content
        tab_content = tk.Frame(left, bg=C["bg"])
        tab_content.grid(row=1, column=0, sticky="nsew")
        tab_content.rowconfigure(0, weight=1)
        tab_content.columnconfigure(0, weight=1)

        # ── Upload panel ──────────────────────────────────────────────────────
        self._upload_panel = tk.Frame(tab_content, bg=C["bg"])
        self._upload_panel.grid(row=0, column=0, sticky="nsew")
        self._upload_panel.rowconfigure(0, weight=1)
        self._upload_panel.columnconfigure(0, weight=1)

        self.drop_zone = ImageDropZone(self._upload_panel,
                                        on_image=self._on_upload_image)
        self.drop_zone.grid(row=0, column=0, sticky="nsew",
                             highlightthickness=1,
                             highlightbackground=C["border"])

        self._upload_info = tk.Label(
            self._upload_panel, text="No image loaded",
            bg=C["bg2"], fg=C["text3"],
            font=("Helvetica", 10), anchor="w", padx=10, pady=5)
        self._upload_info.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        # ── Draw panel ────────────────────────────────────────────────────────
        self._draw_panel = tk.Frame(tab_content, bg=C["bg"])
        self._draw_panel.grid(row=0, column=0, sticky="nsew")
        self._draw_panel.rowconfigure(0, weight=1)
        self._draw_panel.columnconfigure(0, weight=1)

        self.draw_canvas_widget = DrawCanvas(self._draw_panel,
                                              on_image=self._on_canvas_image)
        self.draw_canvas_widget.grid(row=0, column=0, sticky="nsew",
                                      highlightthickness=1,
                                      highlightbackground=C["border"])

        # Start with upload panel on top
        self._upload_panel.tkraise()

        # ── Right: results panel ──────────────────────────────────────────────
        right = tk.Frame(body, bg=C["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        self._sec_label("PREDICTION METRICS", right, row=0)

        mr1 = tk.Frame(right, bg=C["bg"])
        mr1.grid(row=1, column=0, sticky="ew")
        mr1.columnconfigure((0, 1, 2), weight=1)

        self._c_char    = MetricCard(mr1, "Top Character",  "—", "result",       C["accent"])
        self._c_conf    = MetricCard(mr1, "Confidence",     "—", "probability",  C["accent3"])
        self._c_ms      = MetricCard(mr1, "Inference Time", "—", "latency",      C["accent2"])
        self._c_char.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._c_conf.grid(row=0, column=1, sticky="nsew", padx=(0, 6))
        self._c_ms.grid  (row=0, column=2, sticky="nsew")

        mr2 = tk.Frame(right, bg=C["bg"])
        mr2.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        mr2.columnconfigure((0, 1), weight=1)

        self._c_class   = MetricCard(mr2, "Class Type", "—",
                                      "digit / upper / lower", C["warning"])
        self._c_entropy = MetricCard(mr2, "Entropy",    "—",
                                      "uncertainty (bits)",    "#E879F9")
        self._c_class.grid  (row=0, column=0, sticky="nsew", padx=(0, 6))
        self._c_entropy.grid(row=0, column=1, sticky="nsew")

        self._sec_label("TOP-5 PREDICTIONS", right, row=3, pady=(14, 6))

        self._bars_frame = tk.Frame(right, bg=C["card"],
                                     highlightthickness=1,
                                     highlightbackground=C["border"])
        self._bars_frame.grid(row=4, column=0, sticky="nsew")
        right.rowconfigure(4, weight=1)
        self._draw_empty_bars()

        self._sec_label("ANALYSIS", right, row=5, pady=(12, 6))

        af = tk.Frame(right, bg=C["card"],
                       highlightthickness=1, highlightbackground=C["border"])
        af.grid(row=6, column=0, sticky="ew")

        self._analysis = tk.Text(
            af, height=4, bg=C["card"], fg=C["text2"],
            font=("Helvetica", 11), relief="flat",
            padx=12, pady=8, wrap="word", state="disabled",
            insertbackground=C["text"], selectbackground=C["accent_dim"])
        self._analysis.pack(fill="both", expand=True)
        self._set_analysis(
            "Choose a tab above — Upload an image file or Draw on the canvas.\n"
            "Then click ▶ Run Prediction (or press Enter).")

    @staticmethod
    def _sec_label(text, parent, row, pady=(0, 6)):
        tk.Label(parent, text=text, bg=C["bg"], fg=C["text3"],
                 font=("Helvetica", 9, "bold")).grid(
            row=row, column=0, sticky="w", pady=pady)

    # ── Tab switching ─────────────────────────────────────────────────────────
    def _switch_tab(self, which):
        self._active_src = which
        if which == "upload":
            self._upload_panel.tkraise()
            self._tb_upload.config(bg=C["tab_active"], fg=C["accent"])
            self._tb_draw.config(bg=C["tab_idle"],   fg=C["text3"])
            self._indicator.place(in_=self._tb_upload, relwidth=1.0, rely=1.0, y=-2)
            self._btn_predict.config(
                state="normal" if self._upload_img else "disabled")
        else:
            self._draw_panel.tkraise()
            self._tb_draw.config(bg=C["tab_active"], fg=C["accent"])
            self._tb_upload.config(bg=C["tab_idle"],   fg=C["text3"])
            self._indicator.place(in_=self._tb_draw, relwidth=1.0, rely=1.0, y=-2)
            self._btn_predict.config(
                state="normal" if self._canvas_img else "disabled")

    # ── Image callbacks ───────────────────────────────────────────────────────
    def _on_upload_image(self, pil_img, path):
        self._upload_img = pil_img
        fname  = Path(path).name
        W, H   = pil_img.size
        mode   = pil_img.mode
        sz_kb  = (Path(path).stat().st_size / 1024
                  if path != "canvas" and Path(path).exists() else 0)
        self._upload_info.config(
            text=f"  {fname}  ·  {W}×{H}px  ·  {mode}  ·  {sz_kb:.1f} KB",
            fg=C["text2"])
        if self._active_src == "upload":
            self._btn_predict.config(state="normal")
        self._reset_results()
        self.status.set(f"Image loaded: {fname}", C["accent"])

    def _on_canvas_image(self, pil_img, _):
        self._canvas_img = pil_img
        if self._active_src == "draw":
            self._btn_predict.config(state="normal")

    # ── Prediction ────────────────────────────────────────────────────────────
    def _on_predict(self):
        img = self._upload_img if self._active_src == "upload" else self._canvas_img
        if img is None:
            messagebox.showinfo("Nothing to predict",
                                "Please upload an image or draw something first.")
            return
        if str(self._btn_predict["state"]) == "disabled":
            return
        self._btn_predict.config(state="disabled", text="⏳ Running…")
        self.status.set("Running prediction…", C["warning"])
        src = self._active_src
        threading.Thread(
            target=lambda: self.after(0,
                lambda: self._on_results(*run_predict(self._model, img), src)),
            daemon=True).start()

    def _on_results(self, results, ms, source):
        top_char, top_prob = results[0]
        pct   = f"{top_prob*100:.1f}%"
        ctype, _ = self._class_type(top_char)
        clbl  = self._confidence_label(top_prob)
        ent   = self._entropy([p for _, p in results])

        self._c_char.update(f'"{top_char}"', "predicted character")
        self._c_conf.update(pct, clbl)
        self._c_ms.update(f"{ms:.1f} ms", "inference latency")
        self._c_class.update(ctype, "character class")
        self._c_entropy.update(f"{ent:.2f}", "bits (lower = more certain)")
        self._draw_bars(results)

        analysis = (
            f'Prediction: "{top_char}" ({ctype})  —  {clbl} confidence ({pct})\n'
            f'Top-2: "{results[1][0]}" ({results[1][1]*100:.1f}%)  ·  '
            f'Top-3: "{results[2][0]}" ({results[2][1]*100:.1f}%)\n'
        )
        if top_prob < 0.40:
            analysis += "\n⚠  Low confidence. Try a cleaner drawing or higher-quality image."
        elif top_prob > 0.80:
            analysis += "\n✓  High confidence prediction."
        self._set_analysis(analysis)

        src_label = "Upload" if source == "upload" else "Draw Canvas"
        self._history.append((src_label, top_char, top_prob, ms))
        self._btn_predict.config(state="normal", text="▶  Run Prediction")
        self.status.set(f'Predicted: "{top_char}"  ({pct})',
                        C["success"], f"Inference: {ms:.1f} ms")

    def _on_clear(self):
        if self._active_src == "upload":
            self.drop_zone.clear()
            self._upload_img = None
            self._upload_info.config(text="No image loaded", fg=C["text3"])
        else:
            self.draw_canvas_widget._reset_canvas()
            self._canvas_img = None
        self._btn_predict.config(state="disabled", text="▶  Run Prediction")
        self._reset_results()
        self.status.set("Cleared", C["text3"])

    # ── Shared helpers ────────────────────────────────────────────────────────
    def _reset_results(self):
        for c in (self._c_char, self._c_conf, self._c_ms,
                  self._c_class, self._c_entropy):
            c.update("—")
        self._draw_empty_bars()
        self._set_analysis(
            "Choose a tab — Upload an image file or Draw on the canvas.\n"
            "Then click ▶ Run Prediction (or press Enter).")

    def _draw_empty_bars(self):
        for w in self._bar_widgets:
            w.destroy()
        self._bar_widgets = []
        for i in range(5):
            b = AnimatedBar(self._bars_frame, "—", 0.0, i, self.BAR_COLORS[i])
            b.pack(fill="x")
            self._bar_widgets.append(b)

    def _draw_bars(self, results):
        for w in self._bar_widgets:
            w.destroy()
        self._bar_widgets = []
        for i, (char, prob) in enumerate(results):
            b = AnimatedBar(self._bars_frame, char, prob, i, self.BAR_COLORS[i])
            b.pack(fill="x")
            self._bar_widgets.append(b)

    def _set_analysis(self, text):
        self._analysis.config(state="normal")
        self._analysis.delete("1.0", "end")
        self._analysis.insert("1.0", text)
        self._analysis.config(state="disabled")

    @staticmethod
    def _entropy(probs):
        arr = np.array(probs, dtype=np.float64)
        arr = arr[arr > 0]
        return float(-np.sum(arr * np.log2(arr)))

    @staticmethod
    def _class_type(char):
        if char.isdigit():  return "Digit",     C["warning"]
        if char.isupper():  return "Uppercase",  C["accent"]
        return "Lowercase", C["accent3"]

    @staticmethod
    def _confidence_label(p):
        if p >= 0.85: return "Very High"
        if p >= 0.65: return "High"
        if p >= 0.45: return "Moderate"
        if p >= 0.30: return "Low"
        return "Very Low"

    # ── Modals ────────────────────────────────────────────────────────────────
    def _show_history(self):
        win = tk.Toplevel(self)
        win.title("Prediction History")
        win.configure(bg=C["bg"])
        win.geometry("760x420")
        tk.Label(win, text="PREDICTION HISTORY", bg=C["bg"], fg=C["text3"],
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=16, pady=(16, 8))
        frame = tk.Frame(win, bg=C["bg"])
        frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        cols = ("Source", "Result", "Confidence", "Inference (ms)")
        style = ttk.Style(); style.theme_use("clam")
        style.configure("Treeview", background=C["card"],
                        fieldbackground=C["card"], foreground=C["text"],
                        rowheight=32, font=("Helvetica", 11))
        style.configure("Treeview.Heading", background=C["bg3"],
                        foreground=C["text2"], font=("Helvetica", 10, "bold"))
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=14)
        for col, w in zip(cols, [160, 100, 120, 130]):
            tree.heading(col, text=col); tree.column(col, anchor="center", width=w)
        for src, char, prob, ms in reversed(self._history):
            tree.insert("", "end",
                        values=(src, f'"{char}"', f"{prob*100:.1f}%", f"{ms:.1f}"))
        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        if not self._history:
            tk.Label(win, text="No predictions yet.", bg=C["bg"],
                     fg=C["text3"], font=("Helvetica", 12)).pack(pady=40)

    def _show_model_info(self):
        win = tk.Toplevel(self)
        win.title("Model Information")
        win.configure(bg=C["bg"])
        win.geometry("580x440")
        win.resizable(False, False)
        tk.Label(win, text="MODEL INFORMATION", bg=C["bg"], fg=C["text3"],
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=20, pady=(20, 10))
        rows = [
            ("Architecture",    "HandwritingResNet (custom ResNet-style CNN)"),
            ("Parameters",      "2,770,000  (2.77M)"),
            ("Classes",         "62   (0–9 · A–Z · a–z)"),
            ("Dataset",         "EMNIST ByClass — 814,255 samples"),
            ("Best Accuracy",   "84.74%  (epoch 11 / 20)"),
            ("Optimizer",       "AdamW  ·  LR 1e-3 → 1e-6 cosine annealing"),
            ("Loss",            "CrossEntropy + label smoothing 0.1"),
            ("Mixed Precision", "FP16 via PyTorch AMP"),
            ("Batch Size",      "256"),
            ("Checkpoint",      str(CHECKPOINT_PATH)),
            ("PyTorch",         "Available ✓" if TORCH_AVAILABLE
                                              else "Not installed — demo mode active"),
        ]
        for title, val in rows:
            row = tk.Frame(win, bg=C["card"], pady=8, padx=16)
            row.pack(fill="x", padx=20, pady=1)
            tk.Label(row, text=title, bg=C["card"], fg=C["text3"],
                     font=("Helvetica", 10), width=20, anchor="w").pack(side="left")
            tk.Label(row, text=val, bg=C["card"], fg=C["text"],
                     font=("Helvetica", 10)).pack(side="left")
        tk.Button(win, text="  Close  ", command=win.destroy,
                  bg=C["accent"], fg=C["white"], relief="flat",
                  font=("Helvetica", 11), padx=20, pady=6).pack(pady=20)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = HandwritingRecognitionApp()
    app.mainloop()
