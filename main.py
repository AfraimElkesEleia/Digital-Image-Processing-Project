"""
Digital Image Processing — Main Application
PyQt6 · OpenCV · Matplotlib
Two-channel pipeline: Filters  |  Noise
"""
import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFileDialog, QSlider,
    QComboBox, QSpinBox, QDoubleSpinBox, QFrame, QButtonGroup,
    QGridLayout, QSizePolicy, QToolButton, QGraphicsDropShadowEffect,
    QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette

from models import ALGORITHMS, CATEGORIES, AlgorithmModel, Parameter
from filters import apply_filter


# ──────────────────────────────────────────────────────────────────────
# Theme constants
# ──────────────────────────────────────────────────────────────────────
PRIMARY        = "#1a73e8"
PRIMARY_LIGHT  = "#e8f0fe"
PRIMARY_DARK   = "#1557b0"
BG_APP         = "#f0f4f8"
BG_CARD        = "#ffffff"
BG_SIDEBAR     = "#f8fafc"
BORDER         = "#dde3ea"
TEXT_PRIMARY   = "#1c2b3a"
TEXT_SECONDARY = "#5f6b7a"
TEXT_MUTED     = "#9aa3ac"
ACCENT_GREEN   = "#1e8e3e"
ACCENT_RED     = "#d93025"
ACCENT_ORANGE  = "#f29900"

CAT_COLORS = {
    "Point Processing": ("#e8f4fd", "#1a6eb5"),
    "Histogram":        ("#fef3e2", "#c96a00"),
    "Spatial Filter":   ("#e6f4ea", "#1e7e34"),
    "Frequency Domain": ("#f3e8fd", "#7b1fa2"),
    "Restoration":      ("#fce8e6", "#c62828"),
    "Segmentation":     ("#e8faf4", "#00695c"),
    "Noise":            ("#fff3e0", "#e65100"),
    "Arithmetic":       ("#fff8e1", "#f57f17"),
}

SLIDER_SS = """
QSlider::groove:horizontal {
    background: #e0e7f0; height: 4px; border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1a73e8,stop:1 #4285f4);
    height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff; border: 2px solid #1a73e8;
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #e8f0fe; border-color: #1557b0; }
"""
SPINBOX_SS = """
QSpinBox, QDoubleSpinBox {
    border: 1px solid #dde3ea; border-radius: 6px; padding: 2px 4px;
    background: #f8fafc; font-size: 11px; color: #1c2b3a;
    min-width: 56px; max-width: 72px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #1a73e8; background: #fff; }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 16px; border: none; background: transparent;
}
"""
COMBO_SS = """
QComboBox {
    border: 1px solid #dde3ea; border-radius: 6px; padding: 3px 8px;
    background: #f8fafc; font-size: 11px; color: #1c2b3a;
}
QComboBox:focus { border-color: #1a73e8; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    border: 1px solid #dde3ea; border-radius: 6px; background: #fff;
    selection-background-color: #e8f0fe; selection-color: #1a73e8;
}
"""


def shadow(widget, radius=12, offset=2, alpha=30):
    fx = QGraphicsDropShadowEffect(widget)
    fx.setBlurRadius(radius)
    fx.setOffset(0, offset)
    fx.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(fx)


# ──────────────────────────────────────────────────────────────────────
# Worker thread
# ──────────────────────────────────────────────────────────────────────
class PipelineWorker(QThread):
    result_ready = pyqtSignal(list)   # list of np.ndarray

    def __init__(self, image, pipeline_items):
        super().__init__()
        self._image = image.copy()
        self._items = pipeline_items   # list of (alg_id, params_dict)

    def run(self):
        results = []
        current = self._image.copy()
        for alg_id, params in self._items:
            try:
                enriched = dict(params)
                if alg_id == "subtraction":
                    target_key = params.get("target", "Original")
                    enriched["_target_image"] = (
                        self._image if target_key == "Original"
                        else (results[-2] if len(results) >= 2 else self._image)
                    )
                current = apply_filter(current, alg_id, enriched)
            except Exception as e:
                print(f"Filter error [{alg_id}]: {e}")
            results.append(current.copy())
        self.result_ready.emit(results)


# ──────────────────────────────────────────────────────────────────────
# Histogram canvas
# ──────────────────────────────────────────────────────────────────────
class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self._fig = Figure(figsize=(3, 1.6), dpi=90, tight_layout=True)
        self._ax  = self._fig.add_subplot(111)
        self._style()
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: transparent;")

    def _style(self):
        self._fig.patch.set_facecolor("#ffffff")
        ax = self._ax
        ax.set_facecolor("#fafcff")
        ax.tick_params(axis="x", labelsize=7, colors=TEXT_SECONDARY, pad=2)
        ax.tick_params(axis="y", labelsize=7, colors=TEXT_SECONDARY)
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)

    def plot(self, image):
        self._ax.clear()
        self._style()
        if image is not None:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
            xs   = np.arange(256)
            self._ax.fill_between(xs, hist, color=PRIMARY, alpha=0.25, linewidth=0)
            self._ax.plot(xs, hist, color=PRIMARY, linewidth=1.0)
            self._ax.set_xlim(0, 255)
            self._ax.set_ylim(bottom=0)
        self.draw_idle()


# ──────────────────────────────────────────────────────────────────────
# Image display
# ──────────────────────────────────────────────────────────────────────
class ImageDisplay(QLabel):
    def __init__(self, placeholder="No image", parent=None):
        super().__init__(parent)
        self._placeholder  = placeholder
        self._pixmap_cache = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(180, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"""
            ImageDisplay {{
                background: #eef2f7; border-radius: 10px;
                color: {TEXT_MUTED}; font-size: 13px;
            }}
        """)
        self.setText(placeholder)

    def set_image(self, img):
        if img is None:
            self._pixmap_cache = None
            self.setText(self._placeholder)
            return
        h, w  = img.shape
        q_img = QImage(img.data.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        self._pixmap_cache = QPixmap.fromImage(q_img)
        self._update_display()

    def _update_display(self):
        if self._pixmap_cache is None:
            return
        s = self.size()
        if s.width() < 2 or s.height() < 2:
            return
        scaled = self._pixmap_cache.scaled(
            s, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


# ──────────────────────────────────────────────────────────────────────
# Kernel selector — discrete odd-only segmented buttons
# ──────────────────────────────────────────────────────────────────────
class KernelSelector(QWidget):
    value_changed = pyqtSignal(int)

    def __init__(self, choices, default, parent=None):
        super().__init__(parent)
        self._current = int(default)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._buttons = {}
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        for c in choices:
            v   = int(c)
            btn = QPushButton(f"{v}×{v}")
            btn.setCheckable(True)
            btn.setChecked(v == self._current)
            btn.setFixedHeight(24)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, val=v: self._select(val))
            grp.addButton(btn)
            self._buttons[v] = btn
            layout.addWidget(btn)
        self._refresh()

    def _select(self, val):
        self._current = val
        self._refresh()
        self.value_changed.emit(val)

    def _refresh(self):
        for v, btn in self._buttons.items():
            if v == self._current:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background:{PRIMARY};color:white;border:none;
                        border-radius:5px;font-size:9px;font-weight:700;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background:{PRIMARY_LIGHT};color:{PRIMARY};border:none;
                        border-radius:5px;font-size:9px;font-weight:500;
                    }}
                    QPushButton:hover{{background:#d2e3fc;}}
                """)

    def value(self):
        return self._current


# ──────────────────────────────────────────────────────────────────────
# Padding selector — four-mode segmented control
# ──────────────────────────────────────────────────────────────────────
class PaddingSelector(QWidget):
    padding_changed = pyqtSignal(str)

    _MODES = [("Zero", "zero"), ("Repl.", "replicate"),
              ("Wrap", "wrap"), ("Trunc.", "truncate")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = "zero"
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._buttons = {}
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        for label, key in self._MODES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(key == "zero")
            btn.setFixedHeight(22)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, k=key: self._select(k))
            grp.addButton(btn)
            self._buttons[key] = btn
            layout.addWidget(btn)
        self._refresh()

    def _select(self, key):
        self._current = key
        self._refresh()
        self.padding_changed.emit(key)

    def _refresh(self):
        for key, btn in self._buttons.items():
            if key == self._current:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background:{PRIMARY};color:white;border:none;
                        border-radius:5px;font-size:9px;font-weight:700;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background:{PRIMARY_LIGHT};color:{PRIMARY};border:none;
                        border-radius:5px;font-size:9px;font-weight:500;
                    }}
                    QPushButton:hover{{background:#d2e3fc;}}
                """)

    def value(self):
        return self._current


# ──────────────────────────────────────────────────────────────────────
# Available filter card (sidebar)
# ──────────────────────────────────────────────────────────────────────
class AvailableFilterCard(QFrame):
    add_clicked = pyqtSignal(AlgorithmModel)

    def __init__(self, model: AlgorithmModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.setObjectName("AvailCard")
        self.setStyleSheet("""
            QFrame#AvailCard {
                background:#fff;border:1px solid #e4e9f0;
                border-radius:10px;margin:2px 4px;
            }
            QFrame#AvailCard:hover {
                border:1.5px solid #1a73e8;background:#f8fbff;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(8)

        bg, fg = CAT_COLORS.get(model.category, ("#e0e0e0", "#555"))
        abbr = QLabel(model.abbreviation)
        abbr.setFixedSize(36, 36)
        abbr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        abbr.setStyleSheet(f"""
            QLabel {{background:{bg};color:{fg};border-radius:8px;
                     font-size:9px;font-weight:700;}}
        """)

        info = QVBoxLayout()
        info.setSpacing(2)
        n = QLabel(model.name)
        n.setStyleSheet(f"font-size:11px;font-weight:600;color:{TEXT_PRIMARY};")
        d = QLabel(model.description.split("\n")[0])
        d.setStyleSheet(f"font-size:9px;color:{TEXT_SECONDARY};")
        d.setWordWrap(True)
        info.addWidget(n)
        info.addWidget(d)

        add_btn = QPushButton("+")
        add_btn.setFixedSize(28, 28)
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.setStyleSheet(f"""
            QPushButton {{border-radius:14px;background:{PRIMARY_LIGHT};
                          color:{PRIMARY};font-size:16px;font-weight:700;border:none;}}
            QPushButton:hover{{background:{PRIMARY};color:white;}}
            QPushButton:pressed{{background:{PRIMARY_DARK};color:white;}}
        """)
        add_btn.clicked.connect(lambda: self.add_clicked.emit(self.model))

        layout.addWidget(abbr)
        layout.addLayout(info, stretch=1)
        layout.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignVCenter)


# ──────────────────────────────────────────────────────────────────────
# Applied filter card (pipeline row)
# ──────────────────────────────────────────────────────────────────────
class AppliedFilterCard(QFrame):
    delete_clicked = pyqtSignal(str)
    params_changed = pyqtSignal()
    view_requested = pyqtSignal(str)

    def __init__(self, uid, model: AlgorithmModel,
                 view_group: QButtonGroup, parent=None):
        super().__init__(parent)
        self.uid    = uid
        self.model  = model
        self.params = {p.name: p.default for p in model.parameters}

        self.setObjectName("PipeCard")
        self.setFixedWidth(224)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._apply_style(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        # ── Header ──
        hdr = QHBoxLayout()
        hdr.setSpacing(4)
        bg, fg = CAT_COLORS.get(model.category, ("#e0e0e0", "#555"))
        abbr = QLabel(model.abbreviation)
        abbr.setFixedSize(32, 20)
        abbr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        abbr.setStyleSheet(f"""
            QLabel{{background:{bg};color:{fg};border-radius:4px;
                    font-size:8px;font-weight:700;}}
        """)
        name_lbl = QLabel(model.name)
        name_lbl.setStyleSheet(
            f"font-size:10px;font-weight:700;color:{TEXT_PRIMARY};")
        name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(20, 20)
        del_btn.setStyleSheet(f"""
            QPushButton{{border:none;background:transparent;
                         color:{TEXT_MUTED};font-size:11px;font-weight:700;}}
            QPushButton:hover{{color:{ACCENT_RED};}}
        """)
        del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_btn.clicked.connect(lambda: self.delete_clicked.emit(self.uid))
        hdr.addWidget(abbr)
        hdr.addWidget(name_lbl, stretch=1)
        hdr.addWidget(del_btn)
        outer.addLayout(hdr)

        # ── Description ──
        desc = QLabel(model.description.split("\n")[0])
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size:8px;color:{TEXT_MUTED};line-height:130%;")
        outer.addWidget(desc)

        # ── Parameters ──
        if model.parameters:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"border-top:1px solid {BORDER};")
            outer.addWidget(sep)

            grid = QGridLayout()
            grid.setSpacing(4)
            grid.setColumnStretch(1, 1)

            for row, p in enumerate(model.parameters):
                lbl = QLabel(p.display_name or p.name)
                lbl.setStyleSheet(
                    f"font-size:9px;color:{TEXT_SECONDARY};font-weight:500;")
                grid.addWidget(lbl, row, 0)

                if p.type == "kernel":
                    w = KernelSelector(p.choices, int(p.default))
                    w.value_changed.connect(
                        lambda val, pn=p.name: self._update(pn, val))
                    self.params[p.name] = int(p.default)
                    grid.addWidget(w, row, 1, 1, 2)

                elif p.type == "choice":
                    w = QComboBox()
                    w.setStyleSheet(COMBO_SS)
                    w.addItems(p.choices)
                    w.setCurrentText(str(p.default))
                    w.currentTextChanged.connect(
                        lambda val, pn=p.name: self._update(pn, val))
                    grid.addWidget(w, row, 1, 1, 2)

                elif p.type == "bool":
                    w = QComboBox()
                    w.setStyleSheet(COMBO_SS)
                    w.addItems(["True", "False"])
                    w.setCurrentText(str(p.default))
                    w.currentTextChanged.connect(
                        lambda val, pn=p.name: self._update(pn, val == "True"))
                    grid.addWidget(w, row, 1, 1, 2)

                elif p.type == "float":
                    factor = 100
                    slider = QSlider(Qt.Orientation.Horizontal)
                    slider.setStyleSheet(SLIDER_SS)
                    slider.setRange(int(p.min_val * factor), int(p.max_val * factor))
                    slider.setValue(int(p.default * factor))
                    inp = QDoubleSpinBox()
                    inp.setStyleSheet(SPINBOX_SS)
                    inp.setRange(p.min_val, p.max_val)
                    inp.setSingleStep(p.step)
                    inp.setDecimals(2)
                    inp.setValue(p.default)
                    slider.valueChanged.connect(lambda v, i=inp, f=factor: i.setValue(v / f))
                    inp.valueChanged.connect(lambda v, s=slider, f=factor: s.setValue(int(v * f)))
                    inp.valueChanged.connect(lambda val, pn=p.name: self._update(pn, val))
                    grid.addWidget(slider, row, 1)
                    grid.addWidget(inp, row, 2)

                else:  # int
                    slider = QSlider(Qt.Orientation.Horizontal)
                    slider.setStyleSheet(SLIDER_SS)
                    slider.setRange(int(p.min_val), int(p.max_val))
                    slider.setSingleStep(max(1, int(p.step)))
                    slider.setValue(int(p.default))
                    inp = QSpinBox()
                    inp.setStyleSheet(SPINBOX_SS)
                    inp.setRange(int(p.min_val), int(p.max_val))
                    inp.setSingleStep(max(1, int(p.step)))
                    inp.setValue(int(p.default))
                    slider.valueChanged.connect(inp.setValue)
                    inp.valueChanged.connect(slider.setValue)
                    inp.valueChanged.connect(lambda val, pn=p.name: self._update(pn, val))
                    grid.addWidget(slider, row, 1)
                    grid.addWidget(inp, row, 2)

            outer.addLayout(grid)

        # ── Padding (kernel filters only) ──
        # After
        if model.has_padding:
            pad_sep = QFrame()
            pad_sep.setFrameShape(QFrame.Shape.HLine)
            pad_sep.setStyleSheet(f"border-top:1px solid {BORDER};")
            outer.addWidget(pad_sep)

            pad_row = QHBoxLayout()
            pad_row.setContentsMargins(0, 0, 0, 0)
            pad_lbl = QLabel("PAD")
            pad_lbl.setStyleSheet(
                f"font-size:8px;font-weight:700;color:{TEXT_MUTED};letter-spacing:0.5px;")
            pad_lbl.setFixedWidth(24)
            self._pad_sel = PaddingSelector()
            self._pad_sel.padding_changed.connect(
                lambda v: self._update("padding_mode", v))
            self.params["padding_mode"] = "zero"
            pad_row.addWidget(pad_lbl)
            pad_row.addWidget(self._pad_sel)
            outer.addLayout(pad_row)

        # ── View button ──
        self.view_btn = QPushButton("👁  View here")
        self.view_btn.setCheckable(True)
        view_group.addButton(self.view_btn)
        self.view_btn.setStyleSheet(self._view_ss(False))
        self.view_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.view_btn.toggled.connect(self._on_view_toggled)
        outer.addWidget(self.view_btn)

    # ── helpers ──────────────────────────────────────────────────────
    def _apply_style(self, active):
        border = f"2px solid {PRIMARY}" if active else f"1px solid {BORDER}"
        bg     = "#f8fbff" if active else BG_CARD
        self.setStyleSheet(f"""
            QFrame#PipeCard {{background:{bg};border:{border};border-radius:12px;}}
        """)

    def _view_ss(self, checked):
        if checked:
            return f"""QPushButton {{
                background:{PRIMARY};color:white;border:none;border-radius:6px;
                font-size:10px;font-weight:600;padding:4px 0;}}"""
        return f"""
            QPushButton {{
                background:{PRIMARY_LIGHT};color:{PRIMARY};border:none;
                border-radius:6px;font-size:10px;font-weight:500;padding:4px 0;}}
            QPushButton:hover{{background:#d2e3fc;}}"""

    def _update(self, name, value):
        self.params[name] = value
        self.params_changed.emit()

    def _on_view_toggled(self, checked):
        self.view_btn.setStyleSheet(self._view_ss(checked))
        self._apply_style(checked)
        if checked:
            self.view_requested.emit(self.uid)

    def set_active(self, active):
        if not active and self.view_btn.isChecked():
            self.view_btn.setChecked(False)
        self._apply_style(active)


# ──────────────────────────────────────────────────────────────────────
# Arrow separator
# ──────────────────────────────────────────────────────────────────────
class ArrowSep(QLabel):
    def __init__(self, parent=None):
        super().__init__("→", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"font-size:18px;color:{PRIMARY};margin:0 2px;")
        self.setFixedWidth(24)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)


# ──────────────────────────────────────────────────────────────────────
# Pipeline row (shared by both channels)
# ──────────────────────────────────────────────────────────────────────
class PipelineRow(QWidget):
    """
    A horizontal scrollable row of AppliedFilterCards with arrows.
    Emits pipeline_changed when cards are added/removed/modified.
    """
    pipeline_changed = pyqtSignal()
    view_requested   = pyqtSignal(str)

    def __init__(self, empty_hint: str, view_group: QButtonGroup, parent=None):
        super().__init__(parent)
        self._hint_text  = empty_hint
        self._view_group = view_group
        self.pipeline: list = []
        self.view_level_uid = None
        self._worker  = None
        self._pending = False

        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(192)
        scroll.setStyleSheet("""
            QScrollArea{background:transparent;border:none;}
            QScrollBar:horizontal{height:6px;background:#f0f4f8;border-radius:3px;}
            QScrollBar::handle:horizontal{background:#c5d9f7;border-radius:3px;min-width:30px;}
        """)

        self._inner = QWidget()
        self._inner.setStyleSheet("background:transparent;")
        self._row = QHBoxLayout(self._inner)
        self._row.setContentsMargins(4, 4, 4, 4)
        self._row.setSpacing(4)
        self._row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._empty_lbl = QLabel(f"  ← {self._hint_text}")
        self._empty_lbl.setStyleSheet(
            f"color:{TEXT_MUTED};font-size:11px;font-style:italic;")
        self._row.addWidget(self._empty_lbl)

        scroll.setWidget(self._inner)
        outer.addWidget(scroll)

    # ── public API ────────────────────────────────────────────────────
    def add_filter(self, model: AlgorithmModel):
        uid = os.urandom(4).hex()
        if self._empty_lbl.parent():
            self._empty_lbl.setParent(None)
        if self.pipeline:
            arrow = ArrowSep()
            self._row.addWidget(arrow)

        card = AppliedFilterCard(uid, model, self._view_group)
        shadow(card, 10, 2, 25)
        card.delete_clicked.connect(self.remove_filter)
        card.params_changed.connect(self.pipeline_changed.emit)
        card.view_requested.connect(self._on_view)
        self._row.addWidget(card)
        self.pipeline.append({"uid": uid, "model": model, "card": card, "result": None})
        card.view_btn.setChecked(True)
        self.pipeline_changed.emit()

    def remove_filter(self, uid):
        idx = next((i for i, it in enumerate(self.pipeline) if it["uid"] == uid), None)
        if idx is None:
            return
        item = self.pipeline.pop(idx)
        card = item["card"]
        ci   = self._row.indexOf(card)
        if ci >= 0:
            ai = ci - 1 if ci > 0 else ci + 1
            ai_item = self._row.itemAt(ai) if 0 <= ai < self._row.count() else None
            if ai_item and isinstance(ai_item.widget(), ArrowSep):
                ai_item.widget().deleteLater()
            self._row.removeWidget(card)
        card.deleteLater()
        if not self.pipeline:
            self._row.addWidget(self._empty_lbl)
            self._empty_lbl.show()
        if self.view_level_uid == uid:
            self.view_level_uid = None
            if self.pipeline:
                self.pipeline[-1]["card"].view_btn.setChecked(True)
        self.pipeline_changed.emit()

    def clear(self):
        for it in list(self.pipeline):
            self.remove_filter(it["uid"])

    def get_tasks(self):
        return [(it["model"].id, dict(it["card"].params)) for it in self.pipeline]

    def store_results(self, results):
        for item, res in zip(self.pipeline, results):
            item["result"] = res

    def get_view_image(self, fallback):
        if not self.pipeline:
            return fallback
        if self.view_level_uid:
            for it in self.pipeline:
                if it["uid"] == self.view_level_uid and it["result"] is not None:
                    return it["result"]
        for it in reversed(self.pipeline):
            if it["result"] is not None:
                return it["result"]
        return fallback

    def _on_view(self, uid):
        self.view_level_uid = uid
        self.view_requested.emit(uid)


# ──────────────────────────────────────────────────────────────────────
# Summary bar
# ──────────────────────────────────────────────────────────────────────
class SummaryBar(QFrame):
    def __init__(self, label="Filters", parent=None):
        super().__init__(parent)
        self.setObjectName("SBar")
        self.setStyleSheet(f"""
            QFrame#SBar {{
                background:{PRIMARY_LIGHT};border:1px solid #c5d9f7;border-radius:8px;
            }}
        """)
        self.setFixedHeight(34)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(6)
        icon = QLabel("⚙")
        icon.setStyleSheet(f"color:{PRIMARY};font-size:12px;")
        self._text = QLabel(f"No {label.lower()} applied")
        self._text.setStyleSheet(f"color:{PRIMARY};font-size:10px;font-weight:500;")
        layout.addWidget(icon)
        layout.addWidget(self._text, stretch=1)
        self._label = label

    def update_pipeline(self, pipeline):
        if not pipeline:
            self._text.setText(f"No {self._label.lower()} applied")
            return
        parts = []
        for it in pipeline:
            model  = it["model"]
            params = it["card"].params
            p_strs = []
            for p in model.parameters:
                val = params.get(p.name, p.default)
                if isinstance(val, float):
                    p_strs.append(f"{p.display_name}={val:.2f}")
                else:
                    p_strs.append(f"{p.display_name}={val}")
            tag = model.abbreviation
            if p_strs:
                tag += f"({', '.join(p_strs)})"
            parts.append(tag)
        self._text.setText("  →  ".join(parts))


# ──────────────────────────────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIP — Digital Image Processing Studio")
        self.setGeometry(80, 80, 1440, 900)
        self.setMinimumSize(1100, 680)
        self.setStyleSheet(f"QMainWindow,QWidget{{background:{BG_APP};}}")

        self.original_image = None
        self._noisy_image   = None   # result after noise channel
        self._filter_worker = None
        self._noise_worker  = None
        self._filter_pending = False
        self._noise_pending  = False

        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)

        self._build_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Left sidebar ──────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(296)
        sidebar.setObjectName("Sidebar")
        sidebar.setStyleSheet(f"""
            QWidget#Sidebar{{background:{BG_SIDEBAR};border-right:1px solid {BORDER};}}
        """)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(0)

        # Logo
        logo_bar = QWidget()
        logo_bar.setFixedHeight(52)
        logo_bar.setStyleSheet(f"background:{PRIMARY};border:none;")
        logo_row = QHBoxLayout(logo_bar)
        logo_row.setContentsMargins(16, 0, 16, 0)
        logo_row.addWidget(QLabel("⬡  DIP Studio", styleSheet=
            "color:white;font-size:15px;font-weight:700;"))
        sb.addWidget(logo_bar)

        # Load + Save buttons
        btn_bar = QWidget()
        btn_bar.setFixedHeight(52)
        btn_bar.setStyleSheet("background:transparent;")
        bb_row = QHBoxLayout(btn_bar)
        bb_row.setContentsMargins(10, 8, 10, 8)
        bb_row.setSpacing(6)
        self.load_btn = QPushButton("📂  Load Image")
        self.load_btn.setStyleSheet(self._primary_btn_ss())
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.clicked.connect(self._load_image)
        save_btn = QPushButton("💾  Save")
        save_btn.setStyleSheet(f"""
            QPushButton{{background:transparent;color:{ACCENT_GREEN};
                border:1px solid {ACCENT_GREEN};border-radius:8px;
                font-size:11px;font-weight:600;padding:6px 10px;}}
            QPushButton:hover{{background:#e6f4ea;}}
        """)
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._save_image)
        bb_row.addWidget(self.load_btn, stretch=1)
        bb_row.addWidget(save_btn)
        sb.addWidget(btn_bar)

        # Channel tab selector in sidebar
        ch_lbl = QLabel("CHANNEL")
        ch_lbl.setContentsMargins(16, 8, 0, 4)
        ch_lbl.setStyleSheet(
            f"font-size:9px;font-weight:700;color:{TEXT_MUTED};letter-spacing:1px;")
        sb.addWidget(ch_lbl)

        ch_row = QHBoxLayout()
        ch_row.setContentsMargins(10, 0, 10, 6)
        ch_row.setSpacing(6)
        self._ch_filter_btn = QPushButton("🔧  Filters")
        self._ch_noise_btn  = QPushButton("🌩  Noise")
        for btn in (self._ch_filter_btn, self._ch_noise_btn):
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            ch_row.addWidget(btn)
        self._ch_filter_btn.setChecked(True)
        self._ch_filter_btn.clicked.connect(lambda: self._switch_channel(0))
        self._ch_noise_btn.clicked.connect(lambda: self._switch_channel(1))
        self._update_channel_btns(0)
        sb.addLayout(ch_row)

        # Category label
        cat_lbl = QLabel("CATEGORIES")
        cat_lbl.setContentsMargins(16, 4, 0, 4)
        cat_lbl.setStyleSheet(
            f"font-size:9px;font-weight:700;color:{TEXT_MUTED};letter-spacing:1px;")
        sb.addWidget(cat_lbl)

        # Category grid
        cat_w = QWidget()
        cat_w.setStyleSheet("background:transparent;")
        from PyQt6.QtWidgets import QGridLayout
        cat_grid = QGridLayout(cat_w)
        cat_grid.setContentsMargins(10, 0, 10, 6)
        cat_grid.setSpacing(6)
        self._cat_buttons = []
        self._current_channel = 0

        # Separate categories by channel
        self._filter_cats = [c for c in CATEGORIES if c != "Noise"]
        self._noise_cats  = ["Noise"]
        self._all_cats    = CATEGORIES

        for i, cat in enumerate(self._filter_cats):
            btn = QPushButton(cat)
            btn.setCheckable(True)
            btn.setMinimumHeight(30)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._cat_ss(False, cat))
            btn.clicked.connect(lambda _, c=cat: self._select_category(c))
            self._cat_buttons.append(btn)
            cat_grid.addWidget(btn, i // 2, i % 2)
        sb.addWidget(cat_w)

        self._cat_container = cat_w

        # Filters label
        self._filters_lbl = QLabel("FILTERS")
        self._filters_lbl.setContentsMargins(16, 2, 0, 4)
        self._filters_lbl.setStyleSheet(
            f"font-size:9px;font-weight:700;color:{TEXT_MUTED};letter-spacing:1px;")
        sb.addWidget(self._filters_lbl)

        # Available scroll
        avail_scroll = QScrollArea()
        avail_scroll.setWidgetResizable(True)
        avail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        avail_scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        avail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._avail_widget = QWidget()
        self._avail_widget.setStyleSheet("background:transparent;")
        self._avail_layout = QVBoxLayout(self._avail_widget)
        self._avail_layout.setContentsMargins(6, 0, 6, 12)
        self._avail_layout.setSpacing(4)
        self._avail_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        avail_scroll.setWidget(self._avail_widget)
        sb.addWidget(avail_scroll, stretch=1)

        # ── Right content ─────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet("background:transparent;")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(14, 12, 14, 10)
        cl.setSpacing(10)

        # ── Image row ──
        img_row = QHBoxLayout()
        img_row.setSpacing(10)

        in_col = QVBoxLayout()
        in_col.setSpacing(3)
        in_col.addWidget(self._sec_lbl("Input Image"))
        self._img_in = ImageDisplay("Load an image to begin")
        self._img_in.setMinimumHeight(200)
        in_col.addWidget(self._img_in, stretch=1)

        # Noisy image column
        noise_col = QVBoxLayout()
        noise_col.setSpacing(3)
        self._noise_col_lbl = self._sec_lbl("Noisy Image")
        noise_col.addWidget(self._noise_col_lbl)
        self._img_noisy = ImageDisplay("Add noise to see result")
        self._img_noisy.setMinimumHeight(200)
        noise_col.addWidget(self._img_noisy, stretch=1)

        out_col = QVBoxLayout()
        out_col.setSpacing(3)
        self._out_lbl = self._sec_lbl("Output Image")
        out_col.addWidget(self._out_lbl)
        self._img_out = ImageDisplay("Apply a filter to see output")
        self._img_out.setMinimumHeight(200)
        out_col.addWidget(self._img_out, stretch=1)

        img_row.addLayout(in_col, stretch=1)
        img_row.addLayout(noise_col, stretch=1)
        img_row.addLayout(out_col, stretch=1)
        cl.addLayout(img_row, stretch=3)

        # ── Histogram row ──
        hist_row = QHBoxLayout()
        hist_row.setSpacing(10)
        for attr, title in (("_hist_in",  "Input Histogram"),
                            ("_hist_noisy","Noisy Histogram"),
                            ("_hist_out", "Output Histogram")):
            card = self._make_card()
            shadow(card, 8, 1, 20)
            lay = QVBoxLayout(card)
            lay.setContentsMargins(8, 4, 8, 4)
            lay.setSpacing(2)
            lay.addWidget(self._sec_lbl(title, small=True))
            hc = HistogramCanvas()
            lay.addWidget(hc)
            setattr(self, attr, hc)
            hist_row.addWidget(card, stretch=1)
        cl.addLayout(hist_row, stretch=1)

        # ── Pipeline tabs ──
        pipe_tabs = QTabWidget()
        pipe_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border:1px solid {BORDER};border-radius:10px;
                background:{BG_CARD};margin-top:-1px;
            }}
            QTabBar::tab {{
                background:#eef2f7;border:1px solid {BORDER};
                border-bottom:none;border-radius:6px 6px 0 0;
                padding:5px 14px;font-size:10px;font-weight:600;
                color:{TEXT_SECONDARY};margin-right:3px;
            }}
            QTabBar::tab:selected {{
                background:{BG_CARD};color:{PRIMARY};
                border-bottom:1px solid {BG_CARD};
            }}
        """)

        # Noise channel tab
        noise_tab = QWidget()
        nt_lay = QVBoxLayout(noise_tab)
        nt_lay.setContentsMargins(8, 8, 8, 6)
        nt_lay.setSpacing(4)
        nt_hdr = QHBoxLayout()
        nt_hdr.addWidget(self._sec_lbl("Noise Pipeline"))
        nt_hdr.addStretch()
        n_clear = QPushButton("Clear All")
        n_clear.setStyleSheet(self._clear_btn_ss())
        n_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        n_clear.clicked.connect(lambda: (self._noise_row.clear(),
                                          self._run_noise()))
        nt_hdr.addWidget(n_clear)
        nt_lay.addLayout(nt_hdr)

        noise_card = self._make_card()
        shadow(noise_card, 8, 1, 20)
        nc_lay = QVBoxLayout(noise_card)
        nc_lay.setContentsMargins(6, 6, 6, 6)
        self._noise_row = PipelineRow(
            "Select 'Noise' category, then click + to add noise",
            self._view_group)
        self._noise_row.pipeline_changed.connect(self._run_noise)
        self._noise_row.view_requested.connect(
            lambda _: self._on_noise_results([it["result"] for it in self._noise_row.pipeline])
        )
        nc_lay.addWidget(self._noise_row)
        nt_lay.addWidget(noise_card)
        self._noise_summary = SummaryBar("Noise")
        nt_lay.addWidget(self._noise_summary)

        # Filter channel tab
        filter_tab = QWidget()
        ft_lay = QVBoxLayout(filter_tab)
        ft_lay.setContentsMargins(8, 8, 8, 6)
        ft_lay.setSpacing(4)
        ft_hdr = QHBoxLayout()
        ft_hdr.addWidget(self._sec_lbl("Filter Pipeline"))
        ft_hdr.addStretch()
        f_clear = QPushButton("Clear All")
        f_clear.setStyleSheet(self._clear_btn_ss())
        f_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        f_clear.clicked.connect(lambda: (self._filter_row.clear(),
                                          self._run_filters()))
        ft_hdr.addWidget(f_clear)
        ft_lay.addLayout(ft_hdr)

        filter_card = self._make_card()
        shadow(filter_card, 8, 1, 20)
        fc_lay = QVBoxLayout(filter_card)
        fc_lay.setContentsMargins(6, 6, 6, 6)
        self._filter_row = PipelineRow(
            "Select a category, then click + to add a filter",
            self._view_group)
        self._filter_row.pipeline_changed.connect(self._run_filters)
        self._filter_row.view_requested.connect(
            lambda _: self._on_filter_results([it["result"] for it in self._filter_row.pipeline])
        )
        fc_lay.addWidget(self._filter_row)
        ft_lay.addWidget(filter_card)
        self._filter_summary = SummaryBar("Filters")
        ft_lay.addWidget(self._filter_summary)

        pipe_tabs.addTab(noise_tab,  "🌩  Noise Channel")
        pipe_tabs.addTab(filter_tab, "🔧  Filter Channel")
        pipe_tabs.currentChanged.connect(self._on_tab_changed)
        self._pipe_tabs = pipe_tabs
        cl.addWidget(pipe_tabs)

        root_layout.addWidget(sidebar)
        root_layout.addWidget(content, stretch=1)

        # Default category
        self._select_category(self._filter_cats[0])

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _sec_lbl(text, small=False):
        lbl = QLabel(text)
        sz  = 10 if small else 12
        lbl.setStyleSheet(
            f"font-size:{sz}px;font-weight:700;color:{TEXT_PRIMARY};background:transparent;")
        return lbl

    @staticmethod
    def _make_card():
        card = QFrame()
        card.setObjectName("Card")
        card.setStyleSheet(f"""
            QFrame#Card{{background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;}}
        """)
        return card

    @staticmethod
    def _primary_btn_ss():
        return f"""
            QPushButton{{background:{PRIMARY};color:white;border:none;border-radius:8px;
                font-size:11px;font-weight:600;padding:6px 12px;}}
            QPushButton:hover{{background:{PRIMARY_DARK};}}
            QPushButton:pressed{{background:#0d47a1;}}"""

    @staticmethod
    def _clear_btn_ss():
        return f"""
            QPushButton{{background:transparent;color:{ACCENT_RED};
                border:1px solid {ACCENT_RED};border-radius:6px;
                font-size:10px;padding:3px 10px;}}
            QPushButton:hover{{background:#fce8e6;}}"""

    @staticmethod
    def _cat_ss(checked, cat=""):
        bg, fg = CAT_COLORS.get(cat, ("#e0e0e0", "#555"))
        if checked:
            return f"""
                QPushButton{{background:{bg};color:{fg};border:1.5px solid {fg};
                    border-radius:8px;font-size:9px;font-weight:700;padding:4px 6px;}}"""
        return f"""
            QPushButton{{background:#fff;color:{TEXT_SECONDARY};border:1px solid {BORDER};
                border-radius:8px;font-size:9px;font-weight:500;padding:4px 6px;}}
            QPushButton:hover{{background:{bg};color:{fg};border-color:{fg};}}"""

    # ──────────────────────────────────────────────────────────────────
    # Channel switching
    # ──────────────────────────────────────────────────────────────────
    def _switch_channel(self, idx):
        self._current_channel = idx
        self._update_channel_btns(idx)
        # Show appropriate categories
        cats = self._noise_cats if idx == 1 else self._filter_cats
        # Rebuild category buttons
        for btn in self._cat_buttons:
            btn.setParent(None)
        self._cat_buttons.clear()

        from PyQt6.QtWidgets import QGridLayout
        # Remove old cat container layout items
        old_lay = self._cat_container.layout()
        while old_lay.count():
            item = old_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, cat in enumerate(cats):
            btn = QPushButton(cat)
            btn.setCheckable(True)
            btn.setMinimumHeight(30)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._cat_ss(False, cat))
            btn.clicked.connect(lambda _, c=cat: self._select_category(c))
            self._cat_buttons.append(btn)
            old_lay.addWidget(btn, i // 2, i % 2)

        if cats:
            self._select_category(cats[0])

        # Switch pipeline tab to match
        self._pipe_tabs.setCurrentIndex(1 - idx)  # noise=tab0, filter=tab1

    def _update_channel_btns(self, idx):
        active_ss = f"""QPushButton{{
            background:{PRIMARY};color:white;border:none;border-radius:8px;
            font-size:10px;font-weight:700;padding:4px 8px;}}"""
        inactive_ss = f"""QPushButton{{
            background:#eef2f7;color:{TEXT_SECONDARY};border:1px solid {BORDER};
            border-radius:8px;font-size:10px;font-weight:500;padding:4px 8px;}}
            QPushButton:hover{{background:{PRIMARY_LIGHT};color:{PRIMARY};}}"""
        self._ch_filter_btn.setStyleSheet(active_ss if idx == 0 else inactive_ss)
        self._ch_noise_btn.setStyleSheet(active_ss if idx == 1 else inactive_ss)

    def _on_tab_changed(self, idx):
        # tabs: 0=noise, 1=filter → channel: 0=filter, 1=noise
        self._switch_channel(1 - idx)

    # ──────────────────────────────────────────────────────────────────
    # Category selection
    # ──────────────────────────────────────────────────────────────────
    def _select_category(self, category):
        for btn in self._cat_buttons:
            is_sel = btn.text() == category
            btn.setChecked(is_sel)
            btn.setStyleSheet(self._cat_ss(is_sel, btn.text()))

        while self._avail_layout.count():
            item = self._avail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for alg in ALGORITHMS:
            if alg.category == category:
                card = AvailableFilterCard(alg)
                card.add_clicked.connect(self._add_to_active_channel)
                self._avail_layout.addWidget(card)

    # ──────────────────────────────────────────────────────────────────
    # Add to the correct pipeline
    # ──────────────────────────────────────────────────────────────────
    def _add_to_active_channel(self, model: AlgorithmModel):
        if model.category == "Noise":
            self._noise_row.add_filter(model)
        else:
            self._filter_row.add_filter(model)

    # ──────────────────────────────────────────────────────────────────
    # Image loading / saving
    # ──────────────────────────────────────────────────────────────────
    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._noisy_image   = None
        self._img_in.set_image(self.original_image)
        self._hist_in.plot(self.original_image)
        self._img_noisy.set_image(None)
        self._hist_noisy.plot(None)
        self._run_noise()

    def _save_image(self):
        # Save the final filter output, or noisy if no filters
        out = self._filter_row.get_view_image(
              self._noise_row.get_view_image(self.original_image))
        if out is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Image", "output.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff);;BMP (*.bmp)")
        if not path:
            return
        if out.dtype != np.uint8:
            out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if not cv2.imwrite(path, out):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Save Failed",
                                f"Could not write to:\n{path}")

    # ──────────────────────────────────────────────────────────────────
    # Noise pipeline
    # ──────────────────────────────────────────────────────────────────
    def _run_noise(self):
        if self.original_image is None:
            return
        if not self._noise_row.pipeline:
            self._noisy_image = None
            self._img_noisy.set_image(None)
            self._hist_noisy.plot(None)
            self._noise_col_lbl.setText("Noisy Image")
            self._noise_summary.update_pipeline([])
            self._run_filters()
            return
        if self._noise_worker and self._noise_worker.isRunning():
            self._noise_pending = True
            return
        tasks = self._noise_row.get_tasks()
        self._noise_worker = PipelineWorker(self.original_image, tasks)
        self._noise_worker.result_ready.connect(self._on_noise_results)
        self._noise_worker.start()

    @pyqtSlot(list)
    def _on_noise_results(self, results):
        self._noise_row.store_results(results)
        noisy = self._noise_row.get_view_image(self.original_image)
        self._noisy_image = noisy
        self._img_noisy.set_image(noisy)
        self._hist_noisy.plot(noisy)

        if self._noise_row.view_level_uid:
            for it in self._noise_row.pipeline:
                if it["uid"] == self._noise_row.view_level_uid:
                    self._noise_col_lbl.setText(f"Noisy — {it['model'].name}")
                    break
        else:
            names = " + ".join(it["model"].name for it in self._noise_row.pipeline)
            if names:
                self._noise_col_lbl.setText(f"Noisy — {names}")
            else:
                self._noise_col_lbl.setText("Noisy Image")
        self._noise_summary.update_pipeline(self._noise_row.pipeline)
        if self._noise_pending:
            self._noise_pending = False
            self._run_noise()
        else:
            self._run_filters()

    # ──────────────────────────────────────────────────────────────────
    # Filter pipeline — runs on the noisy image (or original if no noise)
    # ──────────────────────────────────────────────────────────────────
    def _run_filters(self):
        source = self._noisy_image if self._noisy_image is not None \
                 else self.original_image
        if source is None:
            return
        if not self._filter_row.pipeline:
            self._img_out.set_image(source)
            self._hist_out.plot(source)
            self._out_lbl.setText("Output Image")
            self._filter_summary.update_pipeline([])
            return
        if self._filter_worker and self._filter_worker.isRunning():
            self._filter_pending = True
            return
        tasks = self._filter_row.get_tasks()
        self._filter_worker = PipelineWorker(source, tasks)
        self._filter_worker.result_ready.connect(self._on_filter_results)
        self._filter_worker.start()

    @pyqtSlot(list)
    def _on_filter_results(self, results):
        self._filter_row.store_results(results)
        out = self._filter_row.get_view_image(
            self._noisy_image if self._noisy_image is not None
            else self.original_image)
        self._img_out.set_image(out)
        self._hist_out.plot(out)

        if self._filter_row.view_level_uid:
            for it in self._filter_row.pipeline:
                if it["uid"] == self._filter_row.view_level_uid:
                    self._out_lbl.setText(f"Output — {it['model'].name}")
                    break
        else:
            self._out_lbl.setText("Output Image")

        self._filter_summary.update_pipeline(self._filter_row.pipeline)

        if self._filter_pending:
            self._filter_pending = False
            self._run_filters()


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(BG_APP))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base,            QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(BG_SIDEBAR))
    palette.setColor(QPalette.ColorRole.Text,            QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button,          QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(PRIMARY))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())