# Real-Time Color Detection App

> Click anywhere on your live webcam feed **or any loaded image** to instantly identify the color name, RGB values, and HEX code at that pixel.

---

## Problem Statement

Identifying colors precisely is a challenge for designers, painters, and — most critically — people with color blindness. This app provides an instant, accessible way to identify any color visible through a camera or in an image file, returning a human-readable name along with its RGB and HEX representation.

---

## Features

- **Click-to-detect** — click any pixel on the live feed or image
- **Dual mode** — switch between Webcam and Image mode with a single key
- **Image file support** — open any JPG, PNG, BMP, or WEBP file via file dialog
- **130+ named colors** — matched by nearest Euclidean distance in RGB space
- **Full color info** — color name, RGB values, and HEX code
- **Live color swatch** — panel background fills with the exact detected color
- **Zero extra dependencies** — only OpenCV needed, no model downloads

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- A working webcam (for webcam mode)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/mehkarkarshruti/Color-Detection-App.git
cd color-detection

# 2. Install the only dependency
pip install opencv-python

# 3. Run
python color_detection.py
```

---

## Controls

| Key / Action | Result |
|---|---|
| **Click** on any pixel | Detect and name the color |
| **S** | Switch between Webcam ↔ Image mode |
| **O** | Open a file dialog to load an image |
| **C** | Clear the last detected color |
| **Q** or **ESC** | Quit the app |

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│              INPUT SOURCE (toggle with S)           │
│                                                     │
│   Webcam Mode              Image Mode               │
│   cv2.VideoCapture(0)      cv2.imread(filepath)     │
│         │                        │                  │
│         └──────────┬─────────────┘                  │
│                    ▼                                │
│            Mouse Click Event                        │
│                    │                                │
│                    ▼                                │
│         Sample pixel (B, G, R) at (x, y)            │
│                    │                                │
│                    ▼                                │
│            Convert BGR → RGB                        │
│                    │                                │
│                    ▼                                │
│     Euclidean distance against 130+ named colors    │
│     dist = √((R−Rc)² + (G−Gc)² + (B−Bc)²)           │
│                    │                                │
│                    ▼                                │
│       Return closest color name + RGB + HEX         │
│                    │                                │
│                    ▼                                │
│      Draw color swatch panel on frame               │
└─────────────────────────────────────────────────────┘
```

### Color Matching
Each pixel's RGB value is compared against a database of 130+ named colors using **Euclidean distance** in RGB space — equivalent to a 1-Nearest Neighbor classifier in 3D color space.

### Image Mode
Images are loaded via a native OS file dialog (tkinter). The image is letterbox-scaled to fit the display window while preserving aspect ratio. Clicking any pixel on the scaled image samples the correct underlying color.

### Text Contrast
The info panel background is filled with the detected color. Text color (black or white) is chosen automatically using the WCAG perceptual brightness formula: `L = 0.299R + 0.587G + 0.114B`.

---

## Project Structure

```
color-detection/
│
├── color_detection.py   # Complete application — run this
└── README.md            # This file
```

---

## Demo

```
┌──────────────────────────────────────────────────────────────┐
│ [WEBCAM]  Color Detection App    Click=detect S=switch Q=quit│
│──────────────────────────────────────────────────────────────│
│                                                              │
│   [live webcam / loaded image]                               │
│                                   ┌──────────────────────┐   │
│                      ✚ ●──────────│  Sky Blue            │  │
│                                   │  RGB  (135, 206, 235) │  │
│                                   │  HEX  #87CEEB         │  │
│                                   └──────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `opencv-python` | Webcam capture, mouse events, frame rendering |
| `numpy` | Euclidean distance computation |
| `tkinter` | Native OS file dialog (built into Python) |

*(numpy is installed automatically with opencv-python. tkinter is built into Python.)*

---

## Future Work

- LAB color space matching for perceptually uniform results
- Region-average mode — drag a box to detect dominant color via k-means
- Export detected colors to CSS / JSON palette file
- Colorblind simulation overlay mode
- Pantone / RAL color system support

---

## License

MIT License — free for educational and personal use.
