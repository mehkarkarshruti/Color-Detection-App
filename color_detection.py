"""
Real-Time Color Detection App
==============================
Click anywhere on the webcam feed OR a loaded image
to identify the color name, RGB values, and HEX code.

Author  : Shruti Mehkarkar (23BAI10242)
Course  : Computer Vision (CSE3010)

Controls:
  Click       → Detect color at that pixel
  S           → Switch between Webcam and Image mode
  O           → Open an image file (in Image mode)
  C           → Clear last detected color
  Q / ESC     → Quit
"""

import cv2
import numpy as np
import os
import sys

# ── Color Database ────────────────────────────────────────────────────────────
COLORS = {
    "Red":               (255, 0, 0),
    "Dark Red":          (139, 0, 0),
    "Crimson":           (220, 20, 60),
    "Tomato":            (255, 99, 71),
    "Coral":             (255, 127, 80),
    "Salmon":            (250, 128, 114),
    "Light Salmon":      (255, 160, 122),
    "Orange Red":        (255, 69, 0),
    "Orange":            (255, 165, 0),
    "Dark Orange":       (255, 140, 0),
    "Gold":              (255, 215, 0),
    "Yellow":            (255, 255, 0),
    "Light Yellow":      (255, 255, 224),
    "Lemon Chiffon":     (255, 250, 205),
    "Yellow Green":      (154, 205, 50),
    "Green Yellow":      (173, 255, 47),
    "Chartreuse":        (127, 255, 0),
    "Lime":              (0, 255, 0),
    "Lime Green":        (50, 205, 50),
    "Pale Green":        (152, 251, 152),
    "Light Green":       (144, 238, 144),
    "Spring Green":      (0, 255, 127),
    "Medium Sea Green":  (60, 179, 113),
    "Sea Green":         (46, 139, 87),
    "Forest Green":      (34, 139, 34),
    "Green":             (0, 128, 0),
    "Dark Green":        (0, 100, 0),
    "Olive":             (128, 128, 0),
    "Dark Olive Green":  (85, 107, 47),
    "Teal":              (0, 128, 128),
    "Aqua":              (0, 255, 255),
    "Cyan":              (0, 255, 255),
    "Light Cyan":        (224, 255, 255),
    "Pale Turquoise":    (175, 238, 238),
    "Aquamarine":        (127, 255, 212),
    "Turquoise":         (64, 224, 208),
    "Dark Turquoise":    (0, 206, 209),
    "Cadet Blue":        (95, 158, 160),
    "Steel Blue":        (70, 130, 180),
    "Light Blue":        (173, 216, 230),
    "Sky Blue":          (135, 206, 235),
    "Deep Sky Blue":     (0, 191, 255),
    "Dodger Blue":       (30, 144, 255),
    "Cornflower Blue":   (100, 149, 237),
    "Royal Blue":        (65, 105, 225),
    "Blue":              (0, 0, 255),
    "Medium Blue":       (0, 0, 205),
    "Dark Blue":         (0, 0, 139),
    "Navy":              (0, 0, 128),
    "Midnight Blue":     (25, 25, 112),
    "Indigo":            (75, 0, 130),
    "Slate Blue":        (106, 90, 205),
    "Blue Violet":       (138, 43, 226),
    "Dark Violet":       (148, 0, 211),
    "Purple":            (128, 0, 128),
    "Violet":            (238, 130, 238),
    "Magenta":           (255, 0, 255),
    "Orchid":            (218, 112, 214),
    "Deep Pink":         (255, 20, 147),
    "Hot Pink":          (255, 105, 180),
    "Light Pink":        (255, 182, 193),
    "Pink":              (255, 192, 203),
    "Lavender":          (230, 230, 250),
    "Thistle":           (216, 191, 216),
    "Plum":              (221, 160, 221),
    "Antique White":     (250, 235, 215),
    "Beige":             (245, 245, 220),
    "Bisque":            (255, 228, 196),
    "Wheat":             (245, 222, 179),
    "White":             (255, 255, 255),
    "White Smoke":       (245, 245, 245),
    "Gainsboro":         (220, 220, 220),
    "Light Gray":        (211, 211, 211),
    "Silver":            (192, 192, 192),
    "Dark Gray":         (169, 169, 169),
    "Gray":              (128, 128, 128),
    "Dim Gray":          (105, 105, 105),
    "Slate Gray":        (112, 128, 144),
    "Dark Slate Gray":   (47, 79, 79),
    "Black":             (0, 0, 0),
    "Brown":             (165, 42, 42),
    "Saddle Brown":      (139, 69, 19),
    "Sienna":            (160, 82, 45),
    "Chocolate":         (210, 105, 30),
    "Goldenrod":         (218, 165, 32),
    "Tan":               (210, 180, 140),
    "Burlywood":         (222, 184, 135),
    "Sandy Brown":       (244, 164, 96),
    "Peru":              (205, 133, 63),
    "Rosy Brown":        (188, 143, 143),
    "Maroon":            (128, 0, 0),
    "Ivory":             (255, 255, 240),
    "Snow":              (255, 250, 250),
    "Honeydew":          (240, 255, 240),
    "Azure":             (240, 255, 255),
    "Alice Blue":        (240, 248, 255),
    "Ghost White":       (248, 248, 255),
    "Papaya Whip":       (255, 239, 213),
    "Peach Puff":        (255, 218, 185),
    "Moccasin":          (255, 228, 181),
    "Navajo White":      (255, 222, 173),
    "Khaki":             (240, 230, 140),
    "Dark Khaki":        (189, 183, 107),
    "Skin":              (255, 220, 177),
    "Dark Skin":         (139, 90, 43),
}

# ── Color matching ────────────────────────────────────────────────────────────
def get_color_name(r, g, b):
    min_dist, best = float('inf'), "Unknown"
    for name, (cr, cg, cb) in COLORS.items():
        d = (r-cr)**2 + (g-cg)**2 + (b-cb)**2
        if d < min_dist:
            min_dist, best = d, name
    return best

def get_text_color(r, g, b):
    return (0,0,0) if (0.299*r + 0.587*g + 0.114*b) > 128 else (255,255,255)

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, r, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(overlay, (cx,cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_header(frame, mode, image_name=""):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,52), (20,20,20), -1)

    # Mode badge
    badge_color = (0, 180, 80) if mode == "WEBCAM" else (0, 140, 220)
    cv2.rectangle(frame, (12, 10), (110, 42), badge_color, -1)
    cv2.putText(frame, mode, (18, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # Title
    title = "Color Detection App"
    if mode == "IMAGE" and image_name:
        title += f"  —  {image_name}"
    cv2.putText(frame, title, (122, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 210, 255), 2, cv2.LINE_AA)

    # Hints
    hints = "Click=detect  S=switch mode  O=open image  C=clear  Q=quit"
    cv2.putText(frame, hints, (w-620, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1, cv2.LINE_AA)

def draw_info_panel(frame, x, y, r, g, b, color_name):
    h, w    = frame.shape[:2]
    hex_code = f"#{r:02X}{g:02X}{b:02X}"
    tc       = get_text_color(r, g, b)

    pw, ph = 300, 110
    px = min(x + 18, w - pw - 8)
    py = max(y - ph - 18, 58)

    # Swatch panel
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (int(b), int(g), int(r)), -1)
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (80,80,80), 2)

    cv2.putText(frame, color_name,      (px+12, py+34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, tc, 2, cv2.LINE_AA)
    cv2.putText(frame, f"RGB  ({r}, {g}, {b})", (px+12, py+62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, tc, 1, cv2.LINE_AA)
    cv2.putText(frame, f"HEX  {hex_code}",       (px+12, py+86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, tc, 1, cv2.LINE_AA)

    # Click marker
    cv2.circle(frame, (x, y), 9,  (int(b), int(g), int(r)), -1)
    cv2.circle(frame, (x, y), 9,  (255,255,255), 2)
    cv2.circle(frame, (x, y), 2,  (0,0,0), -1)

    # Connector line
    lx = px if x < px else px + pw
    cv2.line(frame, (x, y), (lx, py + ph//2), (200,200,200), 1, cv2.LINE_AA)

def draw_crosshair(frame, x, y):
    if x < 0 or y < 52:
        return
    cv2.line(frame, (x-18, y), (x+18, y), (255,255,255), 1, cv2.LINE_AA)
    cv2.line(frame, (x, y-18), (x, y+18), (255,255,255), 1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 4, (255,255,255), 1, cv2.LINE_AA)

def draw_no_image_screen(frame):
    h, w = frame.shape[:2]
    frame[:] = (30, 30, 30)
    cv2.putText(frame, "No image loaded.", (w//2 - 160, h//2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100,100,100), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press  O  to open an image file.",
                (w//2 - 230, h//2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 1, cv2.LINE_AA)

# ── Image loader (simple file dialog via tkinter) ─────────────────────────────
def open_image_dialog():
    """Open a file dialog and return the loaded image + filename."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("All files", "*.*")]
        )
        root.destroy()
        if path:
            img = cv2.imread(path)
            if img is not None:
                return img, os.path.basename(path)
    except Exception as e:
        print(f"File dialog error: {e}")
    return None, ""

# ── State ─────────────────────────────────────────────────────────────────────
mouse_x = mouse_y = 0
click_x = click_y = -1
clicked_r = clicked_g = clicked_b = 0
color_name = ""
has_click  = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, click_x, click_y, has_click
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN and y > 52:
        click_x, click_y = x, y
        has_click = True

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global click_x, click_y, clicked_r, clicked_g, clicked_b
    global color_name, has_click, mouse_x, mouse_y

    cap        = cv2.VideoCapture(0)
    mode       = "WEBCAM"       # "WEBCAM" or "IMAGE"
    loaded_img = None
    img_name   = ""
    display_w, display_h = 1280, 720

    WIN = "Color Detection App  |  S=switch  O=open image  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, display_w, display_h)
    cv2.setMouseCallback(WIN, mouse_callback)

    print("\nColor Detection App started.")
    print("  Click  = detect color")
    print("  S      = switch Webcam / Image mode")
    print("  O      = open an image file")
    print("  C      = clear detection")
    print("  Q/ESC  = quit\n")

    while True:
        # ── Build base frame ──────────────────────────────────────────────────
        if mode == "WEBCAM":
            ok, raw = cap.read()
            if not ok:
                raw = np.zeros((display_h, display_w, 3), dtype=np.uint8)
            frame = cv2.flip(raw, 1)
            frame = cv2.resize(frame, (display_w, display_h))

        else:  # IMAGE mode
            if loaded_img is not None:
                # Fit image into display window (letterbox)
                ih, iw = loaded_img.shape[:2]
                scale  = min(display_w / iw, (display_h - 52) / ih)
                nw, nh = int(iw * scale), int(ih * scale)
                resized = cv2.resize(loaded_img, (nw, nh))
                frame   = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                ox = (display_w - nw) // 2
                oy = 52 + ((display_h - 52 - nh) // 2)
                frame[oy:oy+nh, ox:ox+nw] = resized
            else:
                frame = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                draw_no_image_screen(frame)

        # ── Sample pixel on click ─────────────────────────────────────────────
        if has_click:
            fh, fw = frame.shape[:2]
            cx = min(max(click_x, 0), fw-1)
            cy = min(max(click_y, 52), fh-1)
            bgr        = frame[cy, cx]
            clicked_b, clicked_g, clicked_r = int(bgr[0]), int(bgr[1]), int(bgr[2])
            color_name = get_color_name(clicked_r, clicked_g, clicked_b)

        # ── Draw UI ───────────────────────────────────────────────────────────
        draw_header(frame, mode, img_name)
        draw_crosshair(frame, mouse_x, mouse_y)

        if has_click:
            draw_info_panel(frame, click_x, click_y,
                            clicked_r, clicked_g, clicked_b, color_name)

        cv2.imshow(WIN, frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(5) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break

        elif key in (ord('s'), ord('S')):
            # Toggle mode
            mode = "IMAGE" if mode == "WEBCAM" else "WEBCAM"
            has_click = False
            print(f"Switched to {mode} mode.")

        elif key in (ord('o'), ord('O')):
            # Open image file
            img, name = open_image_dialog()
            if img is not None:
                loaded_img = img
                img_name   = name
                mode       = "IMAGE"
                has_click  = False
                print(f"Loaded image: {name}")
            else:
                print("No image selected.")

        elif key in (ord('c'), ord('C')):
            has_click = False
            print("Detection cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")

if __name__ == "__main__":
    main()
