import cv2 as cv
import numpy as np
from pathlib import Path

# ---------- Settings ----------
IMAGE_NAME = "foto3.jpg"
ANSWER_KEY_NAME = "respuestas.txt"   # one correct answer (A-E) per line, line 1 = question 1
PHOTO_ROTATION = None   # None | cv.ROTATE_90_CLOCKWISE | cv.ROTATE_90_COUNTERCLOCKWISE | cv.ROTATE_180

# Photos bigger than this (longest side) are shrunk before processing
MAX_WORK_SIDE = 1600

# --- Template geometry (measured on plantilla.png, in template pixels) ---
# Centers of the 4 big black squares, ordered TL, TR, BR, BL
TEMPLATE_MARKERS = np.array([[191, 191], [1211, 191], [1211, 1513], [191, 1513]], dtype="float32")
# Outer border of the answer box: x0, y0, x1, y1
TEMPLATE_BOX = (220, 412, 1185, 1425)

# The answer box is always rebuilt at this fixed size, whatever the photo's resolution,
# aspect ratio or perspective. MIN_SIZE / MAX_SIZE / kernels / margins refer to this size.
CANON_W = 360
BOX_W = TEMPLATE_BOX[2] - TEMPLATE_BOX[0]
BOX_H = TEMPLATE_BOX[3] - TEMPLATE_BOX[1]
CANON_H = int(round(CANON_W * BOX_H / BOX_W))
SUPERSAMPLE = 3          # warp at 3x and shrink with INTER_AREA (avoids aliasing)

# Marker detection (fractions of the photo width)
MARKER_MIN, MARKER_MAX = 0.008, 0.08
ASPECT_TOLERANCE = 0.30  # allowed deviation of the marker quad from the template proportions

# ---------- Load image ----------
BASE_DIR = Path(__file__).resolve().parent
image_path = BASE_DIR / IMAGE_NAME
key_path = BASE_DIR / ANSWER_KEY_NAME

TOTAL_QUESTIONS = 20
OPTIONS = "ABCDE"


def cargar_clave(path):
    """Read the answer key: one letter per line, line N = correct answer of question N."""
    if not path.exists():
        raise FileNotFoundError(f"Answer key not found at: {path}")

    # utf-8-sig also handles the invisible BOM that Windows Notepad may add
    lineas = path.read_text(encoding="utf-8-sig").splitlines()
    while lineas and not lineas[-1].strip():      # ignore blank lines at the end of the file
        lineas.pop()

    if len(lineas) != TOTAL_QUESTIONS:
        raise ValueError(f"{path.name} must have exactly {TOTAL_QUESTIONS} lines (one per question), "
                         f"but it has {len(lineas)}")

    clave = {}
    for n, linea in enumerate(lineas, start=1):
        letra = linea.strip().upper()
        if letra not in tuple(OPTIONS):
            raise ValueError(f"{path.name}, line {n}: '{linea.strip()}' is not a valid answer "
                             f"(use one of {', '.join(OPTIONS)})")
        clave[n] = letra
    return clave


answer_key = cargar_clave(key_path)

imagen_original = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
if imagen_original is None:
    raise FileNotFoundError(f"Could not read image at: {image_path}")

if PHOTO_ROTATION is not None:
    imagen_original = cv.rotate(imagen_original, PHOTO_ROTATION)

h_img, w_img = imagen_original.shape
if max(h_img, w_img) > MAX_WORK_SIDE:
    s = MAX_WORK_SIDE / max(h_img, w_img)
    imagen_original = cv.resize(imagen_original, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
h_img, w_img = imagen_original.shape


# ---------- Find the 4 corner markers of the template ----------
def encontrar_marcadores(gray):
    """Return the centers of the 4 corner squares ordered TL, TR, BR, BL."""
    h, w = gray.shape
    suave = cv.GaussianBlur(gray, (5, 5), 0)
    # Local threshold: works with uneven light and shadows, unlike a fixed one
    block = int(w * 0.08) // 2 * 2 + 1
    binaria = cv.adaptiveThreshold(suave, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY_INV, block, 25)
    contornos, _ = cv.findContours(binaria, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    candidatos = []   # solid, square-ish dark blobs of the right size
    for c in contornos:
        (cx, cy), (rw, rh), _ = cv.minAreaRect(c)
        if rw == 0 or rh == 0:
            continue
        lado = np.sqrt(rw * rh)
        if not (MARKER_MIN * w <= lado <= MARKER_MAX * w):
            continue
        if not 0.6 <= rw / rh <= 1.67:   # perspective can skew the squares
            continue
        if cv.contourArea(c) / (rw * rh) < 0.75:     # must be solid (filled square)
            continue
        candidatos.append((cx, cy, lado))

    if len(candidatos) < 4:
        raise ValueError(f"Only {len(candidatos)} corner markers found (need 4). "
                         "Check that the 4 black corner squares are visible in the photo.")

    pts = np.array([[c[0], c[1]] for c in candidatos], dtype="float32")
    lados = np.array([c[2] for c in candidatos])
    suma = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]
    idx = [np.argmin(suma), np.argmin(diff), np.argmax(suma), np.argmax(diff)]  # TL TR BR BL
    if len(set(idx)) != 4:
        raise ValueError("Could not tell the 4 corner markers apart")

    esquinas = pts[idx]
    tam = lados[idx]
    if tam.max() / tam.min() > 1.7:
        raise ValueError("Corner markers have very different sizes, probably a wrong detection")

    tl, tr, br, bl = esquinas
    ancho = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    alto = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    esperado = (TEMPLATE_MARKERS[1][0] - TEMPLATE_MARKERS[0][0]) / (TEMPLATE_MARKERS[2][1] - TEMPLATE_MARKERS[1][1])
    if abs((ancho / alto) / esperado - 1) > ASPECT_TOLERANCE:
        raise ValueError("The detected markers do not have the template's proportions. "
                         "Check the photo orientation (PHOTO_ROTATION) and the markers.")
    return esquinas


marcadores = encontrar_marcadores(imagen_original)

# ---------- Step 1: warp so the answer box fills a fixed-size canvas ----------
# The markers lie OUTSIDE the box, so their destination points fall outside the canvas
# (negative coordinates are fine). Everything is scaled so the box is CANON_W x CANON_H.
escala_plantilla = CANON_W / BOX_W
destino = (TEMPLATE_MARKERS - np.array(TEMPLATE_BOX[:2], dtype="float32")) * escala_plantilla

matriz = cv.getPerspectiveTransform(marcadores, (destino * SUPERSAMPLE).astype("float32"))
alta = cv.warpPerspective(imagen_original, matriz, (CANON_W * SUPERSAMPLE, CANON_H * SUPERSAMPLE),
                          flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
flattened = cv.resize(alta, (CANON_W, CANON_H), interpolation=cv.INTER_AREA)
print(f"Photo {w_img}x{h_img} -> answer box {CANON_W}x{CANON_H}")

# ---------- Step 1b: flatten the illumination (shadows, gradients) ----------
# A big median filter ignores the marks (they cover well under half of the window) and
# returns the paper's brightness; dividing by it makes the paper ~white everywhere,
# whatever the lighting. (A median has no bias on noisy paper, unlike a max/closing filter.)
fondo = cv.medianBlur(flattened, 51)
fondo = cv.GaussianBlur(fondo, (0, 0), 10)
flattened = cv.divide(flattened, np.maximum(fondo, 1), scale=255)

blurred = cv.GaussianBlur(flattened, (3, 3), 0)

# ---------- Step 2: adaptive binarization (keeps empty bubble outlines) ----------
binary = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY_INV, 21, 6)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

# Remove the printed border of the box if it shows up: a long line / closed frame would
# swallow every bubble as an inner contour with RETR_EXTERNAL
n_comp, etiquetas, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
for i in range(1, n_comp):
    cw, ch = stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT]
    if cw > 0.5 * CANON_W or ch > 0.5 * CANON_H:
        binary[etiquetas == i] = 0

# ---------- Step 3: find clean bubble candidates (used only to calibrate the grid) ----------
MIN_SIZE, MAX_SIZE = 15, 40
H, W = flattened.shape
MARGIN_LEFT = int(W * 0.08)    # ignore the printed question numbers on the left
MARGIN_RIGHT = int(W * 0.015)  # tiny margin: the last column of bubbles sits near the right edge


def get_center(contour):
    x, y, w, h = cv.boundingRect(contour)
    return (x + w // 2, y + h // 2)


contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

bubbles = []
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    if not (MIN_SIZE <= w <= MAX_SIZE and MIN_SIZE <= h <= MAX_SIZE):
        continue
    if x < MARGIN_LEFT or x + w > W - MARGIN_RIGHT:
        continue
    perimeter = cv.arcLength(c, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * cv.contourArea(c) / (perimeter ** 2)
    if 0.8 <= w / float(h) <= 1.2 and circularity > 0.6:
        bubbles.append(c)

print(f"Clean bubbles found: {len(bubbles)}  (used to calibrate the grid)")
if len(bubbles) < 20:
    raise ValueError("Too few bubbles to calibrate the grid, check MIN_SIZE / MAX_SIZE")


# ---------- Step 4: build the full 10x10 grid from the detected bubbles ----------
def cluster_1d(values, gap, expected, min_count=3):
    """Group sorted 1D values into clusters and return the median of the `expected`
    most populated ones (extra noise clusters are discarded), sorted by position."""
    values = np.sort(np.asarray(values, dtype=float))
    groups = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] > gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    groups = [g for g in groups if len(g) >= min_count]  # drop noise clusters
    groups = sorted(groups, key=len, reverse=True)[:expected]
    return sorted(float(np.median(g)) for g in groups)


centers = np.array([get_center(c) for c in bubbles])
sizes = np.array([cv.boundingRect(c)[2:] for c in bubbles])
bubble_size = float(np.median(sizes))

cols = cluster_1d(centers[:, 0], bubble_size * 0.5, 10)  # 10 columns: 5 per block
rows = cluster_1d(centers[:, 1], bubble_size * 0.5, 10)  # 10 rows: questions per block

print(f"Columns found: {len(cols)} (expected 10) | Rows found: {len(rows)} (expected 10)")
print("Column x:", [round(v) for v in cols])
print("Row y:", [round(v) for v in rows])
if len(cols) != 10 or len(rows) != 10:
    raise ValueError("Grid calibration failed, check MIN_SIZE / MAX_SIZE / MARGIN_LEFT / MARGIN_RIGHT")

# ---------- Step 5: measure how filled each bubble is ----------
# Each bubble gets a "darkness" score: the mean darkness (0 = white paper, 1 = black) inside
# its inner disk. A full black mark scores ~0.85, a light pencil hatch ~0.10-0.15, an empty
# bubble ~0.00-0.03. Using the MEAN (not a pixel count) keeps faint, partial marks visible.
#
# --- Sensitivity settings (the 2 knobs to tune) ---
# Minimum score, above the sheet's own baseline, to be considered a mark at all.
#   -> lower it (e.g. 0.03) to accept even fainter / more partial marks,
#      raise it (e.g. 0.10) if empty bubbles trigger
MIN_FILL = 0.04
# A bubble also has to reach this fraction of the darkest bubble in its row, so a faint
# smudge or an eraser ghost next to a real mark is ignored (two real marks still give "*").
#   -> lower it (e.g. 0.3) to be more permissive with double marks
REL_FILL = 0.5
# Set to True to print the 5 scores of every question (to see how far apart marks and
# empty bubbles are in YOUR photos and pick MIN_FILL accordingly)
PRINT_SCORES = False

inner_radius = int(bubble_size * 0.5 * 0.6)  # inner disk, avoids the printed outline


def darkness(cx, cy):
    mask = np.zeros(blurred.shape, dtype=np.uint8)
    cv.circle(mask, (int(round(cx)), int(round(cy))), inner_radius, 255, -1)
    return (255 - cv.mean(blurred, mask=mask)[0]) / 255


# Scores of all 100 bubbles; most are empty, so their median is the sheet's baseline
scores = {}
for r, cy in enumerate(rows):
    for block in range(2):
        for k in range(5):
            scores[(r, block, k)] = darkness(cols[block * 5 + k], cy)
baseline = float(np.median(list(scores.values())))
print(f"Baseline darkness of the sheet: {baseline:.3f}")

answers = {}
debug = cv.cvtColor(flattened, cv.COLOR_GRAY2BGR)
radius = int(bubble_size / 2)

for r, cy in enumerate(rows):
    for block in range(2):  # block 0 = questions 1-10, block 1 = questions 11-20
        question = r + 1 + block * 10
        ratios = [scores[(r, block, k)] - baseline for k in range(5)]
        top = max(ratios)
        if PRINT_SCORES:
            print(f"Q{question:2d} scores:", " ".join(f"{OPTIONS[k]}={v:+.3f}" for k, v in enumerate(ratios)))

        marked = []
        for k, ratio in enumerate(ratios):
            is_marked = ratio >= MIN_FILL and ratio >= REL_FILL * top
            if is_marked:
                marked.append(OPTIONS[k])
            color = (0, 0, 255) if is_marked else (0, 200, 0)
            cv.circle(debug, (int(round(cols[block * 5 + k])), int(round(cy))), radius, color, 1)

        if len(marked) == 1:
            answers[question] = marked[0]
        elif len(marked) == 0:
            answers[question] = "-"      # blank
        else:
            answers[question] = "*"      # more than one option marked

# Orientation check: Q1 must appear top-left and Q20 bottom-right
for q, (bx, by) in {1: (cols[0], rows[0]), 10: (cols[0], rows[-1]),
                    11: (cols[5], rows[0]), 20: (cols[5], rows[-1])}.items():
    cv.putText(debug, f"Q{q}", (int(bx) - 5, int(by) - 12),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# ---------- Grade ----------
# Blank ("-") and double-marked ("*") answers never match the key, so they count as wrong
correctas = 0
for q in sorted(answers):
    if answers[q] == answer_key[q]:
        correctas += 1
        print(f"Question {q:2d}: {answers[q]}  OK")
    else:
        print(f"Question {q:2d}: {answers[q]}  X  (correct: {answer_key[q]})")

calificacion = correctas / TOTAL_QUESTIONS * 100
print(f"Respuestas correctas: {correctas}/{TOTAL_QUESTIONS}. Calificación {calificacion:g}/100")

# ---------- Debug windows ----------
vista = cv.cvtColor(imagen_original, cv.COLOR_GRAY2BGR)
for (mx, my) in marcadores:
    cv.circle(vista, (int(mx), int(my)), int(0.02 * w_img), (0, 0, 255), max(2, w_img // 300))
f = 900 / max(h_img, w_img)
cv.imshow("Markers found", cv.resize(vista, None, fx=f, fy=f, interpolation=cv.INTER_AREA))
cv.imshow("Binary", cv.resize(binary, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST))
cv.imshow("Grid (red = marked, green = empty)", cv.resize(debug, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST))

cv.waitKey(0)
cv.destroyAllWindows()