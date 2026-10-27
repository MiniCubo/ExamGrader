import cv2 as cv
import numpy as np
from pathlib import Path

# ---------- Load image ----------
BASE_DIR = Path(__file__).resolve().parent
image_path = BASE_DIR / "exam2.png"

imagen_original = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
if imagen_original is None:
    raise FileNotFoundError(f"Could not read image at: {image_path}")

# ---------- Find the exam sheet ----------
imagen_suavizada = cv.blur(imagen_original, (5, 5))
_, imagen_binarizada = cv.threshold(imagen_suavizada, 150, 255, cv.THRESH_BINARY_INV)

contornos, _ = cv.findContours(imagen_binarizada, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contorno = sorted(contornos, key=cv.contourArea, reverse=True)[0]
poi = cv.convexHull(contorno)


# ---------- Perspective correction (homography) ----------
def ordenar_puntos(puntos):
    rect = np.zeros((4, 2), dtype="float32")
    puntos = puntos.reshape(4, 2)

    suma = puntos.sum(axis=1)
    rect[0] = puntos[np.argmin(suma)]
    rect[2] = puntos[np.argmax(suma)]

    diff = np.diff(puntos, axis=1)
    rect[1] = puntos[np.argmin(diff)]
    rect[3] = puntos[np.argmax(diff)]

    return rect


perimetro = cv.arcLength(poi, True)
epsilon = 0.02 * perimetro
contorno_examen = cv.approxPolyDP(poi, epsilon, True)

# If it does not give 4 points, keep increasing epsilon
intentos = 0
while len(contorno_examen) != 4 and intentos < 10:
    epsilon *= 1.1
    contorno_examen = cv.approxPolyDP(poi, epsilon, True)
    intentos += 1

if len(contorno_examen) != 4:
    raise ValueError("Could not approximate the contour to 4 points")

rectangulo = ordenar_puntos(contorno_examen)
(tl, tr, br, bl) = rectangulo

anchuraA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
anchuraB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
max_ancho = max(int(anchuraA), int(anchuraB))

alturaA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
alturaB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
max_alto = max(int(alturaA), int(alturaB))

destino = np.array([
    [0, 0],
    [max_ancho - 1, 0],
    [max_ancho - 1, max_alto - 1],
    [0, max_alto - 1]
], dtype="float32")

matriz = cv.getPerspectiveTransform(rectangulo, destino)

# ---------- Step 1: flatten the ORIGINAL image and rotate it upright ----------
flattened = cv.warpPerspective(imagen_original, matriz, (max_ancho, max_alto))
flattened = cv.rotate(flattened, cv.ROTATE_90_COUNTERCLOCKWISE)
blurred = cv.GaussianBlur(flattened, (3, 3), 0)

# ---------- Step 2: adaptive binarization (keeps empty bubble outlines) ----------
binary = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY_INV, 21, 6)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

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
def cluster_1d(values, gap, min_count=3):
    """Group sorted 1D values into clusters; return the median of each cluster."""
    values = np.sort(np.asarray(values, dtype=float))
    groups = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] > gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    groups = [g for g in groups if len(g) >= min_count]  # drop noise clusters
    return [float(np.median(g)) for g in groups]


centers = np.array([get_center(c) for c in bubbles])
sizes = np.array([cv.boundingRect(c)[2:] for c in bubbles])
bubble_size = float(np.median(sizes))

cols = cluster_1d(centers[:, 0], bubble_size * 0.5)  # 10 columns: 5 per block
rows = cluster_1d(centers[:, 1], bubble_size * 0.5)  # 10 rows: questions per block

print(f"Columns found: {len(cols)} (expected 10) | Rows found: {len(rows)} (expected 10)")
print("Column x:", [round(v) for v in cols])
print("Row y:", [round(v) for v in rows])
if len(cols) != 10 or len(rows) != 10:
    raise ValueError("Grid calibration failed, check MIN_SIZE / MAX_SIZE / MARGIN_LEFT / MARGIN_RIGHT")

# ---------- Step 5: measure how filled each bubble is ----------
# Global Otsu keeps only the dark marks and drops the faint gray outlines
_, fill_bin = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

inner_radius = int(bubble_size * 0.5 * 0.6)  # inner disk, avoids the printed outline
FILL_THRESHOLD = 0.25                        # fraction of dark pixels to count as marked
OPTIONS = "ABCDE"


def fill_ratio(cx, cy):
    mask = np.zeros(fill_bin.shape, dtype=np.uint8)
    cv.circle(mask, (int(round(cx)), int(round(cy))), inner_radius, 255, -1)
    dark = cv.countNonZero(cv.bitwise_and(fill_bin, mask))
    return dark / cv.countNonZero(mask)


answers = {}
debug = cv.cvtColor(flattened, cv.COLOR_GRAY2BGR)
radius = int(bubble_size / 2)

for r, cy in enumerate(rows):
    for block in range(2):  # block 0 = questions 1-10, block 1 = questions 11-20
        question = r + 1 + block * 10
        marked = []
        for k in range(5):
            cx = cols[block * 5 + k]
            ratio = fill_ratio(cx, cy)
            is_marked = ratio >= FILL_THRESHOLD
            if is_marked:
                marked.append(OPTIONS[k])
            color = (0, 0, 255) if is_marked else (0, 200, 0)
            cv.circle(debug, (int(round(cx)), int(round(cy))), radius, color, 1)

        if len(marked) == 1:
            answers[question] = marked[0]
        elif len(marked) == 0:
            answers[question] = "-"      # blank
        else:
            answers[question] = "*"      # more than one option marked

for q in sorted(answers):
    print(f"Question {q:2d}: {answers[q]}")

cv.imshow("Binary", binary)
cv.imshow("Grid (red = marked, green = empty)", debug)

cv.waitKey(0)
cv.destroyAllWindows()