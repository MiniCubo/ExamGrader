import base64
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import cv2 as cv
import numpy as np

# ---------- Settings ----------
BASE_DIR = Path(__file__).resolve().parent
EXAMS_DIR = BASE_DIR                  # folder with the exam photos (change it to e.g. BASE_DIR / "examenes")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
IGNORE_FILES = {"plantilla.png"}      # images in that folder that are NOT exams (case-insensitive)
ANSWER_KEY_NAME = "respuestas.txt"    # one correct answer (A-E) per line, line 1 = question 1
REPORT_NAME = "reporte.txt"           # the report is saved here (and also printed)
VERBOSE = False                       # True: print diagnostics and the OK/X detail of every question

PHOTO_ROTATION = None   # None | cv.ROTATE_90_CLOCKWISE | cv.ROTATE_90_COUNTERCLOCKWISE | cv.ROTATE_180

TOTAL_QUESTIONS = 20
OPTIONS = "ABCDE"

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

# Bubble detection (in canonical pixels)
MIN_SIZE, MAX_SIZE = 15, 40

# --- Mark sensitivity (the 2 knobs to tune) ---
# Each bubble gets a "darkness" score: the mean darkness (0 = white paper, 1 = black) inside
# its inner disk. A full black mark scores ~0.85, a light pencil hatch ~0.10-0.15, an empty
# bubble ~0.00-0.03.
# Minimum score, above the sheet's own baseline, to be considered a mark at all.
#   -> lower it (e.g. 0.03) to accept fainter / more partial marks, raise it (e.g. 0.10)
#      if empty bubbles trigger
MIN_FILL = 0.04
# A bubble also has to reach this fraction of the darkest bubble in its row, so a faint
# smudge or an eraser ghost next to a real mark is ignored (two real marks still give "*").
REL_FILL = 0.5


def log(*args):
    if VERBOSE:
        print(*args)


# =====================================================================
#  Answer key
# =====================================================================
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


# =====================================================================
#  Image analysis
# =====================================================================
def cargar_imagen(path):
    gris = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if gris is None:
        raise ValueError("El archivo no se pudo leer como imagen")

    if PHOTO_ROTATION is not None:
        gris = cv.rotate(gris, PHOTO_ROTATION)

    h, w = gris.shape
    if max(h, w) > MAX_WORK_SIDE:
        s = MAX_WORK_SIDE / max(h, w)
        gris = cv.resize(gris, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    return gris


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
        raise ValueError(f"Solo se encontraron {len(candidatos)} marcadores de esquina (se necesitan 4). "
                         "Verifica que los 4 cuadros negros de las esquinas se vean completos en la foto.")

    pts = np.array([[c[0], c[1]] for c in candidatos], dtype="float32")
    lados = np.array([c[2] for c in candidatos])
    suma = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]
    idx = [np.argmin(suma), np.argmin(diff), np.argmax(suma), np.argmax(diff)]  # TL TR BR BL
    if len(set(idx)) != 4:
        raise ValueError("No se pudieron distinguir los 4 marcadores de esquina")

    esquinas = pts[idx]
    tam = lados[idx]
    if tam.max() / tam.min() > 1.7:
        raise ValueError("Los marcadores de esquina tienen tamaños muy distintos, probablemente se detectó algo que no es un marcador")

    tl, tr, br, bl = esquinas
    ancho = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    alto = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    esperado = (TEMPLATE_MARKERS[1][0] - TEMPLATE_MARKERS[0][0]) / (TEMPLATE_MARKERS[2][1] - TEMPLATE_MARKERS[1][1])
    if abs((ancho / alto) / esperado - 1) > ASPECT_TOLERANCE:
        raise ValueError("Los marcadores detectados no tienen las proporciones de la plantilla. "
                         "Revisa la orientación de la foto (PHOTO_ROTATION) y que se vean los 4 marcadores.")
    return esquinas


def get_center(contour):
    x, y, w, h = cv.boundingRect(contour)
    return (x + w // 2, y + h // 2)


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


def analizar_hoja(gris):
    """Read the 20 answers of one sheet.

    Returns (answers, marcadores, binary, debug):
      answers    {question: "A".."E" | "-" (blank) | "*" (more than one)}
      marcadores the 4 corner markers found in the photo
      binary     binarized answer box (for the viewer)
      debug      the answer box with every bubble circled (red = marked, green = empty)
    """
    marcadores = encontrar_marcadores(gris)

    # ---------- Warp so the answer box fills a fixed-size canvas ----------
    # The markers lie OUTSIDE the box, so their destination points fall outside the canvas
    # (negative coordinates are fine). Everything is scaled so the box is CANON_W x CANON_H.
    escala_plantilla = CANON_W / BOX_W
    destino = (TEMPLATE_MARKERS - np.array(TEMPLATE_BOX[:2], dtype="float32")) * escala_plantilla

    matriz = cv.getPerspectiveTransform(marcadores, (destino * SUPERSAMPLE).astype("float32"))
    alta = cv.warpPerspective(gris, matriz, (CANON_W * SUPERSAMPLE, CANON_H * SUPERSAMPLE),
                              flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    flattened = cv.resize(alta, (CANON_W, CANON_H), interpolation=cv.INTER_AREA)

    # ---------- Flatten the illumination (shadows, gradients) ----------
    # A big median filter ignores the marks (they cover well under half of the window) and
    # returns the paper's brightness; dividing by it makes the paper ~white everywhere.
    fondo = cv.medianBlur(flattened, 51)
    fondo = cv.GaussianBlur(fondo, (0, 0), 10)
    flattened = cv.divide(flattened, np.maximum(fondo, 1), scale=255)

    blurred = cv.GaussianBlur(flattened, (3, 3), 0)

    # ---------- Adaptive binarization (keeps empty bubble outlines) ----------
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

    # ---------- Clean bubble candidates (used only to calibrate the grid) ----------
    H, W = flattened.shape
    margin_left = int(W * 0.08)    # ignore the printed question numbers on the left
    margin_right = int(W * 0.015)  # tiny margin: the last column of bubbles sits near the right edge

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if not (MIN_SIZE <= w <= MAX_SIZE and MIN_SIZE <= h <= MAX_SIZE):
            continue
        if x < margin_left or x + w > W - margin_right:
            continue
        perimeter = cv.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * cv.contourArea(c) / (perimeter ** 2)
        if 0.8 <= w / float(h) <= 1.2 and circularity > 0.6:
            bubbles.append(c)

    log(f"Clean bubbles found: {len(bubbles)}  (used to calibrate the grid)")
    if len(bubbles) < 20:
        raise ValueError("Se encontraron muy pocas burbujas para calibrar la cuadrícula (¿es la plantilla correcta?)")

    # ---------- Build the full 10x10 grid from the detected bubbles ----------
    centers = np.array([get_center(c) for c in bubbles])
    sizes = np.array([cv.boundingRect(c)[2:] for c in bubbles])
    bubble_size = float(np.median(sizes))

    cols = cluster_1d(centers[:, 0], bubble_size * 0.5, 10)  # 10 columns: 5 per block
    rows = cluster_1d(centers[:, 1], bubble_size * 0.5, 10)  # 10 rows: questions per block

    log(f"Columns found: {len(cols)} (expected 10) | Rows found: {len(rows)} (expected 10)")
    if len(cols) != 10 or len(rows) != 10:
        raise ValueError(f"No se pudo calibrar la cuadrícula ({len(cols)} columnas / {len(rows)} filas "
                         "encontradas, se esperaban 10 / 10)")

    # ---------- Measure how filled each bubble is ----------
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
    log(f"Baseline darkness of the sheet: {baseline:.3f}")

    answers = {}
    debug = cv.cvtColor(flattened, cv.COLOR_GRAY2BGR)
    radius = int(bubble_size / 2)

    for r, cy in enumerate(rows):
        for block in range(2):  # block 0 = questions 1-10, block 1 = questions 11-20
            question = r + 1 + block * 10
            ratios = [scores[(r, block, k)] - baseline for k in range(5)]
            top = max(ratios)
            log(f"Q{question:2d} scores:", " ".join(f"{OPTIONS[k]}={v:+.3f}" for k, v in enumerate(ratios)))

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

    return answers, marcadores, binary, debug


def png(img):
    """Compress an image in memory (so dozens of exams do not eat the RAM)."""
    ok, buf = cv.imencode(".png", img)
    return buf


def calificar_imagen(path, clave):
    """Analyze one exam photo and grade it. Never raises: errors are returned in the result."""
    resultado = {"archivo": path, "nombre": path.stem, "error": None}
    try:
        gris = cargar_imagen(path)
        answers, marcadores, binary, debug = analizar_hoja(gris)

        # Images for the viewer: markers on the photo, binarized box, circled grid
        h, w = gris.shape
        vista = cv.cvtColor(gris, cv.COLOR_GRAY2BGR)
        for (mx, my) in marcadores:
            cv.circle(vista, (int(mx), int(my)), int(0.02 * w), (0, 0, 255), max(2, w // 300))
        f = 900 / max(h, w)
        vista = cv.resize(vista, None, fx=f, fy=f, interpolation=cv.INTER_AREA)
        binary = cv.cvtColor(cv.resize(binary, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST),
                             cv.COLOR_GRAY2BGR)
        debug = cv.resize(debug, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)

        # Blank ("-") and double-marked ("*") answers never match the key: they count as wrong
        correctas = sum(1 for q in answers if answers[q] == clave[q])
        resultado.update(
            respuestas=answers,
            correctas=correctas,
            calificacion=correctas / TOTAL_QUESTIONS * 100,
            imagenes=[("Markers found", png(vista)),
                      ("Binary", png(binary)),
                      ("Grid (red = marked, green = empty)", png(debug))],
        )

        if VERBOSE:
            for q in sorted(answers):
                if answers[q] == clave[q]:
                    print(f"  Question {q:2d}: {answers[q]}  OK")
                else:
                    print(f"  Question {q:2d}: {answers[q]}  X  (correct: {clave[q]})")
    except Exception as e:      # one bad photo must not stop the whole batch
        resultado["error"] = str(e)
    return resultado


# =====================================================================
#  Batch + report
# =====================================================================
def listar_examenes(carpeta):
    archivos = [p for p in carpeta.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
                and p.name.lower() not in IGNORE_FILES]
    return sorted(archivos, key=lambda p: p.name.lower())


def texto_resultado(r):
    if r["error"]:
        return f"No se pudo procesar: {r['error']}"
    return (f"Respuestas correctas: {r['correctas']}/{TOTAL_QUESTIONS}. "
            f"Calificación {r['calificacion']:g}/100")


def construir_reporte(resultados):
    """Student name first, then the result line, one block per student."""
    return "\n\n".join(f"{r['nombre']}\n{texto_resultado(r)}" for r in resultados) + "\n"


# =====================================================================
#  Viewer: one window with a scrollable list of students
# =====================================================================
class Visor:
    def __init__(self, resultados):
        self.resultados = resultados
        self.ventanas = []

        self.root = tk.Tk()
        self.root.title("Exámenes calificados")
        self.root.geometry("320x620+10+10")
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

        tk.Label(self.root, text="Haz clic en un alumno para ver el análisis",
                 pady=8).pack(fill="x")
        marco = tk.Frame(self.root)
        marco.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        barra = tk.Scrollbar(marco)
        barra.pack(side="right", fill="y")
        self.lista = tk.Listbox(marco, yscrollcommand=barra.set, font=("Segoe UI", 11),
                                activestyle="none", exportselection=False)
        self.lista.pack(side="left", fill="both", expand=True)
        barra.config(command=self.lista.yview)

        for r in self.resultados:
            self.lista.insert("end", r["nombre"] + ("   (error)" if r["error"] else ""))

        # A click opens the 3 windows (also when clicking the same student again)
        self.lista.bind("<ButtonRelease-1>", self.al_hacer_clic)
        self.lista.bind("<Return>", lambda e: self.abrir_seleccion())

    def al_hacer_clic(self, evento):
        i = self.lista.nearest(evento.y)
        caja = self.lista.bbox(i)
        if caja is None or not (caja[1] <= evento.y <= caja[1] + caja[3]):
            return                      # click on the empty area below the last name
        self.lista.selection_clear(0, "end")
        self.lista.selection_set(i)
        self.abrir(i)

    def abrir_seleccion(self):
        seleccion = self.lista.curselection()
        if seleccion:
            self.abrir(seleccion[0])

    def cerrar_ventanas(self):
        for v in self.ventanas:
            try:
                v.destroy()
            except tk.TclError:
                pass
        self.ventanas = []

    def abrir(self, i):
        r = self.resultados[i]
        self.cerrar_ventanas()

        if r["error"]:
            messagebox.showerror(r["nombre"], f"No se pudo procesar:\n\n{r['error']}",
                                 parent=self.root)
            return

        # Fit the 3 images side by side, to the right of the list, inside the screen
        self.root.update_idletasks()
        x0 = self.root.winfo_x() + self.root.winfo_width() + 20
        ancho_pantalla = self.root.winfo_screenwidth() - x0 - 10
        alto = min(640, self.root.winfo_screenheight() - 160)

        imgs = [(titulo, cv.imdecode(buf, cv.IMREAD_COLOR)) for titulo, buf in r["imagenes"]]
        total = sum(im.shape[1] / im.shape[0] for _, im in imgs) * alto + 16 * len(imgs)
        if total > ancho_pantalla:
            alto = int(alto * ancho_pantalla / total)

        x = x0
        for titulo, im in imgs:
            ancho = int(im.shape[1] * alto / im.shape[0])
            im = cv.resize(im, (ancho, alto), interpolation=cv.INTER_AREA)
            foto = tk.PhotoImage(data=base64.b64encode(png(im).tobytes()))

            ventana = tk.Toplevel(self.root)
            ventana.title(f"{r['nombre']} - {titulo}")
            ventana.resizable(False, False)
            ventana.geometry(f"+{x}+20")
            etiqueta = tk.Label(ventana, image=foto, borderwidth=0)
            etiqueta.image = foto       # keep a reference or Tk drops the image
            etiqueta.pack()

            self.ventanas.append(ventana)
            x += ancho + 16

    def cerrar(self):
        self.cerrar_ventanas()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()


# =====================================================================
#  Main
# =====================================================================
def main():
    clave = cargar_clave(BASE_DIR / ANSWER_KEY_NAME)    # fail early if the key is wrong

    archivos = listar_examenes(EXAMS_DIR)
    if not archivos:
        print(f"No .png / .jpg images found in {EXAMS_DIR}")
        return

    # Two files with the same name but different extension: show the extension too
    nombres = [p.stem.lower() for p in archivos]

    resultados = []
    for n, ruta in enumerate(archivos, start=1):
        print(f"[{n}/{len(archivos)}] {ruta.name}")
        r = calificar_imagen(ruta, clave)
        if nombres.count(ruta.stem.lower()) > 1:
            r["nombre"] = ruta.name
        resultados.append(r)

    reporte = construir_reporte(resultados)
    print("\n" + "=" * 50 + "\n" + reporte)

    ruta_reporte = BASE_DIR / REPORT_NAME
    ruta_reporte.write_text(reporte, encoding="utf-8-sig")   # BOM so Notepad shows the accents
    print(f"Reporte guardado en: {ruta_reporte}")

    Visor(resultados).mainloop()


if __name__ == "__main__":
    main()