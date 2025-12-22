"""
Font Size Detection Demo using Tesseract OCR

Input:
    - A PDF file (e.g., ocr_zh_1.pdf)
    - A JSON file with OCR results containing bbox info (e.g., ocr_zh_1.json)

Output:
    - Print each Text/Title element's text and detected font size (x_fsize)
    - Print timing statistics
"""

import json
import re
import time
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def load_pdf_as_image(pdf_path: str, dpi: int = 200) -> Image.Image:
    """Load PDF and render as image at specified DPI."""
    doc = fitz.open(pdf_path)
    page = doc[0]  # First page

    # Calculate zoom factor for target DPI (default PDF is 72 DPI)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    # Render page to pixmap
    pix = page.get_pixmap(matrix=mat)

    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    return img


def load_json_elements(json_path: str) -> list[dict]:
    """Load JSON and filter Text/Title elements."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    elements = data.get("elements", [])
    # Filter only Text and Title categories
    filtered = [e for e in elements if e.get("category") in ("Text", "Title")]

    return filtered


def crop_element(img: Image.Image, bbox: list[float]) -> Image.Image:
    """Crop image region based on bbox [left, top, right, bottom]."""
    left, top, right, bottom = bbox
    return img.crop((int(left), int(top), int(right), int(bottom)))


def parse_xfsize_from_hocr(hocr: str) -> list[float]:
    """Parse x_fsize values from hOCR output."""
    # x_fsize pattern: x_fsize 12.5
    pattern = r'x_fsize\s+([\d.]+)'
    matches = re.findall(pattern, hocr)
    return [float(m) for m in matches]


def detect_font_size(img: Image.Image) -> tuple[float | None, float]:
    """
    Detect font size using Tesseract hOCR output.

    Returns:
        (average_font_size, elapsed_time_ms)
    """
    # Tesseract config for Chinese + hOCR output with font info
    config = r'--oem 3 --psm 6 -l chi_sim+eng -c tessedit_create_hocr=1 -c hocr_font_info=1'

    start = time.perf_counter()

    try:
        hocr = pytesseract.image_to_pdf_or_hocr(img, extension='hocr', config=config)
        hocr_str = hocr.decode('utf-8')

        font_sizes = parse_xfsize_from_hocr(hocr_str)

        elapsed = (time.perf_counter() - start) * 1000  # ms

        if font_sizes:
            avg_size = sum(font_sizes) / len(font_sizes)
            return avg_size, elapsed
        else:
            return None, elapsed

    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Error: {e}")
        return None, elapsed


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "ocr_zh_1.pdf"
    json_path = project_root / "ocr_zh_1.json"

    print(f"PDF: {pdf_path}")
    print(f"JSON: {json_path}")
    print("-" * 60)

    # Load PDF as 200 DPI image
    print("Loading PDF at 200 DPI...")
    load_start = time.perf_counter()
    page_img = load_pdf_as_image(str(pdf_path), dpi=200)
    load_time = (time.perf_counter() - load_start) * 1000
    print(f"PDF loaded: {page_img.width}x{page_img.height} ({load_time:.1f}ms)")
    print("-" * 60)

    # Load JSON elements
    elements = load_json_elements(str(json_path))
    print(f"Found {len(elements)} Text/Title elements")
    print("=" * 60)

    # Process each element
    total_time = 0
    results = []

    for i, elem in enumerate(elements):
        category = elem["category"]
        text = elem["text"]
        bbox = elem["bbox"]

        # Truncate text for display
        display_text = text[:40] + "..." if len(text) > 40 else text
        display_text = display_text.replace("\n", " ")

        print(f"[{i+1}/{len(elements)}] {category}: {display_text}")

        # Crop element region
        cropped = crop_element(page_img, bbox)

        # Detect font size
        font_size, elapsed = detect_font_size(cropped)
        total_time += elapsed

        if font_size:
            print(f"  -> x_fsize: {font_size:.1f} pt ({elapsed:.1f}ms)")
            results.append({"text": display_text, "font_size": font_size})
        else:
            print(f"  -> x_fsize: N/A ({elapsed:.1f}ms)")
            results.append({"text": display_text, "font_size": None})

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Total elements processed: {len(elements)}")
    print(f"Total OCR time: {total_time:.1f}ms")
    print(f"Average time per element: {total_time/len(elements):.1f}ms")

    # Font size statistics
    valid_sizes = [r["font_size"] for r in results if r["font_size"]]
    if valid_sizes:
        print(f"Font sizes detected: {len(valid_sizes)}/{len(elements)}")
        print(f"Min font size: {min(valid_sizes):.1f} pt")
        print(f"Max font size: {max(valid_sizes):.1f} pt")
        print(f"Avg font size: {sum(valid_sizes)/len(valid_sizes):.1f} pt")


if __name__ == "__main__":
    main()
