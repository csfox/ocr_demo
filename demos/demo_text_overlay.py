"""
Demo 3: Text Overlay from JSON with Letter Replacement

This demo demonstrates:
1. Reading PDF and OCR JSON via command-line arguments
2. Detecting background color for each element (using Mean strategy)
3. Replacing all letters with 'c' while preserving formulas and spaces
4. Rendering back to PDF with adaptive font sizing and inline LaTeX
"""

import fitz  # PyMuPDF
import json
import argparse
import numpy as np
import re
import math
import time
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
import io
from PIL import Image
from weasyprint import HTML
from json_translator import translate_element_text, should_skip_translation as should_skip_translation_check


# Language-specific line height map (from restorer.py)
LANG_LINEHEIGHT_MAP = {
    "zh-cn": 1.4,  # Chinese
    "ja": 1.1,     # Japanese
    "ko": 1.2,     # Korean
    "en": 1.2,     # English
    "ar": 1.0,     # Arabic
    "ru": 0.8,     # Russian
}


def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns.

    Args:
        text: Input text

    Returns:
        Language code ('zh-cn', 'en', etc.)
    """
    # Remove formulas for detection
    text_without_formulas = re.sub(r'\$(.+?)\$', '', text)

    # Count character types
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', text_without_formulas))
    latin_count = len(re.findall(r'[a-zA-Z]', text_without_formulas))

    # If CJK characters are significant, consider it Chinese
    if cjk_count > latin_count * 0.3:
        return "zh-cn"
    return "en"


def is_cjk_char(ch: str) -> bool:
    """
    Check if a character needs CJK font to render properly.
    Includes CJK ideographs, punctuation, symbols, and full-width characters.
    """
    if not ch:
        return False
    code = ord(ch)
    # CJK Unified Ideographs
    if 0x4E00 <= code <= 0x9FFF:
        return True
    # CJK Unified Ideographs Extension A
    if 0x3400 <= code <= 0x4DBF:
        return True
    # CJK Unified Ideographs Extension B-F (rarely used)
    if 0x20000 <= code <= 0x2EBEF:
        return True
    # CJK Compatibility Ideographs
    if 0xF900 <= code <= 0xFAFF:
        return True
    # CJK Symbols and Punctuation (。、「」『』【】〈〉《》等)
    if 0x3000 <= code <= 0x303F:
        return True
    # Hiragana (Japanese)
    if 0x3040 <= code <= 0x309F:
        return True
    # Katakana (Japanese)
    if 0x30A0 <= code <= 0x30FF:
        return True
    # Hangul (Korean)
    if 0xAC00 <= code <= 0xD7AF:
        return True
    # Fullwidth ASCII variants (！＂＃＄％＆等) and Halfwidth/Fullwidth Forms
    if 0xFF00 <= code <= 0xFFEF:
        return True
    # General Punctuation (includes ·, –, —, ', ', ", " etc.)
    if 0x2000 <= code <= 0x206F:
        return True
    # CJK Radicals Supplement
    if 0x2E80 <= code <= 0x2EFF:
        return True
    # Kangxi Radicals
    if 0x2F00 <= code <= 0x2FDF:
        return True
    # Enclosed CJK Letters and Months
    if 0x3200 <= code <= 0x32FF:
        return True
    # CJK Compatibility
    if 0x3300 <= code <= 0x33FF:
        return True
    # Vertical Forms
    if 0xFE10 <= code <= 0xFE1F:
        return True
    # CJK Compatibility Forms
    if 0xFE30 <= code <= 0xFE4F:
        return True
    # Small Form Variants
    if 0xFE50 <= code <= 0xFE6F:
        return True
    # Halfwidth and Fullwidth Forms (extra)
    if 0xFFE0 <= code <= 0xFFEF:
        return True
    # Mathematical symbols that may need special handling (×, ÷, etc.)
    if code == 0x00D7 or code == 0x00F7:  # × ÷
        return True
    return False


def detect_color_mean(pixels: np.ndarray) -> tuple:
    """
    Detect color using MEAN (average color).
    Fast and accurate for both solid colors and gradients.

    Args:
        pixels: Numpy array of RGB pixels, shape (n, 3)

    Returns:
        RGB tuple (r, g, b)
    """
    mean_color = np.mean(pixels, axis=0).astype(int)
    return tuple(mean_color)


def detect_background_color(page: fitz.Page, bbox: tuple, border_width: int = 1) -> tuple:
    """
    Detect background color from the border area OUTSIDE the bbox.

    Args:
        page: PyMuPDF page object
        bbox: Tuple of (x0, y0, x1, y1) coordinates
        border_width: Width of border area to sample (default: 1 pixels)

    Returns:
        RGB tuple (r, g, b), defaults to white (255, 255, 255) on error
    """
    try:
        x0, y0, x1, y1 = bbox

        # Expand bbox to include surrounding border
        expanded_rect = fitz.Rect(
            x0 - border_width,
            y0 - border_width,
            x1 + border_width,
            y1 + border_width
        )

        # Clip to page boundaries
        page_rect = page.rect
        expanded_rect = expanded_rect & page_rect

        # Check if rect is valid
        if expanded_rect.is_empty or expanded_rect.width < 1 or expanded_rect.height < 1:
            print(f"  [WARNING] Invalid bbox: {bbox}, using white background")
            return (255, 255, 255)

        # Get pixmap of expanded area
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat, clip=expanded_rect)

        # Check if pixmap has data
        if pix.width < 1 or pix.height < 1:
            print(f"  [WARNING] Empty pixmap for bbox: {bbox}, using white background")
            return (255, 255, 255)

        # Convert to numpy array
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)

        if pix.n == 3:  # RGB
            img_array = img_data.reshape(pix.height, pix.width, 3)
        elif pix.n == 4:  # RGBA
            img_array = img_data.reshape(pix.height, pix.width, 4)[:, :, :3]
        else:
            img_array = img_data.reshape(pix.height, pix.width, pix.n)

        # Calculate border dimensions in the pixmap
        # Scale border_width according to DPI ratio (150/72)
        scaled_border = int(border_width * 150 / 72)

        # Sample only LEFT and RIGHT border areas (not top/bottom)
        # Left border (full height)
        left_border = img_array[:, :scaled_border, :].reshape(-1, 3)
        # Right border (full height)
        right_border = img_array[:, -scaled_border:, :].reshape(-1, 3)

        # Combine left and right border pixels only
        border_pixels = np.vstack([left_border, right_border])

        # Check if we have pixels
        if len(border_pixels) == 0:
            print(f"  [WARNING] No pixels sampled for bbox: {bbox}, using white background")
            return (255, 255, 255)

        # Detect using Mean strategy (fast and accurate)
        bg_color = detect_color_mean(border_pixels)

        # Validate color (check for NaN)
        if np.any(np.isnan(bg_color)):
            print(f"  [WARNING] Invalid color detected for bbox: {bbox}, using white background")
            return (255, 255, 255)

        return bg_color

    except Exception as e:
        print(f"  [ERROR] Color detection failed for bbox {bbox}: {e}, using white background")
        return (255, 255, 255)


def replace_letters_preserve_formulas(text: str) -> str:
    """
    Replace all letters with 'c' while preserving LaTeX formulas and spaces.

    Args:
        text: Input text with possible LaTeX formulas

    Returns:
        Text with letters replaced

    Examples:
        "Hello world" -> "ccccc ccccc"
        "Einstein's $E=mc^2$ theory" -> "ccccccccc'c $E=mc^2$ cccccc"
        "Test 123" -> "cccc 123"
    """
    # Extract LaTeX formulas and replace with placeholders
    # Use placeholders without letters to avoid being replaced
    formula_pattern = r'\$(.+?)\$'
    formulas = []

    def extract_formula(match):
        formulas.append(match.group(0))  # Store with $ delimiters
        return f"###_{len(formulas)-1}_###"  # Use ### instead of letters

    # Replace formulas with placeholders
    text_without_formulas = re.sub(formula_pattern, extract_formula, text)

    # Replace all letters (a-z, A-Z) with 'c', but preserve spaces and other characters
    replaced_text = re.sub(r'[a-zA-Z]', 'c', text_without_formulas)

    # Restore formulas
    for i, formula in enumerate(formulas):
        replaced_text = replaced_text.replace(f"###_{i}_###", formula)

    return replaced_text


# Reuse functions from demo_pdf_render_enhanced.py

def parse_mixed_content(text: str) -> list:
    """
    Parse text and extract LaTeX formulas.

    Args:
        text: Input text with LaTeX formulas in $...$ format

    Returns:
        List of (type, content) tuples where type is 'text' or 'latex'
    """
    segments = []
    pattern = r'\$(.+?)\$'
    last_end = 0

    for match in re.finditer(pattern, text):
        # Add text before formula
        if match.start() > last_end:
            text_segment = text[last_end:match.start()]
            if text_segment:
                segments.append(('text', text_segment))

        # Add formula (without $ delimiters)
        segments.append(('latex', match.group(1)))
        last_end = match.end()

    # Add remaining text after last formula
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining:
            segments.append(('text', remaining))

    # If no formulas found, return the whole text as one segment
    if not segments:
        segments.append(('text', text))

    return segments


def render_latex_to_bytes_with_size(latex_formula: str, fontsize: int = 20, dpi: int = 300) -> tuple:
    """
    Render a LaTeX formula to PNG image bytes and return size info.
    Automatically crops transparent borders to minimize white space.

    Args:
        latex_formula: LaTeX formula string (without $ delimiters)
        fontsize: Font size for rendering
        dpi: DPI for image resolution

    Returns:
        Tuple of (image_bytes, width_pixels, height_pixels)
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)

    # Render the LaTeX formula
    text = fig.text(0, 0, f'${latex_formula}$', fontsize=fontsize)

    # Get the bounding box
    fig.canvas.draw()
    bbox = text.get_window_extent(fig.canvas.get_renderer())

    # Adjust figure size to fit the text
    fig.set_size_inches(bbox.width / dpi, bbox.height / dpi)

    # Re-position text
    text.set_position((0, 0))

    # Save to bytes buffer with minimal padding
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                transparent=True, pad_inches=0.02)
    plt.close(fig)

    buf.seek(0)

    # Load image and crop transparent borders
    pil_img = Image.open(buf)

    # Get the bounding box of non-transparent pixels
    if pil_img.mode == 'RGBA':
        alpha = pil_img.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            margin = 2
            bbox = (
                max(0, bbox[0] - margin),
                max(0, bbox[1] - margin),
                min(pil_img.width, bbox[2] + margin),
                min(pil_img.height, bbox[3] + margin)
            )
            pil_img = pil_img.crop(bbox)

    # Convert back to bytes
    output_buf = io.BytesIO()
    pil_img.save(output_buf, format='PNG')
    output_buf.seek(0)
    img_bytes = output_buf.getvalue()

    width, height = pil_img.size

    return img_bytes, width, height


def calculate_text_dimensions(text: str, fontname: str, fontsize: float, max_width: float, line_spacing: float = 1.2) -> tuple:
    """
    Calculate dimensions of text when rendered with wrapping.
    Simulates the actual line-breaking logic used in render_mixed_content.

    Height calculation matches render_mixed_content:
    - First line: fontsize (baseline at 0.85, descender at 0.15)
    - Subsequent lines: (n-1) * fontsize * line_spacing
    - Total: fontsize + (n-1) * fontsize * line_spacing

    Args:
        text: Text to measure
        fontname: Font name
        fontsize: Font size
        max_width: Maximum width before wrapping
        line_spacing: Line spacing multiplier (default 1.2)

    Returns:
        Tuple of (total_width, total_height, num_lines)
    """
    if not text:
        return 0, 0, 0

    # Check if text contains CJK characters
    # This determines which font will be used for rendering
    has_cjk = any(is_cjk_char(ch) for ch in text)

    # Character width function matching render_mixed_content logic
    def get_char_width(ch):
        if is_cjk_char(ch):
            # Full-width characters (CJK ideographs, punctuation, etc.)
            return fontsize * 1.0
        elif has_cjk:
            # When using CJK font for mixed text:
            # Latin letters and numbers are rendered at ~0.5 fontsize width
            if ch == ' ':
                return fontsize * 0.35  # Space is narrower
            elif ch in 'mwMW':
                return fontsize * 0.7  # Wide letters
            elif ch in 'il1!|':
                return fontsize * 0.3  # Narrow characters
            else:
                return fontsize * 0.5  # Average half-width
        else:
            # Pure Latin text using helv font
            return fitz.get_text_length(ch, fontname=fontname, fontsize=fontsize)

    # Simulate line-by-line rendering
    cursor_x = 0
    num_lines = 1
    max_line_width = 0

    for ch in text:
        # Handle newline
        if ch == '\n':
            max_line_width = max(max_line_width, cursor_x)
            cursor_x = 0
            num_lines += 1
            continue

        # Calculate character width
        char_width = get_char_width(ch)

        # Check if need to wrap
        if cursor_x + char_width > max_width:
            max_line_width = max(max_line_width, cursor_x)
            cursor_x = 0
            num_lines += 1
            # Skip leading space on new line
            if ch == ' ':
                continue

        cursor_x += char_width

    max_line_width = max(max_line_width, cursor_x)

    # Height calculation matching render_mixed_content:
    # First line takes fontsize, subsequent lines add fontsize * line_spacing each
    # total = fontsize + (num_lines - 1) * fontsize * line_spacing
    if num_lines == 1:
        total_height = fontsize
    else:
        total_height = fontsize + (num_lines - 1) * fontsize * line_spacing

    return max_line_width, total_height, num_lines


def calculate_adaptive_font_size(text: str, bbox: fitz.Rect,
                                   default_size: int = 12, min_size: int = 4,
                                   fontname: str = "helv") -> tuple:
    """
    Calculate adaptive font size and line spacing.

    Strategy:
    1. First try reducing line height (from 1.2 down to 0.9) to fit more lines
    2. If still doesn't fit, then reduce font size
    3. Goal: Ensure all text is displayed, never truncated

    Args:
        text: Text to fit in bbox
        bbox: Bounding rectangle
        default_size: Default starting size (will not grow beyond this)
        min_size: Minimum size threshold (default: 4pt)

    Returns:
        Tuple of (font_size, line_spacing) that fits the bbox
    """
    bbox_width = bbox.x1 - bbox.x0
    bbox_height = bbox.y1 - bbox.y0

    # Match the margin values in render_mixed_content
    margin_x = 1
    margin_y = 1

    # Calculate actual usable space (matching render_mixed_content)
    # Add extra safety margin (2pt) to ensure text definitely fits
    usable_width = bbox_width - margin_x * 2
    usable_height = bbox_height - margin_y * 2 - 2  # 2pt safety margin

    # Line spacing values to try (from loose to tight)
    line_spacings = [1.2, 1.1, 1.0, 0.95, 0.9]

    # Step 1: Try reducing line spacing first with default font size
    for line_spacing in line_spacings:
        _, height, _ = calculate_text_dimensions(text, fontname, default_size, usable_width, line_spacing)
        if height <= usable_height:
            return default_size, line_spacing

    # Step 2: Line spacing reduction wasn't enough, now reduce font size
    # Use the tightest line spacing (0.9) and binary search for font size
    min_line_spacing = 0.9

    low, high = min_size, default_size
    best_size = min_size

    while low <= high:
        mid = (low + high) // 2
        _, height, _ = calculate_text_dimensions(text, fontname, mid, usable_width, min_line_spacing)

        if height <= usable_height:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1

    # Step 3: With the found font size, try to use a looser line spacing if possible
    for line_spacing in line_spacings:
        _, height, _ = calculate_text_dimensions(text, fontname, best_size, usable_width, line_spacing)
        if height <= usable_height:
            return best_size, line_spacing

    return best_size, min_line_spacing


def render_mixed_content(page: fitz.Page, text: str, bbox: fitz.Rect,
                          fontsize: int, fontname: str = "helv", bg_color_rgb: tuple = (255, 255, 255),
                          line_spacing: float = None):
    """
    Render text with inline LaTeX formulas using character-by-character processing.

    Args:
        page: PDF page object
        text: Text with embedded LaTeX formulas
        bbox: Bounding rectangle
        fontsize: Font size to use
        fontname: Font name
        bg_color_rgb: Background color RGB tuple (r, g, b), range 0-255
        line_spacing: Line spacing multiplier (if None, auto-detect from language)
    """
    segments = parse_mixed_content(text)

    # Use provided line_spacing or auto-detect from language
    if line_spacing is None:
        lang = detect_language(text)
        line_spacing = LANG_LINEHEIGHT_MAP.get(lang, 1.2)

    # Auto-detect text color based on background brightness
    brightness = (bg_color_rgb[0] * 299 + bg_color_rgb[1] * 587 + bg_color_rgb[2] * 114) / 1000
    if brightness > 128:
        text_color = (0, 0, 0)  # Black text on light background
    else:
        text_color = (1, 1, 1)  # White text on dark background

    # Check if text contains CJK characters and load appropriate font
    has_cjk = any(is_cjk_char(ch) for ch in text)
    cjk_fontname = None
    cjk_fontfile = None

    if has_cjk:
        # Try to find a CJK font
        import os
        cjk_font_paths = [
            "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
            "C:/Windows/Fonts/simsun.ttc",    # SimSun
            "C:/Windows/Fonts/simhei.ttf",    # SimHei
        ]
        for font_path in cjk_font_paths:
            if os.path.exists(font_path):
                cjk_fontfile = font_path
                cjk_fontname = "cjk-font"
                break

    # Minimal margins
    margin_x = 1  # Horizontal margin
    margin_y = 1  # Vertical margin

    cursor_x = bbox.x0 + margin_x
    cursor_y = bbox.y0 + margin_y + fontsize * 0.85  # Adjust baseline position
    line_height = fontsize * line_spacing
    max_x = bbox.x1 - margin_x

    # Character buffer for batch insertion
    cstk = ""  # Character stack
    cstk_x = cursor_x  # Starting x position of buffer

    def flush_buffer():
        """Output the character buffer to PDF using appropriate font."""
        nonlocal cstk, cursor_x, cstk_x
        if not cstk:
            return

        # When we have CJK content, use CJK font for ALL characters
        # This ensures consistent appearance for mixed text
        if has_cjk and cjk_fontfile:
            page.insert_text(
                point=(cstk_x, cursor_y),
                text=cstk,
                fontsize=fontsize,
                fontname=cjk_fontname,
                fontfile=cjk_fontfile,
                color=text_color
            )
        else:
            page.insert_text(
                point=(cstk_x, cursor_y),
                text=cstk,
                fontsize=fontsize,
                fontname=fontname,
                color=text_color
            )

        cstk = ""

    # Character width calculation function
    # When using CJK font, all characters are rendered with that font
    # CJK chars = full-width, Latin/numbers = half-width (approximately)
    def get_char_width(ch):
        if is_cjk_char(ch):
            # Full-width characters (CJK ideographs, punctuation, etc.)
            return fontsize * 1.0
        elif has_cjk and cjk_fontfile:
            # When using CJK font for mixed text:
            # Latin letters and numbers are rendered at ~0.5 fontsize width
            if ch == ' ':
                return fontsize * 0.35  # Space is narrower
            elif ch in 'mwMW':
                return fontsize * 0.7  # Wide letters
            elif ch in 'il1!|':
                return fontsize * 0.3  # Narrow characters
            else:
                return fontsize * 0.5  # Average half-width
        else:
            # Pure Latin text using helv font
            return fitz.get_text_length(ch, fontname=fontname, fontsize=fontsize)

    for seg_type, content in segments:
        if seg_type == 'text':
            # Process character by character
            for i, ch in enumerate(content):
                # Handle newline character
                if ch == '\n':
                    flush_buffer()
                    cursor_x = bbox.x0 + margin_x
                    cursor_y += line_height
                    cstk_x = cursor_x
                    continue
                # Calculate character width
                char_width = get_char_width(ch)

                # Check if character fits on current line
                if cursor_x + char_width > max_x:
                    # Flush buffer before wrapping
                    flush_buffer()

                    # Move to next line
                    cursor_x = bbox.x0 + margin_x
                    cursor_y += line_height

                    # Skip leading space on new line
                    if ch == ' ':
                        continue
                if not cstk:
                    cstk_x = cursor_x
                # Add character to buffer
                cstk += ch
                cursor_x += char_width

            # Flush remaining characters
            flush_buffer()

        elif seg_type == 'latex':
            # Flush text buffer before inserting formula
            flush_buffer()

            try:
                # Render formula with larger fontsize for better quality
                formula_fontsize = fontsize * 0.9  # Increase formula size
                formula_bytes, img_width_px, img_height_px = render_latex_to_bytes_with_size(
                    content, fontsize=formula_fontsize, dpi=300
                )

                # Convert pixels to points: 72 points/inch, 300 dpi
                # Formula: points = pixels * (72 / dpi)
                px_to_pt = 72.0 / 300.0  # 0.24
                img_width_pt = img_width_px * px_to_pt
                img_height_pt = img_height_px * px_to_pt

                # Check if formula fits on current line
                if cursor_x + img_width_pt > max_x:
                    cursor_x = bbox.x0 + margin_x
                    cursor_y += line_height
                    cstk_x = cursor_x

                img_rect = fitz.Rect(
                    cursor_x,
                    cursor_y - img_height_pt * 0.90,
                    cursor_x + img_width_pt,
                    cursor_y + img_height_pt * 0.10
                )

                page.insert_image(img_rect, stream=formula_bytes)
                cursor_x += img_width_pt + 2  # Small spacing after formula
                cstk_x = cursor_x

            except Exception as e:
                print(f"Warning: Failed to render formula '{content}': {e}")
                fallback_text = f"${content}$"
                fallback_width = fitz.get_text_length(fallback_text, fontname=fontname, fontsize=fontsize)

                if cursor_x + fallback_width > max_x:
                    cursor_x = bbox.x0 + margin_x
                    cursor_y += line_height
                    cstk_x = cursor_x

                page.insert_text(
                    point=(cursor_x, cursor_y),
                    text=fallback_text,
                    fontsize=fontsize,
                    fontname=fontname,
                    color=(1, 0, 0)
                )
                cursor_x += fallback_width
                cstk_x = cursor_x

    # Final flush in case there's remaining buffer
    flush_buffer()


def _render_table_to_pdf(table_html, width, height, fontsize, line_height, padding, bg_color, text_color, border_color):
    """
    内部函数：使用WeasyPrint渲染HTML表格为PDF并返回文档和内容边界
    """
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: {width}pt {height * 3}pt;
                margin: 0;
            }}
            body {{
                margin: 0;
                padding: 0;
                font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
            }}
            table {{
                width: {width}pt;
                max-width: {width}pt;
                border-collapse: collapse;
                font-size: {fontsize}pt;
                table-layout: fixed;
                background-color: {bg_color};
                color: {text_color};
            }}
            th, td {{
                border: 0.5pt solid {border_color};
                padding: {padding}pt;
                text-align: left;
                vertical-align: middle;
                word-wrap: break-word;
                word-break: break-word;
                overflow-wrap: break-word;
                overflow: hidden;
                line-height: {line_height};
            }}
            th {{
                font-weight: bold;
            }}
            td:nth-child(n+3) {{
                text-align: right;
            }}
            strong {{
                font-weight: bold;
                color: {text_color};
            }}
        </style>
    </head>
    <body>{table_html}</body>
    </html>
    """

    pdf_bytes = BytesIO()
    HTML(string=full_html).write_pdf(pdf_bytes)
    pdf_bytes.seek(0)

    temp_doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
    temp_page = temp_doc[0]

    # 获取表格的实际边界
    content_rects = []
    drawings = temp_page.get_drawings()
    for d in drawings:
        content_rects.append(d["rect"])

    text_dict = temp_page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if "bbox" in block:
            content_rects.append(fitz.Rect(block["bbox"]))

    if content_rects:
        min_x = min(r.x0 for r in content_rects)
        min_y = min(r.y0 for r in content_rects)
        max_x = max(r.x1 for r in content_rects)
        max_y = max(r.y1 for r in content_rects)
        content_rect = fitz.Rect(min_x, min_y, max_x, max_y)
    else:
        content_rect = temp_page.rect

    return temp_doc, content_rect


def render_table(page, bbox, table_html, bg_color_rgb=(255, 255, 255)):
    """
    使用WeasyPrint渲染HTML表格到PDF页面的指定区域

    高效自适应算法：只渲染2次，根据首次渲染结果直接计算最优字号。

    Args:
        page: fitz.Page对象
        bbox: Tuple of (x0, y0, x1, y1) 目标区域坐标
        table_html: HTML表格字符串
        bg_color_rgb: 背景颜色RGB元组 (r, g, b)，范围0-255
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    start_time = time.time()

    # 根据背景色自动选择文本和边框颜色
    # 计算背景亮度
    brightness = (bg_color_rgb[0] * 299 + bg_color_rgb[1] * 587 + bg_color_rgb[2] * 114) / 1000
    if brightness > 128:
        # 浅色背景：黑色文本和边框
        text_color = "#000000"
        border_color = "#000000"
    else:
        # 深色背景：白色文本和边框
        text_color = "#ffffff"
        border_color = "#ffffff"

    bg_color = f"#{bg_color_rgb[0]:02x}{bg_color_rgb[1]:02x}{bg_color_rgb[2]:02x}"

    # 初始参数
    init_fontsize = 10.0
    line_height = 1.2
    padding = 2.0

    # 第1次渲染：获取初始大小
    temp_doc, content_rect = _render_table_to_pdf(
        table_html, width, height, init_fontsize, line_height, padding,
        bg_color, text_color, border_color
    )

    scale_x = width / content_rect.width if content_rect.width > 0 else 1
    scale_y = height / content_rect.height if content_rect.height > 0 else 1
    scale = min(scale_x, scale_y, 1.0)

    # 如果表格刚好能放入bbox（scale在0.90-1.0之间），直接使用
    if 0.90 <= scale <= 1.0:
        final_doc = temp_doc
        final_rect = content_rect
    else:
        temp_doc.close()

        # 计算最优字号，让表格刚好填满bbox
        # 字号与表格高度关系近似平方根（不是线性）
        optimal_fontsize = init_fontsize * math.sqrt(scale) * 0.95
        optimal_line_height = max(1.0, 1.0 + (line_height - 1.0) * math.sqrt(scale))
        optimal_padding = max(0.5, padding * math.sqrt(scale))

        # 第2次渲染：使用优化后的参数
        final_doc, final_rect = _render_table_to_pdf(
            table_html, width, height, optimal_fontsize, optimal_line_height, optimal_padding,
            bg_color, text_color, border_color
        )

        scale_x = width / final_rect.width if final_rect.width > 0 else 1
        scale_y = height / final_rect.height if final_rect.height > 0 else 1
        scale = min(scale_x, scale_y, 1.0)

    # 计算最终渲染尺寸
    actual_width = final_rect.width * scale
    actual_height = final_rect.height * scale
    target_rect = fitz.Rect(x0, y0, x0 + actual_width, y0 + actual_height)

    # 将渲染结果嵌入到目标页面
    page.show_pdf_page(target_rect, final_doc, 0, clip=final_rect)

    final_doc.close()

    elapsed_time = time.time() - start_time
    print(f"    [OK] 表格渲染完成 (尺寸: {actual_width:.1f}x{actual_height:.1f}pt, 耗时: {elapsed_time:.2f}s)")


def process_pdf_with_json(pdf_path: str, json_path: str, output_path: str = None, image_dpi: int = 200, draw_bbox: bool = False,
                          translate: bool = False, src_lang: str = "en", tgt_lang: str = "zh",
                          model_type: str = None, app_id: str = None):
    """
    Process PDF with OCR JSON: optionally translate and render back.

    Args:
        pdf_path: Input PDF file path
        json_path: OCR JSON file path
        output_path: Output PDF path (optional)
        image_dpi: DPI of the image used for OCR (default: 200)
        draw_bbox: Whether to draw bbox borders for debugging (default: False)
        translate: Whether to translate text elements (default: False)
        src_lang: Source language for translation (default: "en")
        tgt_lang: Target language for translation (default: "zh")
        model_type: Translation model type (default: None, uses deepseek_v3)
        app_id: Application ID for translation API (default: None)
    """
    print("\n" + "=" * 80)
    if translate:
        print(f"PDF文本覆盖处理 - 翻译模式 ({src_lang} -> {tgt_lang})")
    else:
        print("PDF文本覆盖处理")
    print("=" * 80)

    # Validate input files
    pdf_file = Path(pdf_path)
    json_file = Path(json_path)

    if not pdf_file.exists():
        print(f"[ERROR] PDF文件不存在: {pdf_path}")
        return

    if not json_file.exists():
        print(f"[ERROR] JSON文件不存在: {json_path}")
        return

    # Load JSON
    print(f"\n[1/5] 加载JSON文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)

    # Open PDF
    print(f"[2/5] 打开PDF文件: {pdf_path}")
    doc = fitz.open(pdf_path)

    # Determine output path
    if output_path is None:
        output_path = f"output/{pdf_file.stem}_overlay.pdf"

    # Support two JSON formats:
    # Format 1: {"pages": [{"page_index": 0, "elements": [...]}]}
    # Format 2: {"page_index": 0, "elements": [...]}
    if 'pages' in ocr_data:
        pages_data = ocr_data['pages']
    elif 'page_index' in ocr_data and 'elements' in ocr_data:
        # Single page format, wrap it in a list
        pages_data = [ocr_data]
    else:
        print(f"[ERROR] 无法识别的JSON格式！")
        print(f"  支持的格式1: {{'pages': [{{'page_index': 0, 'elements': [...]}}]}}")
        print(f"  支持的格式2: {{'page_index': 0, 'elements': [...]}}")
        return

    print(f"[3/5] 处理 {len(pages_data)} 个页面...")
    print(f"  坐标转换: 图片DPI {image_dpi} -> PDF DPI 72")

    # Calculate scaling factor
    pdf_dpi = 72.0
    scale_factor = pdf_dpi / image_dpi
    print(f"  缩放比例: {scale_factor:.4f}")

    # Process each page
    total_elements = 0
    for page_data in pages_data:
        page_index = page_data.get('page_index', 0)
        page = doc[page_index]

        elements = page_data.get('elements', [])
        print(f"\n  页面 {page_index}: {len(elements)} 个元素")

        for elem_idx, element in enumerate(elements):
            bbox = element.get('bbox')
            text = element.get('text', '')
            category = element.get('category', 'text').lower()  # Normalize to lowercase

            if not bbox:
                continue

            # For table elements, allow empty text (will contain HTML)
            if category != 'table' and not text:
                continue

            # Convert bbox to tuple
            if isinstance(bbox, list):
                bbox = tuple(bbox)

            # Convert image coordinates to PDF coordinates (simple scaling)
            x0, y0, x1, y1 = bbox
            bbox = (
                x0 * scale_factor,
                y0 * scale_factor,
                x1 * scale_factor,
                y1 * scale_factor
            )
            # Fix: Move entire text box up by 2 pixels to align with original PDF position
            bbox = (bbox[0], bbox[1] - 2, bbox[2], bbox[3] - 2)

            # Detect background color
            bg_color = detect_background_color(page, bbox)
            bg_color_norm = tuple(c / 255 for c in bg_color)

            # Draw background rectangle (slightly expanded to cover edge artifacts)
            rect = fitz.Rect(bbox)
            expand = 2  # 扩展像素数，防止漏出底部文本
            expanded_rect = fitz.Rect(
                rect.x0 - expand,
                rect.y0 - expand,
                rect.x1 + expand,
                rect.y1 + expand
            )
            page.draw_rect(expanded_rect, color=bg_color_norm, fill=bg_color_norm)

            # ========== Category-based Rendering ==========
            if category == 'table':
                # Table rendering path (WeasyPrint)
                try:
                    print(f"    [表格] 元素 {elem_idx + 1}")
                    render_table(page, bbox, text, bg_color_rgb=bg_color)
                except Exception as e:
                    print(f"    [错误] 表格渲染失败: {e}")
                    page.insert_textbox(rect, "错误: 表格渲染失败",
                                       fontsize=10, color=(1, 0, 0))
            elif category == 'formula':
                # Formula rendering path (display formula)
                try:
                    print(f"    [公式] 元素 {elem_idx + 1}")

                    # Extract formula content (remove $$ delimiters)
                    formula_text = text.strip()
                    if formula_text.startswith('$$') and formula_text.endswith('$$'):
                        formula_content = formula_text[2:-2].strip()
                    elif formula_text.startswith('$') and formula_text.endswith('$'):
                        formula_content = formula_text[1:-1].strip()
                    else:
                        formula_content = formula_text

                    # Calculate appropriate font size based on bbox height
                    bbox_height = rect.height
                    # Use larger font size for display formulas
                    formula_fontsize = min(bbox_height * 0.6, 24)  # Max 24pt
                    formula_fontsize = max(formula_fontsize, 14)   # Min 14pt

                    # Render formula to image
                    formula_bytes, img_width_px, img_height_px = render_latex_to_bytes_with_size(
                        formula_content, fontsize=formula_fontsize, dpi=300
                    )

                    # Convert pixels to points
                    px_to_pt = 72.0 / 300.0
                    img_width_pt = img_width_px * px_to_pt
                    img_height_pt = img_height_px * px_to_pt

                    # Calculate scaling to fit in bbox while maintaining aspect ratio
                    scale_x = rect.width / img_width_pt if img_width_pt > 0 else 1
                    scale_y = rect.height / img_height_pt if img_height_pt > 0 else 1
                    scale = min(scale_x, scale_y, 1.0)  # Don't upscale

                    final_width = img_width_pt * scale
                    final_height = img_height_pt * scale

                    # Center the formula in the bbox
                    center_x = rect.x0 + (rect.width - final_width) / 2
                    center_y = rect.y0 + (rect.height - final_height) / 2

                    # Create image rectangle
                    img_rect = fitz.Rect(
                        center_x,
                        center_y,
                        center_x + final_width,
                        center_y + final_height
                    )

                    # Insert formula image
                    page.insert_image(img_rect, stream=formula_bytes)
                    print(f"    [OK] 公式渲染完成 (尺寸: {final_width:.1f}x{final_height:.1f}pt)")

                except Exception as e:
                    print(f"    [错误] 公式渲染失败: {e}")
                    page.insert_textbox(rect, f"错误: 公式渲染失败\n{text[:50]}",
                                       fontsize=8, color=(1, 0, 0))
            else:
                # Text/Title/Caption rendering path
                modified_text = text

                # Translate if enabled
                if translate and not should_skip_translation_check(text, category):
                    translated_text = translate_element_text(
                        text, category,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        model_type=model_type,
                        app_id=app_id,
                    )
                    if translated_text != text:
                        modified_text = translated_text
                        print(f"    [翻译] 元素 {elem_idx + 1}: OK")
                        print(f"      原文: {text[:80]}{'...' if len(text) > 80 else ''}")
                        print(f"      译文: {translated_text[:80]}{'...' if len(translated_text) > 80 else ''}")

                # Calculate adaptive font size and line spacing based on the FINAL text (after translation)
                # Strategy: First reduce line spacing, then font size if needed
                text_only = re.sub(r'\$.*?\$', 'FORMULA', modified_text)
                fontsize, line_spacing = calculate_adaptive_font_size(text_only, rect, default_size=12, min_size=3)

                # Render mixed content with auto color adaptation and calculated line spacing
                render_mixed_content(page, modified_text, rect, fontsize, bg_color_rgb=bg_color, line_spacing=line_spacing)

            # Draw bbox border if requested
            if draw_bbox:
                # Draw border in red color for visibility
                page.draw_rect(rect, color=(1, 0, 0), width=1.5)

            total_elements += 1

            if (elem_idx + 1) % 10 == 0:
                print(f"    处理进度: {elem_idx + 1}/{len(elements)}")

    # Save output
    print(f"\n[4/5] 保存输出PDF: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()

    print(f"[5/5] 完成!")
    print("\n" + "=" * 80)
    print("处理统计:")
    print(f"  - 总页数: {len(pages_data)}")
    print(f"  - 总元素: {total_elements}")
    print(f"  - 输出文件: {output_path}")
    print("=" * 80 + "\n")


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='PDF文本覆盖工具 - 将JSON中的字母替换为c并渲染回PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo_text_overlay.py input.pdf ocr_result.json
  python demo_text_overlay.py input.pdf ocr_result.json -o custom_output.pdf

JSON格式:
  {
    "pages": [
      {
        "page_index": 0,
        "elements": [
          {
            "bbox": [x0, y0, x1, y1],
            "category": "text",
            "text": "Original text with $E=mc^2$ formula"
          }
        ]
      }
    ]
  }
        """
    )

    parser.add_argument('pdf_path', type=str, help='输入PDF文件路径')
    parser.add_argument('json_path', type=str, help='OCR JSON文件路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出PDF文件路径 (默认: output/{input}_overlay.pdf)')
    parser.add_argument('--dpi', type=int, default=200,
                       help='OCR图片的DPI (默认: 200)')
    parser.add_argument('--draw-bbox', action='store_true',
                       help='在PDF上绘制bbox边框（调试用）')

    # Translation options
    parser.add_argument('--translate', action='store_true',
                       help='启用翻译功能')
    parser.add_argument('--src-lang', type=str, default='en',
                       help='源语言 (默认: en)')
    parser.add_argument('--tgt-lang', type=str, default='zh',
                       help='目标语言 (默认: zh)')
    parser.add_argument('--model', type=str, default=None,
                       help='翻译模型 (deepseek_v3, volcengine)')
    parser.add_argument('--app-id', type=str, default=None,
                       help='翻译API的应用ID')

    args = parser.parse_args()

    process_pdf_with_json(
        args.pdf_path, args.json_path, args.output, args.dpi, args.draw_bbox,
        translate=args.translate,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        model_type=args.model,
        app_id=args.app_id
    )


if __name__ == "__main__":
    main()
