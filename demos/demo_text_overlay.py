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
from pathlib import Path
import matplotlib.pyplot as plt
import io
from PIL import Image
from html.parser import HTMLParser
from pdf_utils import load_unicode_font, get_safe_fontname


# Language-specific line height map (from restorer.py)
LANG_LINEHEIGHT_MAP = {
    "zh-cn": 1.4,  # Chinese
    "ja": 1.1,     # Japanese
    "ko": 1.2,     # Korean
    "en": 1.2,     # English
    "ar": 1.0,     # Arabic
    "ru": 0.8,     # Russian
}


class TableHTMLParser(HTMLParser):
    """Parse HTML table structure and extract data"""

    def __init__(self):
        super().__init__()
        self.headers = []
        self.rows = []
        self.current_row = []
        self.current_cell = {'text': '', 'bold': False, 'sup': False}
        self.in_header = False
        self.in_tbody = False
        self.in_cell = False
        self.in_strong = False
        self.in_sup = False

    def handle_starttag(self, tag, attrs):
        if tag == 'thead':
            self.in_header = True
        elif tag == 'tbody':
            self.in_tbody = True
        elif tag == 'tr':
            self.current_row = []
        elif tag in ['th', 'td']:
            self.in_cell = True
            self.current_cell = {'text': '', 'bold': False, 'sup': False}
        elif tag == 'strong':
            self.in_strong = True
            self.current_cell['bold'] = True
        elif tag == 'sup':
            self.in_sup = True
            self.current_cell['sup'] = True

    def handle_data(self, data):
        if self.in_cell and data.strip():
            self.current_cell['text'] += data.strip()

    def handle_endtag(self, tag):
        if tag in ['th', 'td']:
            self.in_cell = False
            if self.in_header:
                self.headers.append(self.current_cell['text'])
            else:
                self.current_row.append(self.current_cell)
        elif tag == 'tr' and self.current_row:
            self.rows.append(self.current_row)
        elif tag == 'thead':
            self.in_header = False
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'strong':
            self.in_strong = False
        elif tag == 'sup':
            self.in_sup = False

    def get_table_data(self):
        """Return parsed table data"""
        return {
            'headers': self.headers,
            'rows': self.rows
        }


def is_numeric_text(text: str) -> bool:
    """
    Check if text is primarily numeric (for right alignment)

    Args:
        text: Text to check

    Returns:
        bool: True if text matches number patterns
    """
    if not text:
        return False
    # Check if text matches number patterns: digits, dots, dashes, daggers
    return bool(re.match(r'^[\d.—†\-]+$', text))


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
    """Check if a character is CJK (Chinese/Japanese/Korean)."""
    if not ch:
        return False
    code = ord(ch)
    # CJK Unified Ideographs: U+4E00-U+9FFF
    return 0x4E00 <= code <= 0x9FFF


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

        # Sample only the border area (excluding the inner bbox)
        # Top border
        top_border = img_array[:scaled_border, :, :].reshape(-1, 3)
        # Bottom border
        bottom_border = img_array[-scaled_border:, :, :].reshape(-1, 3)
        # Left border (excluding corners already in top/bottom)
        left_border = img_array[scaled_border:-scaled_border, :scaled_border, :].reshape(-1, 3)
        # Right border (excluding corners already in top/bottom)
        right_border = img_array[scaled_border:-scaled_border, -scaled_border:, :].reshape(-1, 3)

        # Combine all border pixels
        border_pixels = np.vstack([top_border, bottom_border, left_border, right_border])

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


def calculate_text_dimensions(text: str, fontname: str, fontsize: float, max_width: float) -> tuple:
    """
    Calculate dimensions of text when rendered with wrapping.

    Args:
        text: Text to measure
        fontname: Font name
        fontsize: Font size
        max_width: Maximum width before wrapping

    Returns:
        Tuple of (total_width, total_height, num_lines)
    """
    if not text:
        return 0, 0, 0

    text_width = fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)

    if text_width <= max_width:
        num_lines = 1
        actual_width = text_width
    else:
        num_lines = int(text_width / max_width) + 1
        actual_width = max_width

    line_spacing = 1.2
    total_height = num_lines * fontsize * line_spacing

    return actual_width, total_height, num_lines


def calculate_adaptive_font_size(text: str, bbox: fitz.Rect,
                                   default_size: int = 12, min_size: int = 4,
                                   fontname: str = "helv") -> int:
    """
    Calculate adaptive font size (shrink-only).

    Args:
        text: Text to fit in bbox
        bbox: Bounding rectangle
        default_size: Default starting size (will not grow beyond this)
        min_size: Minimum size threshold (default: 4pt)

    Returns:
        Font size that fits the bbox
    """
    bbox_width = bbox.x1 - bbox.x0
    bbox_height = bbox.y1 - bbox.y0

    usable_width = bbox_width * 0.95
    usable_height = bbox_height * 0.95

    _, height, _ = calculate_text_dimensions(text, fontname, default_size, usable_width)

    if height <= usable_height:
        return default_size

    low, high = min_size, default_size
    best_size = min_size

    while low <= high:
        mid = (low + high) // 2
        _, height, _ = calculate_text_dimensions(text, fontname, mid, usable_width)

        if height <= usable_height:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_size


def render_mixed_content(page: fitz.Page, text: str, bbox: fitz.Rect,
                          fontsize: int, fontname: str = "helv"):
    """
    Render text with inline LaTeX formulas using character-by-character processing.

    Args:
        page: PDF page object
        text: Text with embedded LaTeX formulas
        bbox: Bounding rectangle
        fontsize: Font size to use
        fontname: Font name
    """
    segments = parse_mixed_content(text)

    # Detect language and get appropriate line spacing
    lang = detect_language(text)
    line_spacing = LANG_LINEHEIGHT_MAP.get(lang, 1.2)

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
        """Output the character buffer to PDF."""
        nonlocal cstk, cursor_x, cstk_x
        if cstk:
            page.insert_text(
                point=(cstk_x, cursor_y),
                text=cstk,
                fontsize=fontsize,
                fontname=fontname,
                color=(0, 0, 0)
            )
            cstk = ""
            # Note: cstk_x is NOT updated here - it will be set when starting a new buffer
            # (at line 413 initially, and at wrap points like line 454)

    # Pre-calculate space width for efficiency
    space_width = fitz.get_text_length(' ', fontname=fontname, fontsize=fontsize)

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
                if ch == ' ':
                    char_width = space_width
                elif is_cjk_char(ch):
                    # CJK characters are typically 1em wide
                    char_width = fontsize * 1.0
                else:
                    # Latin and other characters
                    char_width = fitz.get_text_length(ch, fontname=fontname, fontsize=fontsize)

                # Check if character fits on current line
                if cursor_x + char_width > max_x:
                    # Flush buffer before wrapping
                    flush_buffer()

                    # Move to next line
                    cursor_x = bbox.x0 + margin_x
                    cursor_y += line_height

                    # Check for overflow
                    if cursor_y > bbox.y1 - margin_y - fontsize * 0.3:
                        print(f"Warning: Content overflow at y={cursor_y}")
                        return

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

                    if cursor_y > bbox.y1 - margin_y - fontsize * 0.3:
                        print(f"Warning: Content overflow at y={cursor_y}")
                        return

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


def render_table(page, bbox, table_html, fontsize=10):
    """
    Render HTML table on PDF page

    Args:
        page: fitz.Page object
        bbox: Tuple of (x0, y0, x1, y1) coordinates
        table_html: HTML table string
        fontsize: Base font size for table text
    """
    # Parse HTML table
    parser = TableHTMLParser()
    parser.feed(table_html)
    table_data = parser.get_table_data()

    headers = table_data['headers']
    rows = table_data['rows']

    if not headers:
        print("    [警告] 表格没有表头，跳过渲染")
        return

    # Load Unicode font using shared utility
    loaded_font, fontfile = load_unicode_font()

    # Calculate dimensions
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    num_cols = len(headers)
    num_rows = len(rows) + 1  # +1 for header row

    col_width = width / num_cols
    row_height = height / num_rows

    # Header font size (slightly larger)
    header_fontsize = fontsize + 1

    print(f"    [表格信息] 列数: {num_cols}, 行数: {num_rows}, 列宽: {col_width:.1f}pt, 行高: {row_height:.1f}pt")

    # Render header row with background
    for col_idx, header in enumerate(headers):
        cell_x0 = x0 + col_idx * col_width
        cell_y0 = y0
        cell_x1 = cell_x0 + col_width
        cell_y1 = cell_y0 + row_height

        # Draw cell border (no background fill)
        header_rect = fitz.Rect(cell_x0, cell_y0, cell_x1, cell_y1)
        page.draw_rect(header_rect, color=(0, 0, 0), width=0.5)

        # Draw header text (centered, bold) using loaded font for Unicode support
        if loaded_font:
            # Adaptive font size to fit text in cell
            adaptive_fontsize = header_fontsize
            min_fontsize = 6  # Minimum readable font size
            cell_padding = 4  # Total horizontal padding
            max_text_width = col_width - cell_padding

            text_width = loaded_font.text_length(header, fontsize=adaptive_fontsize)
            while text_width > max_text_width and adaptive_fontsize > min_fontsize:
                adaptive_fontsize -= 0.5
                text_width = loaded_font.text_length(header, fontsize=adaptive_fontsize)

            # Use loaded font with insert_text for proper Unicode rendering
            text_y = cell_y0 + row_height / 2 + adaptive_fontsize / 3

            # Center the text
            text_x = cell_x0 + (col_width - text_width) / 2

            # Use fontfile parameter with the font path
            # Remove spaces from font name as PyMuPDF doesn't accept them
            safe_fontname = get_safe_fontname(loaded_font)
            page.insert_text(
                point=(text_x, text_y),
                text=header,
                fontname=safe_fontname,
                fontfile=fontfile,
                fontsize=adaptive_fontsize,
                color=(0, 0, 0)
            )
        else:
            # Fall back to textbox
            text_rect = fitz.Rect(cell_x0 + 2, cell_y0, cell_x1 - 2, cell_y1)
            page.insert_textbox(
                text_rect,
                header,
                fontsize=header_fontsize,
                fontname="helv",
                align=fitz.TEXT_ALIGN_CENTER,
                color=(0, 0, 0)
            )

    # Render data rows
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            cell_x0 = x0 + col_idx * col_width
            cell_y0 = y0 + (row_idx + 1) * row_height
            cell_x1 = cell_x0 + col_width
            cell_y1 = cell_y0 + row_height

            # Draw cell border
            cell_rect = fitz.Rect(cell_x0, cell_y0, cell_x1, cell_y1)
            page.draw_rect(cell_rect, color=(0, 0, 0), width=0.5)

            # Get cell text and properties
            text = cell['text']
            is_bold = cell['bold']

            # Color: black for bold, dark gray for normal
            color = (0, 0, 0) if is_bold else (0.2, 0.2, 0.2)

            # Insert text using loaded font for better Unicode support
            if loaded_font:
                # Adaptive font size to fit text in cell
                adaptive_fontsize = fontsize
                min_fontsize = 5  # Minimum readable font size for data cells
                cell_padding = 6  # Total horizontal padding
                max_text_width = col_width - cell_padding

                text_width = loaded_font.text_length(text, fontsize=adaptive_fontsize)
                while text_width > max_text_width and adaptive_fontsize > min_fontsize:
                    adaptive_fontsize -= 0.5
                    text_width = loaded_font.text_length(text, fontsize=adaptive_fontsize)

                # Use loaded font with insert_text for proper Unicode rendering
                # Calculate text position based on alignment
                if is_numeric_text(text):
                    # Right align numbers
                    text_x = cell_x1 - text_width - 3
                else:
                    # Left align text
                    text_x = cell_x0 + 3

                text_y = cell_y0 + row_height / 2 + adaptive_fontsize / 3

                # Use fontfile parameter with the font path
                # Remove spaces from font name as PyMuPDF doesn't accept them
                safe_fontname = get_safe_fontname(loaded_font)
                page.insert_text(
                    point=(text_x, text_y),
                    text=text,
                    fontname=safe_fontname,
                    fontfile=fontfile,
                    fontsize=adaptive_fontsize,
                    color=color
                )
            else:
                # Fall back to textbox (automatically handles overflow)
                if is_numeric_text(text):
                    text_rect = fitz.Rect(cell_x0 + 2, cell_y0, cell_x1 - 3, cell_y1)
                    align = fitz.TEXT_ALIGN_RIGHT
                else:
                    text_rect = fitz.Rect(cell_x0 + 3, cell_y0, cell_x1 - 2, cell_y1)
                    align = fitz.TEXT_ALIGN_LEFT

                page.insert_textbox(
                    text_rect,
                    text,
                    fontsize=fontsize,
                    fontname="helv",
                    align=align,
                    color=color
                )

    print("    [OK] 表格渲染完成")


def process_pdf_with_json(pdf_path: str, json_path: str, output_path: str = None, image_dpi: int = 200, draw_bbox: bool = False):
    """
    Process PDF with OCR JSON: replace letters with 'c' and render back.

    Args:
        pdf_path: Input PDF file path
        json_path: OCR JSON file path
        output_path: Output PDF path (optional)
        image_dpi: DPI of the image used for OCR (default: 200)
        draw_bbox: Whether to draw bbox borders for debugging (default: False)
    """
    print("\n" + "=" * 80)
    print("PDF文本覆盖处理 - 字母替换为'c'")
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

            # Draw background rectangle
            rect = fitz.Rect(bbox)
            page.draw_rect(rect, color=bg_color_norm, fill=bg_color_norm)

            # ========== Category-based Rendering ==========
            if category == 'table':
                # Table rendering path
                try:
                    print(f"    [表格] 元素 {elem_idx + 1}")
                    render_table(page, bbox, text, fontsize=9)
                except Exception as e:
                    print(f"    [错误] 表格渲染失败: {e}")
                    page.insert_textbox(rect, "错误: 表格渲染失败",
                                       fontsize=10, color=(1, 0, 0))
            else:
                # Text/Title/Caption rendering path (existing logic)
                # Replace letters with 'c'
                # modified_text = replace_letters_preserve_formulas(text) # TODO
                modified_text = text

                # Calculate adaptive font size
                text_only = re.sub(r'\$.*?\$', 'FORMULA', modified_text)
                fontsize = calculate_adaptive_font_size(text_only, rect, default_size=12, min_size=4)

                # Render mixed content
                render_mixed_content(page, modified_text, rect, fontsize)

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

    args = parser.parse_args()

    process_pdf_with_json(args.pdf_path, args.json_path, args.output, args.dpi, args.draw_bbox)


if __name__ == "__main__":
    main()
