"""
Demo 1 Enhanced: PDF Text and LaTeX Rendering with Advanced Features

This demo demonstrates:
1. Auto-adaptive font sizing (shrink-only, never grows)
2. Inline LaTeX formula rendering (formulas embedded within text)
3. Formula image scaling proportional to font size
4. Proper baseline alignment and line wrapping
"""

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from matplotlib import mathtext
import io
from PIL import Image
import re


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
            if text_segment:  # Only add non-empty segments
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

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                transparent=True, pad_inches=0.1)
    plt.close(fig)

    buf.seek(0)
    img_bytes = buf.getvalue()

    # Get actual image dimensions
    pil_img = Image.open(io.BytesIO(img_bytes))
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

    # Measure text width
    text_width = fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)

    # Calculate number of lines needed
    if text_width <= max_width:
        num_lines = 1
        actual_width = text_width
    else:
        # Estimate lines (rough approximation)
        num_lines = int(text_width / max_width) + 1
        actual_width = max_width

    # Calculate total height
    line_spacing = 1.2
    total_height = num_lines * fontsize * line_spacing

    return actual_width, total_height, num_lines


def calculate_adaptive_font_size(text: str, bbox: fitz.Rect,
                                   default_size: int = 12, min_size: int = 6,
                                   fontname: str = "helv") -> int:
    """
    Calculate adaptive font size (shrink-only).

    Args:
        text: Text to fit in bbox
        bbox: Bounding rectangle
        default_size: Default starting size (will not grow beyond this)
        min_size: Minimum size threshold

    Returns:
        Font size that fits the bbox
    """
    bbox_width = bbox.x1 - bbox.x0
    bbox_height = bbox.y1 - bbox.y0

    # Safety margins
    usable_width = bbox_width * 0.95
    usable_height = bbox_height * 0.95

    # Test at default size first
    _, height, _ = calculate_text_dimensions(text, fontname, default_size, usable_width)

    if height <= usable_height:
        return default_size  # Fits at default, don't enlarge

    # Binary search to find size that fits
    low, high = min_size, default_size
    best_size = min_size

    while low <= high:
        mid = (low + high) // 2
        _, height, _ = calculate_text_dimensions(text, fontname, mid, usable_width)

        if height <= usable_height:
            best_size = mid
            low = mid + 1  # Try slightly larger
        else:
            high = mid - 1  # Too big, shrink more

    return best_size


def render_mixed_content(page: fitz.Page, text: str, bbox: fitz.Rect,
                          fontsize: int, fontname: str = "helv"):
    """
    Render text with inline LaTeX formulas.

    Args:
        page: PDF page object
        text: Text with embedded LaTeX formulas
        bbox: Bounding rectangle
        fontsize: Font size to use
        fontname: Font name
    """
    # Parse mixed content
    segments = parse_mixed_content(text)

    # Initialize cursor at top-left of bbox (with small margin)
    margin = 5
    cursor_x = bbox.x0 + margin
    cursor_y = bbox.y0 + margin + fontsize  # Start one line down for baseline
    line_spacing = 1.2
    line_height = fontsize * line_spacing

    # Track max cursor X for wrapping
    max_x = bbox.x1 - margin

    for seg_type, content in segments:
        if seg_type == 'text':
            # Split text by spaces for word wrapping
            words = content.split(' ')

            for i, word in enumerate(words):
                # Add space before word (except first word)
                if i > 0:
                    word = ' ' + word

                # Measure word width
                word_width = fitz.get_text_length(word, fontname=fontname, fontsize=fontsize)

                # Check if word fits on current line
                if cursor_x + word_width > max_x:
                    # Move to next line
                    cursor_x = bbox.x0 + margin
                    cursor_y += line_height

                    # Check if we exceeded bbox height
                    if cursor_y > bbox.y1 - margin:
                        print(f"Warning: Content overflow at y={cursor_y}")
                        return

                    # Remove leading space if at start of line
                    if word.startswith(' '):
                        word = word[1:]
                        word_width = fitz.get_text_length(word, fontname=fontname, fontsize=fontsize)

                # Insert text at cursor position
                page.insert_text(
                    point=(cursor_x, cursor_y),
                    text=word,
                    fontsize=fontsize,
                    fontname=fontname,
                    color=(0, 0, 0)
                )

                # Advance cursor
                cursor_x += word_width

        elif seg_type == 'latex':
            # Render LaTeX formula to image
            try:
                formula_bytes, img_width_px, img_height_px = render_latex_to_bytes_with_size(
                    content, fontsize=fontsize, dpi=300
                )

                # Scale formula image proportional to font size
                # Base scale: fontsize 12 → scale 1.0
                scale_factor = fontsize / 12.0
                img_width_pt = img_width_px * scale_factor * 0.15  # Convert px to pt
                img_height_pt = img_height_px * scale_factor * 0.15

                # Check if formula fits on current line
                if cursor_x + img_width_pt > max_x:
                    # Move to next line
                    cursor_x = bbox.x0 + margin
                    cursor_y += line_height

                    if cursor_y > bbox.y1 - margin:
                        print(f"Warning: Content overflow at y={cursor_y}")
                        return

                # Calculate image rect to align baseline
                # Formula center should align with text baseline
                img_rect = fitz.Rect(
                    cursor_x,
                    cursor_y - img_height_pt * 0.65,  # Raise formula
                    cursor_x + img_width_pt,
                    cursor_y + img_height_pt * 0.35   # Lower formula
                )

                # Insert formula image
                page.insert_image(img_rect, stream=formula_bytes)

                # Advance cursor
                cursor_x += img_width_pt + 2  # Small spacing after formula

            except Exception as e:
                # Fallback: render formula as text
                print(f"Warning: Failed to render formula '{content}': {e}")
                fallback_text = f"${content}$"
                page.insert_text(
                    point=(cursor_x, cursor_y),
                    text=fallback_text,
                    fontsize=fontsize,
                    fontname=fontname,
                    color=(1, 0, 0)  # Red to indicate error
                )
                cursor_x += fitz.get_text_length(fallback_text, fontname=fontname, fontsize=fontsize)


def create_enhanced_demo():
    """
    Create enhanced demo PDF with adaptive sizing and inline formulas.
    """
    # Create a new PDF document
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Title
    title_rect = fitz.Rect(50, 30, 545, 70)
    page.insert_textbox(
        title_rect,
        "Enhanced Demo: Adaptive Sizing + Inline Formulas",
        fontsize=20,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER
    )

    # Test 1: Adaptive font sizing with different bbox sizes
    print("\n[Test 1] Adaptive Font Sizing")
    print("=" * 50)

    test_boxes = [
        (fitz.Rect(50, 100, 250, 180), "Small box"),
        (fitz.Rect(280, 100, 545, 180), "Medium box"),
        (fitz.Rect(50, 200, 545, 280), "Large box"),
    ]

    long_text = "This text automatically shrinks to fit the bounding box. The default size is 12pt, and it only reduces if the text doesn't fit."

    for bbox, label in test_boxes:
        # Draw bbox border
        page.draw_rect(bbox, color=(0, 0, 1), width=1)

        # Calculate adaptive font size
        fontsize = calculate_adaptive_font_size(long_text, bbox, default_size=12, min_size=6)
        print(f"{label}: fontsize={fontsize}pt")

        # Render text
        page.insert_textbox(
            bbox,
            long_text,
            fontsize=fontsize,
            fontname="helv",
            color=(0, 0, 0)
        )

    # Test 2: Inline LaTeX formulas
    print("\n[Test 2] Inline LaTeX Formula Rendering")
    print("=" * 50)

    mixed_tests = [
        ("Simple formula", "Einstein discovered $E=mc^2$ in physics.", fitz.Rect(50, 300, 545, 360)),
        ("Multiple formulas", "Compare $F=ma$ with $E=mc^2$ and $a^2+b^2=c^2$ concepts.", fitz.Rect(50, 380, 545, 440)),
        ("Complex formula", "The quadratic formula $x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}$ solves equations.", fitz.Rect(50, 460, 545, 520)),
    ]

    for label, mixed_text, bbox in mixed_tests:
        # Draw bbox border
        page.draw_rect(bbox, color=(0, 0.7, 0), width=1)

        # Add label
        label_rect = fitz.Rect(bbox.x0, bbox.y0 - 20, bbox.x1, bbox.y0)
        page.insert_textbox(label_rect, label, fontsize=10, fontname="helv", color=(0, 0.5, 0))

        # Calculate adaptive font size for the mixed content
        # For simplicity, use text-only version for sizing
        text_only = re.sub(r'\$.*?\$', 'FORMULA', mixed_text)
        fontsize = calculate_adaptive_font_size(text_only, bbox, default_size=12, min_size=8)
        print(f"{label}: fontsize={fontsize}pt")

        # Render mixed content
        render_mixed_content(page, mixed_text, bbox, fontsize)

    # Test 3: Formula scaling with different font sizes
    print("\n[Test 3] Formula Scaling")
    print("=" * 50)

    formula = "E=mc^2"
    y_start = 540
    font_sizes = [8, 12, 16, 20]

    for i, fs in enumerate(font_sizes):
        y_pos = y_start + i * 70
        bbox = fitz.Rect(50, y_pos, 545, y_pos + 60)

        # Draw bbox
        page.draw_rect(bbox, color=(0.7, 0, 0.7), width=1)

        # Label
        label_text = f"Font {fs}pt:"
        page.insert_text(
            point=(bbox.x0 + 5, bbox.y0 + 20),
            text=label_text,
            fontsize=10,
            fontname="helv",
            color=(0.5, 0, 0.5)
        )

        # Render mixed content with specific font size
        test_text = f"Formula at size {fs}: ${formula}$ scales proportionally."
        render_mixed_content(page, test_text, bbox, fontsize=fs)

        print(f"Fontsize {fs}pt: formula rendered")

    # Save PDF
    output_path = "output/demo_render_enhanced.pdf"
    doc.save(output_path)
    doc.close()

    print("\n" + "=" * 50)
    print(f"[SUCCESS] Enhanced demo PDF created: {output_path}")
    print("Features demonstrated:")
    print("  - Adaptive font sizing (shrink-only)")
    print("  - Inline LaTeX formula rendering")
    print("  - Formula image scaling with font size")
    print("  - Baseline alignment and line wrapping")
    print("\nPlease open the PDF to verify the results.")


if __name__ == "__main__":
    create_enhanced_demo()
