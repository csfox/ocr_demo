"""
Demo 1: PDF Text and LaTeX Rendering

This demo demonstrates:
1. Creating a blank PDF using PyMuPDF
2. Rendering plain text on the PDF
3. Converting LaTeX formulas to images using matplotlib
4. Inserting LaTeX formula images onto the PDF
"""

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from matplotlib import mathtext
import io
from PIL import Image


def render_latex_to_bytes(latex_formula: str, fontsize: int = 20, dpi: int = 300) -> bytes:
    """
    Render a LaTeX formula to PNG image bytes using matplotlib.

    Args:
        latex_formula: LaTeX formula string (without $ delimiters)
        fontsize: Font size for rendering
        dpi: DPI for image resolution

    Returns:
        PNG image as bytes
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)

    # Render the LaTeX formula
    # Use mathtext for rendering (no full LaTeX installation needed)
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
    return buf.getvalue()


def create_demo_pdf():
    """
    Create a demo PDF with text and LaTeX formulas.
    """
    # Create a new PDF document
    doc = fitz.open()

    # Add a blank page (A4 size: 595 x 842 points)
    page = doc.new_page(width=595, height=842)

    # Draw a title
    title_rect = fitz.Rect(50, 50, 545, 100)
    page.insert_textbox(
        title_rect,
        "PDF Text and LaTeX Rendering Demo",
        fontsize=24,
        fontname="helv",
        fontfile=None,
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER
    )

    # Add plain text
    text_rect = fitz.Rect(50, 120, 545, 200)
    page.insert_textbox(
        text_rect,
        "This is plain text rendered using PyMuPDF. Below are some LaTeX formulas:",
        fontsize=14,
        fontname="helv",
        color=(0, 0, 0)
    )

    # Define LaTeX formulas to render
    formulas = [
        ("Einstein's mass-energy equivalence:", "E = mc^2", 220),
        ("Newton's second law:", "F = ma", 300),
        ("Pythagorean theorem:", "a^2 + b^2 = c^2", 380),
        ("Quadratic formula:", r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", 460),
        ("Euler's identity:", r"e^{i\pi} + 1 = 0", 560),
    ]

    # Render each formula
    for label, formula, y_pos in formulas:
        # Add label text
        label_rect = fitz.Rect(50, y_pos, 300, y_pos + 50)
        page.insert_textbox(
            label_rect,
            label,
            fontsize=12,
            fontname="helv",
            color=(0, 0, 1)  # Blue color
        )

        # Render LaTeX formula to image
        img_bytes = render_latex_to_bytes(formula, fontsize=16, dpi=300)

        # Insert the image onto the PDF
        img_rect = fitz.Rect(50, y_pos + 25, 545, y_pos + 65)
        page.insert_image(img_rect, stream=img_bytes, keep_proportion=True)

    # Add a footer
    footer_rect = fitz.Rect(50, 780, 545, 820)
    page.insert_textbox(
        footer_rect,
        "Demo completed successfully! All LaTeX formulas were rendered as images.",
        fontsize=10,
        fontname="helv",
        color=(0.5, 0.5, 0.5),  # Gray color
        align=fitz.TEXT_ALIGN_CENTER
    )

    # Save the PDF
    output_path = "output/demo_render.pdf"
    doc.save(output_path)
    doc.close()

    print(f"[SUCCESS] Demo PDF created successfully: {output_path}")
    print(f"   - Plain text rendered")
    print(f"   - {len(formulas)} LaTeX formulas rendered as images")
    print(f"\nPlease open the PDF to verify the results.")


if __name__ == "__main__":
    create_demo_pdf()
