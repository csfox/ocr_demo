"""
Demo 2: Background Color Detection

This demo demonstrates:
1. Creating a PDF with different colored rectangles
2. Detecting background colors from specific regions
3. Sampling and analyzing pixel data from PDF pages
4. Visualizing detected colors
"""

import fitz  # PyMuPDF
import numpy as np
from collections import Counter


def detect_background_color(page: fitz.Page, bbox: tuple) -> tuple:
    """
    Detect the background color from a specific region of a PDF page.

    Args:
        page: PyMuPDF page object
        bbox: Tuple of (x0, y0, x1, y1) coordinates

    Returns:
        RGB tuple (r, g, b) in range 0-255
    """
    # Create a rect from bbox
    rect = fitz.Rect(bbox)

    # Get pixmap (image) from the PDF page region
    # Use a moderate DPI for sampling (150 is sufficient for color detection)
    mat = fitz.Matrix(150/72, 150/72)  # 150 DPI scaling
    pix = page.get_pixmap(matrix=mat, clip=rect)

    # Convert pixmap to numpy array
    # PyMuPDF pixmap format: samples are in RGB or RGBA
    img_data = np.frombuffer(pix.samples, dtype=np.uint8)

    # Reshape based on dimensions and number of components
    if pix.n == 3:  # RGB
        img_array = img_data.reshape(pix.height, pix.width, 3)
    elif pix.n == 4:  # RGBA
        img_array = img_data.reshape(pix.height, pix.width, 4)[:, :, :3]  # Drop alpha
    else:
        # Grayscale or other format
        img_array = img_data.reshape(pix.height, pix.width, pix.n)

    # Sample edges and corners to avoid text in the center
    # Strategy: sample pixels from the border regions
    border_width = max(2, min(pix.width, pix.height) // 10)

    # Get border pixels (top, bottom, left, right edges)
    border_pixels = []

    # Top edge
    border_pixels.append(img_array[:border_width, :, :].reshape(-1, 3))
    # Bottom edge
    border_pixels.append(img_array[-border_width:, :, :].reshape(-1, 3))
    # Left edge
    border_pixels.append(img_array[:, :border_width, :].reshape(-1, 3))
    # Right edge
    border_pixels.append(img_array[:, -border_width:, :].reshape(-1, 3))

    # Concatenate all border pixels
    all_border_pixels = np.vstack(border_pixels)

    # Find the most common color (mode)
    # Convert RGB tuples to hashable format for Counter
    pixel_tuples = [tuple(pixel) for pixel in all_border_pixels]
    color_counts = Counter(pixel_tuples)

    # Get the most common color
    most_common_color = color_counts.most_common(1)[0][0]

    return most_common_color


def create_test_pdf_with_colors():
    """
    Create a test PDF with different colored rectangles.
    """
    # Create a new PDF document
    doc = fitz.open()

    # Add a blank page
    page = doc.new_page(width=595, height=842)

    # Define test colors and their positions
    test_regions = [
        {"name": "Red", "color": (1, 0, 0), "rect": fitz.Rect(50, 100, 250, 200)},
        {"name": "Green", "color": (0, 1, 0), "rect": fitz.Rect(270, 100, 470, 200)},
        {"name": "Blue", "color": (0, 0, 1), "rect": fitz.Rect(50, 220, 250, 320)},
        {"name": "Yellow", "color": (1, 1, 0), "rect": fitz.Rect(270, 220, 470, 320)},
        {"name": "Cyan", "color": (0, 1, 1), "rect": fitz.Rect(50, 340, 250, 440)},
        {"name": "Magenta", "color": (1, 0, 1), "rect": fitz.Rect(270, 340, 470, 440)},
        {"name": "White", "color": (1, 1, 1), "rect": fitz.Rect(50, 460, 250, 560)},
        {"name": "Gray", "color": (0.5, 0.5, 0.5), "rect": fitz.Rect(270, 460, 470, 560)},
        {"name": "Dark Red", "color": (0.5, 0, 0), "rect": fitz.Rect(50, 580, 250, 680)},
        {"name": "Light Blue", "color": (0.7, 0.85, 1), "rect": fitz.Rect(270, 580, 470, 680)},
    ]

    # Draw title
    title_rect = fitz.Rect(50, 30, 545, 80)
    page.insert_textbox(
        title_rect,
        "Background Color Detection Test",
        fontsize=24,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER
    )

    # Draw colored rectangles
    for region in test_regions:
        # Draw filled rectangle
        page.draw_rect(region["rect"], color=(0, 0, 0), fill=region["color"], width=2)

        # Add label inside the rectangle
        label_rect = fitz.Rect(
            region["rect"].x0 + 10,
            region["rect"].y0 + 35,
            region["rect"].x1 - 10,
            region["rect"].y1 - 35
        )
        # Choose text color based on background brightness
        r, g, b = region["color"]
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = (0, 0, 0) if brightness > 0.5 else (1, 1, 1)

        page.insert_textbox(
            label_rect,
            region["name"],
            fontsize=14,
            fontname="helv",
            color=text_color,
            align=fitz.TEXT_ALIGN_CENTER
        )

    # Save the test PDF
    test_pdf_path = "output/test_colors.pdf"
    doc.save(test_pdf_path)
    print(f"[SUCCESS] Test PDF created: {test_pdf_path}")

    return test_pdf_path, test_regions


def test_color_detection():
    """
    Test color detection on the created PDF.
    """
    print("\n" + "="*60)
    print("Testing Background Color Detection")
    print("="*60 + "\n")

    # Create test PDF
    pdf_path, test_regions = create_test_pdf_with_colors()

    # Open the PDF for color detection
    doc = fitz.open(pdf_path)
    page = doc[0]

    # Test color detection on each region
    print(f"{'Region':<15} {'Expected RGB':<20} {'Detected RGB':<20} {'Match':<10}")
    print("-" * 70)

    all_match = True
    detection_results = []  # Store results for visualization
    for region in test_regions:
        # Expected color (convert from 0-1 to 0-255)
        expected_rgb = tuple(int(c * 255) for c in region["color"])

        # Detect color
        detected_rgb = detect_background_color(page, region["rect"])

        # Store for visualization
        detection_results.append({
            "name": region["name"],
            "original_color": region["color"],
            "detected_rgb": detected_rgb
        })

        # Check if colors match (allow small tolerance due to compression)
        tolerance = 10
        match = all(abs(e - d) <= tolerance for e, d in zip(expected_rgb, detected_rgb))
        match_str = "[OK]" if match else "[DIFF]"

        if not match:
            all_match = False

        print(f"{region['name']:<15} {str(expected_rgb):<20} {str(detected_rgb):<20} {match_str:<10}")

    doc.close()

    print("\n" + "="*60)
    if all_match:
        print("Result: All colors detected correctly!")
    else:
        print("Result: Some colors have slight differences (may be due to PDF compression)")
    print("="*60 + "\n")

    # Create visualization PDF with detected colors
    create_visualization_pdf(detection_results)


def create_visualization_pdf(detection_results: list):
    """
    Create a visualization PDF showing detected colors next to original colors.
    """
    # Create new document for visualization
    doc = fitz.open()
    vis_page = doc.new_page(width=595, height=842)

    # Draw title
    title_rect = fitz.Rect(50, 30, 545, 80)
    vis_page.insert_textbox(
        title_rect,
        "Color Detection Results (Original vs Detected)",
        fontsize=20,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER
    )

    y_offset = 100
    for result in detection_results:
        # Get colors
        original_color = result["original_color"]
        detected_rgb = result["detected_rgb"]
        detected_color = tuple(c / 255.0 for c in detected_rgb)

        # Draw original color
        original_rect = fitz.Rect(50, y_offset, 180, y_offset + 50)
        vis_page.draw_rect(original_rect, color=(0, 0, 0), fill=original_color, width=1)

        # Draw detected color
        detected_rect = fitz.Rect(190, y_offset, 320, y_offset + 50)
        vis_page.draw_rect(detected_rect, color=(0, 0, 0), fill=detected_color, width=1)

        # Add label
        label_rect = fitz.Rect(330, y_offset, 545, y_offset + 50)
        vis_page.insert_textbox(
            label_rect,
            f"{result['name']}: RGB{detected_rgb}",
            fontsize=11,
            fontname="helv",
            color=(0, 0, 0)
        )

        y_offset += 60

    # Save visualization
    output_path = "output/demo_color_detect.pdf"
    doc.save(output_path)
    doc.close()

    print(f"[SUCCESS] Visualization PDF created: {output_path}")
    print("   - Shows original colors (left) vs detected colors (right)")
    print("\nPlease open both PDFs to verify the results.")


if __name__ == "__main__":
    test_color_detection()
