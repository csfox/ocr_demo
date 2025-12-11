"""
Demo: Render HTML Table to PDF

This demo demonstrates how to parse an HTML table and render it on a PDF
with proper formatting, borders, and text alignment.

Usage:
    python demos/demo_table_render.py

Output:
    output/demo_table_render.pdf
"""

import fitz  # PyMuPDF
from html.parser import HTMLParser
import re
from pathlib import Path


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


def is_numeric_text(text):
    """Check if text is primarily numeric (for right alignment)"""
    if not text:
        return False
    # Check if text matches number patterns: digits, dots, dashes, daggers
    return bool(re.match(r'^[\d.—†\-]+$', text))


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
        print("Warning: No headers found in table")
        return

    # Try to find a Unicode-capable font on the system
    # Priority: fonts with best Unicode coverage for arrows and special chars
    font_candidates = [
        "C:/Windows/Fonts/seguisym.ttf",        # Segoe UI Symbol (best for symbols)
        "C:/Windows/Fonts/arialuni.ttf",        # Arial Unicode MS
        "C:/Windows/Fonts/msgothic.ttc",        # MS Gothic (good Unicode)
        "C:/Windows/Fonts/arial.ttf",           # Arial (basic)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]

    fontfile = None
    loaded_font = None
    for font_path in font_candidates:
        if Path(font_path).exists():
            try:
                # Try to load font using PyMuPDF's Font class for better embedding
                loaded_font = fitz.Font(fontfile=font_path)
                fontfile = font_path
                print(f"Using font: {font_path}")
                break
            except Exception as e:
                print(f"Failed to load {font_path}: {e}")
                continue

    if fontfile is None:
        print("Warning: No Unicode font found, using fallback font")
        # Try using a built-in CJK font which often has better Unicode coverage
        try:
            loaded_font = fitz.Font("cjk")
            print("Using built-in CJK font")
        except:
            loaded_font = None

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

    print(f"\n{'='*80}")
    print(f"Table Rendering Info:")
    print(f"  Columns: {num_cols}")
    print(f"  Rows: {num_rows} (including header)")
    print(f"  Column width: {col_width:.1f}pt")
    print(f"  Row height: {row_height:.1f}pt")
    print(f"  Font size: {fontsize}pt")
    print(f"{'='*80}\n")

    # Render header row with background
    print("Rendering header row...")
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
            # Use loaded font with insert_text for proper Unicode rendering
            text_x = cell_x0 + col_width / 2
            text_y = cell_y0 + row_height / 2 + header_fontsize / 3

            # Center the text
            text_width = loaded_font.text_length(header, fontsize=header_fontsize)
            text_x = cell_x0 + (col_width - text_width) / 2

            # Use fontfile parameter with the font path
            # Remove spaces from font name as PyMuPDF doesn't accept them
            safe_fontname = loaded_font.name.replace(" ", "")
            page.insert_text(
                point=(text_x, text_y),
                text=header,
                fontname=safe_fontname,
                fontfile=fontfile,
                fontsize=header_fontsize,
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
    print(f"Rendering {len(rows)} data rows...")
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
                # Use loaded font with insert_text for proper Unicode rendering
                # Calculate text position based on alignment
                if is_numeric_text(text):
                    # Right align numbers
                    text_width = loaded_font.text_length(text, fontsize=fontsize)
                    text_x = cell_x1 - text_width - 3
                else:
                    # Left align text
                    text_x = cell_x0 + 3

                text_y = cell_y0 + row_height / 2 + fontsize / 3

                # Use fontfile parameter with the font path
                # Remove spaces from font name as PyMuPDF doesn't accept them
                safe_fontname = loaded_font.name.replace(" ", "")
                page.insert_text(
                    point=(text_x, text_y),
                    text=text,
                    fontname=safe_fontname,
                    fontfile=fontfile,
                    fontsize=fontsize,
                    color=color
                )
            else:
                # Fall back to textbox
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

    print("[OK] Table rendering complete\n")


def main():
    """Main function to demonstrate table rendering"""
    print("\n" + "="*80)
    print("HTML Table Rendering Demo")
    print("="*80)

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Create PDF document
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    # Example 1: Simple table from your data
    print("\nExample 1: VAR Model Comparison Table")
    print("-" * 80)

    table_html_1 = """<table>
<thead>
<tr>
<th>Type</th>
<th>Model</th>
<th>FID↓</th>
<th>IS↑</th>
<th>Time</th>
</tr>
</thead>
<tbody>
<tr>
<td>GAN</td>
<td>BigGAN [13]</td>
<td>8.43</td>
<td>177.9</td>
<td>—</td>
</tr>
<tr>
<td>Diff.</td>
<td>ADM [26]</td>
<td>23.24</td>
<td>101.0</td>
<td>—</td>
</tr>
<tr>
<td>Diff.</td>
<td>DiT-XL/2 [63]</td>
<td>3.04</td>
<td>240.8</td>
<td>81</td>
</tr>
<tr>
<td>Mask.</td>
<td>MaskGIT [17]</td>
<td>7.32</td>
<td>156.0</td>
<td>0.5†</td>
</tr>
<tr>
<td>AR</td>
<td>VQGAN [30]</td>
<td>26.52</td>
<td>66.8</td>
<td>25†</td>
</tr>
<tr>
<td>VAR</td>
<td>VAR-d36-s</td>
<td><strong>2.63</strong></td>
<td><strong>303.2</strong></td>
<td>1</td>
</tr>
</tbody>
</table>"""

    # Render at original position (scaled from your bbox)
    bbox_1 = (50, 50, 545, 250)  # Adjusted for A4 page
    render_table(page, bbox_1, table_html_1, fontsize=9)

    # Example 2: Another table on the same page
    print("\nExample 2: Simple Performance Table")
    print("-" * 80)

    table_html_2 = """<table>
<thead>
<tr>
<th>Method</th>
<th>Accuracy</th>
<th>Speed</th>
</tr>
</thead>
<tbody>
<tr>
<td>Method A</td>
<td>95.2</td>
<td>Fast</td>
</tr>
<tr>
<td>Method B</td>
<td><strong>98.7</strong></td>
<td>Slow</td>
</tr>
<tr>
<td>Method C</td>
<td>96.1</td>
<td>Medium</td>
</tr>
</tbody>
</table>"""

    bbox_2 = (50, 280, 350, 400)
    render_table(page, bbox_2, table_html_2, fontsize=10)

    # Add title
    page.insert_text(
        point=(50, 30),
        text="HTML Table Rendering Demo - PyMuPDF",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0)
    )

    # Save PDF
    output_path = output_dir / "demo_table_render.pdf"
    doc.save(str(output_path))
    doc.close()

    print("\n" + "="*80)
    print(f"[OK] PDF saved to: {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
