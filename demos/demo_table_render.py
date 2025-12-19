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
from io import BytesIO
from weasyprint import HTML, CSS


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


def render_table_with_weasyprint(page, bbox, table_html, fontsize=10):
    """
    使用WeasyPrint渲染HTML表格到PDF页面的指定区域

    利用浏览器引擎自动计算列宽，实现内容自适应布局。

    Args:
        page: fitz.Page对象
        bbox: Tuple of (x0, y0, x1, y1) 目标区域坐标
        table_html: HTML表格字符串
        fontsize: 基础字体大小 (pt)
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    print(f"\n{'='*80}")
    print(f"WeasyPrint Table Rendering:")
    print(f"  Target bbox: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
    print(f"  Size: {width:.1f} x {height:.1f} pt")
    print(f"  Font size: {fontsize}pt")
    print(f"{'='*80}\n")

    # 构建完整的HTML文档，包含CSS样式
    # 使用table-layout: auto让浏览器自动计算列宽
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: {width}pt {height * 3}pt;  /* 给足够高度，稍后裁剪 */
                margin: 0;
            }}
            body {{
                margin: 0;
                padding: 0;
                font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
            }}
            table {{
                width: {width}pt;
                border-collapse: collapse;
                font-size: {fontsize}pt;
                table-layout: auto;  /* 关键：自动计算列宽 */
            }}
            th, td {{
                border: 0.5pt solid black;
                padding: 4pt 6pt;
                text-align: left;
                vertical-align: middle;
                word-wrap: break-word;
            }}
            th {{
                font-weight: bold;
                background-color: #f5f5f5;
            }}
            /* 数字右对齐 */
            td:nth-child(n+3) {{
                text-align: right;
            }}
            strong {{
                font-weight: bold;
                color: #000;
            }}
        </style>
    </head>
    <body>{table_html}</body>
    </html>
    """

    # 使用WeasyPrint渲染为PDF字节流
    print("  Rendering HTML with WeasyPrint...")
    pdf_bytes = BytesIO()
    HTML(string=full_html).write_pdf(pdf_bytes)
    pdf_bytes.seek(0)

    # 用PyMuPDF打开渲染后的PDF
    temp_doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
    temp_page = temp_doc[0]

    # 获取实际渲染内容的边界（去除空白）
    # 使用get_text("dict")获取文本块来确定实际内容区域
    src_rect = temp_page.rect

    # 计算缩放比例，使表格适配目标bbox
    # 优先保证宽度适配，高度按比例缩放
    scale_x = width / src_rect.width if src_rect.width > 0 else 1
    scale_y = height / src_rect.height if src_rect.height > 0 else 1

    # 使用较小的缩放比例，保证表格完全在bbox内
    scale = min(scale_x, scale_y, 1.0)  # 不放大，只缩小

    # 计算实际渲染尺寸
    actual_width = src_rect.width * scale
    actual_height = src_rect.height * scale

    # 计算目标矩形（居中或左上对齐）
    # 这里使用左上对齐
    target_rect = fitz.Rect(
        x0,
        y0,
        x0 + actual_width,
        y0 + actual_height
    )

    print(f"  Source size: {src_rect.width:.1f} x {src_rect.height:.1f} pt")
    print(f"  Scale factor: {scale:.3f}")
    print(f"  Actual rendered size: {actual_width:.1f} x {actual_height:.1f} pt")

    # 将渲染结果嵌入到目标页面
    page.show_pdf_page(target_rect, temp_doc, 0)

    temp_doc.close()
    print("[OK] WeasyPrint table rendering complete\n")


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

    # Add title to page 1 (before rendering tables)
    page.insert_text(
        point=(50, 30),
        text="WeasyPrint Adaptive Table Rendering Demo",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0)
    )

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

    # 使用WeasyPrint渲染 - 自动计算列宽
    bbox_1 = (50, 50, 245, 250)  # Adjusted for A4 page
    render_table_with_weasyprint(page, bbox_1, table_html_1, fontsize=9)

    # Example 2: 内容不均匀的表格（测试自适应列宽）
    print("\nExample 2: Uneven Content Table (Testing Adaptive Column Width)")
    print("-" * 80)

    table_html_2 = """<table>
<thead>
<tr>
<th>ID</th>
<th>Description</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>This is a very long description that demonstrates how the column width adapts to content automatically</td>
<td>42.5</td>
</tr>
<tr>
<td>2</td>
<td>Short text</td>
<td><strong>99.9</strong></td>
</tr>
<tr>
<td>3</td>
<td>Medium length description here</td>
<td>73.2</td>
</tr>
</tbody>
</table>"""

    bbox_2 = (50, 280, 245, 450)
    render_table_with_weasyprint(page, bbox_2, table_html_2, fontsize=10)

    # Example 3: 对比 - 使用原始平均分配方法
    print("\nExample 3: Original Method (Equal Column Width) for Comparison")
    print("-" * 80)

    page2 = doc.new_page(width=595, height=842)  # 新页面

    # 先添加所有标题文字（在渲染表格之前）
    page2.insert_text(
        point=(50, 30),
        text="WeasyPrint (Adaptive Width):",
        fontsize=12,
        fontname="helv",
        color=(0, 0, 0)
    )
    page2.insert_text(
        point=(315, 30),
        text="Original (Equal Width):",
        fontsize=12,
        fontname="helv",
        color=(0, 0, 0)
    )
    page2.insert_text(
        point=(150, 230),
        text="Comparison: Adaptive vs Equal Width",
        fontsize=14,
        fontname="helv",
        color=(0, 0, 0)
    )

    # 左边：WeasyPrint自适应
    bbox_3a = (50, 50, 280, 200)
    render_table_with_weasyprint(page2, bbox_3a, table_html_2, fontsize=8)

    # 右边：原始平均分配
    bbox_3b = (315, 50, 545, 200)
    render_table(page2, bbox_3b, table_html_2, fontsize=8)

    # Save PDF
    output_path = output_dir / "demo_table_render.pdf"
    doc.save(str(output_path))
    doc.close()

    print("\n" + "="*80)
    print(f"[OK] PDF saved to: {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
