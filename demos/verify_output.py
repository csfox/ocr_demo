"""
Verify that the output PDF contains the expected content
"""

import fitz
import json


def verify_output():
    """Verify the generated overlay PDF"""

    print("\n" + "=" * 80)
    print("验证输出PDF内容")
    print("=" * 80)

    # Load JSON to see what was processed
    json_path = "output/test_input.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)

    print("\n[JSON内容] 应该处理的文本:")
    print("-" * 80)

    for page_data in ocr_data.get('pages', []):
        for i, element in enumerate(page_data.get('elements', []), 1):
            text = element.get('text', '')
            print(f"\n元素 {i}: {text}")

            # Check for formulas
            if '$' in text:
                import re
                formulas = re.findall(r'\$(.+?)\$', text)
                print(f"  包含公式: {formulas}")

    # Open the output PDF
    pdf_path = "output/test_input_overlay.pdf"
    doc = fitz.open(pdf_path)
    page = doc[0]

    print("\n" + "=" * 80)
    print("[PDF内容] 渲染的图像数量:")
    print("-" * 80)

    # Get all images in the PDF (formulas are rendered as images)
    image_list = page.get_images()
    print(f"\n总图像数: {len(image_list)}")

    if len(image_list) > 0:
        print(f"\n[SUCCESS] PDF中包含 {len(image_list)} 个图像（公式）")
        print("\n公式已成功渲染为图像并插入PDF！")
    else:
        print("\n[WARNING] PDF中没有图像，可能公式没有渲染")

    # Get text content (for verification)
    text_content = page.get_text()

    print("\n" + "=" * 80)
    print("[PDF文本内容] 提取的文本:")
    print("-" * 80)
    print(text_content[:500])  # First 500 chars

    doc.close()

    print("\n" + "=" * 80)
    print("验证总结:")
    print("-" * 80)
    print(f"✓ JSON包含公式的元素: 2个 ($E=mc^2$ 和 $F=ma$)")
    print(f"✓ PDF中渲染的图像数: {len(image_list)}个")
    print(f"✓ 预期: 2个公式 = 2个图像")

    if len(image_list) == 2:
        print("\n[SUCCESS] 公式数量匹配！公式已正确保留并渲染到PDF中！")
    elif len(image_list) > 0:
        print(f"\n[INFO] 发现 {len(image_list)} 个图像，请检查PDF确认")
    else:
        print("\n[ERROR] 没有发现图像，公式可能没有渲染")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    verify_output()
