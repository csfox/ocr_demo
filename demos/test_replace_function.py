"""
Test the replace_letters_preserve_formulas function
"""

import re


def replace_letters_preserve_formulas(text: str) -> str:
    """
    Replace all letters with 'c' while preserving LaTeX formulas and spaces.
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


# Test cases
test_cases = [
    "Sample Document for Text Overlay Demo",
    "Einstein discovered that energy and mass are related.",
    "The famous equation is $E=mc^2$ which is energy equals mass times speed of light squared.",
    "Newton's second law describes motion with $F=ma$ which is force equals mass times acceleration.",
    "The year 2024 marks 100 years since the theory was published.",
]

print("\n" + "=" * 80)
print("测试文本替换函数")
print("=" * 80)

for i, text in enumerate(test_cases, 1):
    result = replace_letters_preserve_formulas(text)
    print(f"\n[测试 {i}]")
    print(f"原文: {text}")
    print(f"结果: {result}")

    # Check if formulas are preserved
    formulas_in_original = re.findall(r'\$(.+?)\$', text)
    formulas_in_result = re.findall(r'\$(.+?)\$', result)

    if formulas_in_original:
        if formulas_in_original == formulas_in_result:
            print(f"[OK] Formula preserved: {formulas_in_result}")
        else:
            print(f"[ERROR] Formula lost! Original: {formulas_in_original}, Result: {formulas_in_result}")

print("\n" + "=" * 80)
