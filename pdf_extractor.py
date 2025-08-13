import langextract as lx
import textwrap
import sys
import os
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 is not installed. Please install it using: pip install PyPDF2")
    sys.exit(1)

try:
    import docx
except ImportError:
    print("python-docx is not installed. Please install it using: pip install python-docx")
    sys.exit(1)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def main():
    # 1. Define the prompt and extraction rules
    with open('/Users/zhoulu/Documents/LLMCoding/langextract/prompt.md', 'r', encoding='utf-8') as f:
        prompt = f.read()

    # 2. Provide a high-quality example to guide the model
    example_text = """
合同名称：中国国际图书贸易集团有限公司数据库订购协议
签订日期：2024-09-26
甲方：北京达佳互联信息技术有限公司
乙方：中国国际图书贸易集团有限公司
服务内容：订购 Lexis®、Lexis Advance HK 数据库产品
合同金额：218000.00元
付款方式：后付费
"""

    examples = [
        lx.data.ExampleData(
            text=example_text,
            extractions=[
                lx.data.Extraction(
                    extraction_class="basic_contract_information",
                    extraction_text="中国国际图书贸易集团有限公司数据库订购协议",
                    attributes={"field": "contract_name"}
                ),
                lx.data.Extraction(
                    extraction_class="basic_contract_information",
                    extraction_text="2024-09-26",
                    attributes={"field": "contract_start_date"}
                ),
                lx.data.Extraction(
                    extraction_class="basic_contract_information",
                    extraction_text="北京达佳互联信息技术有限公司",
                    attributes={"field": "our_party_name"}
                ),
                lx.data.Extraction(
                    extraction_class="basic_contract_information",
                    extraction_text="中国国际图书贸易集团有限公司",
                    attributes={"field": "counterparty_name"}
                ),
                lx.data.Extraction(
                    extraction_class="product_details",
                    extraction_text="Lexis®、Lexis Advance HK",
                    attributes={"field": "goods_or_service_name"}
                ),
                lx.data.Extraction(
                    extraction_class="financial_details",
                    extraction_text="218000.00",
                    attributes={"field": "contract_amount"}
                ),
                lx.data.Extraction(
                    extraction_class="payment_information",
                    extraction_text="后付费",
                    attributes={"field": "payment_type"}
                ),
            ]
        )
    ]

    import argparse

    parser = argparse.ArgumentParser(description='Extract text from a PDF or DOCX file using Gemini API.')
    parser.add_argument('file_path', type=str, help='The path to the PDF or DOCX file.')
    parser.add_argument('--api_key', type=str, help='Your Gemini API key.', default=os.environ.get("GEMINI_API_KEY"))
    args = parser.parse_args()

    file_path = args.file_path
    api_key = args.api_key
    try:
        if file_path.lower().endswith('.pdf'):
            input_text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            input_text = extract_text_from_docx(file_path)
        else:
            print("Unsupported file format. Please provide a .pdf or .docx file.")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    if not api_key:
        print("Error: GEMINI_API_KEY not provided. Please provide it using the --api_key argument or by setting the GEMINI_API_KEY environment variable.")
        sys.exit(1)

    # Run the extraction
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-1.5-flash",
        api_key=api_key,
    )

    # Save the results to a JSONL file
    lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

    # Generate the visualization from the file
    html_content = lx.visualize("extraction_results.jsonl")
    with open("visualization.html", "w") as f:
        f.write(html_content)

    print("Extraction complete. Check 'visualization.html' for the results.")

if __name__ == "__main__":
    main()
