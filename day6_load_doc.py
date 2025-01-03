import fitz  # PyMuPDF
import re
import uuid
import os
import requests
from dotenv import load_dotenv
import tiktoken
from load_pdf import load_pdf

def extract_html_from_pdf(pdf_document: fitz.Document) -> str:
    """Extract the full HTML content from the PDF using PyMuPDF (fitz)."""
    if not pdf_document:
        print("PDF document is not loaded correctly. Exiting extraction process.")
        return ""
    
    full_html = ""
    try:
        for page_num, page in enumerate(pdf_document):
            page_text = page.get_textpage().extractXHTML()
            full_html += page_text + "\n"
        print("Successfully extracted HTML from PDF.")
    except Exception as e:
        print(f"Error while extracting HTML from PDF. Error: {e}")
    
    return full_html

def extract_headers_and_content_from_html(html: str, document_name: str, document_id: str):
    """Extract headers and content from the XHTML of the PDF."""
    structured_data = []
    current_header1 = None
    current_header2 = None
    current_header3 = None
    current_header4 = None
    content_buffer = []

    header1_pattern = r'<h1><b>(.*?)</b></h1>|<h2><b>(.*?)</b></h2>'  # Matches <h1><b> and <h2><b> headers
    header2_pattern = r'<p><b>(\d+\s.*?)</b></p>'  # Matches headers like "1 Introduction"
    header3_pattern = r'<p><b>(\d+\.\d+\s.*?)</b></p>'  # Matches headers like "3.2 Encoder"
    header4_pattern = r'<p><b>(\d+\.\d+\.\d+\s.*?)</b></p>'  # Matches headers like "3.2.2 Multi-Head Attention"
    content_pattern = r'<p>(.*?)</p>'  # Extract plain paragraph content

    # Combine all headers into a list of matches
    matches = re.findall(f'{header1_pattern}|{header2_pattern}|{header3_pattern}|{header4_pattern}|{content_pattern}', html)

    for match in matches:
        header1_text = match[0] or match[1]  # Match header1 in <h1> or <h2>
        header2_text = match[2]  # Match header2 pattern
        header3_text = match[3]  # Match header3 pattern
        header4_text = match[4]  # Match header4 pattern
        content_text = match[5]  # Extract text inside <p> tags

        # Handle Header 1 (like "Attention Is All You Need")
        if header1_text:
            if content_buffer:
                structured_data.append({
                    "page_content": " ".join(content_buffer).strip(),
                    "metadata": {
                        "header1": current_header1,
                        "header2": current_header2,
                        "header3": current_header3,
                        "header4": current_header4,
                        "document_name": document_name,
                        "document_id": document_id,
                       
                    }
                })
                content_buffer = []

            current_header1 = header1_text.strip()
            current_header2 = None
            current_header3 = None
            current_header4 = None

        # Handle Header 2 (like "1 Introduction")
        elif header2_text:
            if content_buffer:
                structured_data.append({
                    "page_content": " ".join(content_buffer).strip(),
                    "metadata": {
                        "header1": current_header1,
                        "header2": current_header2,
                        "header3": current_header3,
                        "header4": current_header4,
                        "document_name": document_name,
                        "document_id": document_id,
                       
                    }
                })
                content_buffer = []

            current_header2 = header2_text.strip()
            current_header3 = None
            current_header4 = None

        # Handle Header 3 (like "3.2 Encoder")
        elif header3_text:
            if content_buffer:
                structured_data.append({
                    "page_content": " ".join(content_buffer).strip(),
                    "metadata": {
                        "header1": current_header1,
                        "header2": current_header2,
                        "header3": current_header3,
                        "header4": current_header4,
                        "document_name": document_name,
                        "document_id": document_id,
                       
                    }
                })
                content_buffer = []

            current_header3 = header3_text.strip()
            current_header4 = None

        # Handle Header 4 (like "3.2.2 Multi-Head Attention")
        elif header4_text:
            if content_buffer:
                structured_data.append({
                    "page_content": " ".join(content_buffer).strip(),
                    "metadata": {
                        "header1": current_header1,
                        "header2": current_header2,
                        "header3": current_header3,
                        "header4": current_header4,
                        "document_name": document_name,
                        "document_id": document_id,
                       
                    }
                })
                content_buffer = []

            current_header4 = header4_text.strip()

        # Handle Content
        elif content_text:
            content_buffer.append(content_text.strip())

    if content_buffer:
        page_content = " ".join(content_buffer).strip()
        structured_data.append({
            "page_content": page_content,
            "metadata": {
                "header1": current_header1,
                "header2": current_header2,
                "header3": current_header3,
                "header4": current_header4,
                "document_name": document_name,
                "document_id": document_id,
            }
        })

    return structured_data

def load_document_update(file_path: str, document_name: str, document_id: str):
    """Load the PDF using fitz and extract structured content from XHTML."""
    pdf_document = load_pdf(file_path)
    html_content = extract_html_from_pdf(pdf_document)
    pdf_document.close()

    #document_id = "1234-5678-uuid"
    #document_name = "attention_is_all_you_need.pdf"

    structured_data = extract_headers_and_content_from_html(html_content, document_name, document_id)
    return structured_data
    
if __name__ == "__main__": 
    url_pdf_path = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf' 
    document_name = "attention is all your need.pdf"
    print("\nExtracting from PDF update...")
    extracted_data = load_document_update(url_pdf_path, document_name)
    
    if not extracted_data:
        print("\nNo sections were extracted.")
    else:
        for i, data in enumerate(extracted_data[:10]):  
            print(f"\n--- Section {i + 1} ---")
            print(f"Header 1: {data['metadata']['header1']}")
            print(f"Header 2: {data['metadata']['header2']}")
            print(f"Header 3: {data['metadata']['header3']}")
            print(f"Header 4: {data['metadata']['header4']}")
            print(f"Document Name: {data['metadata']['document_name']}")
            print(f"Document ID: {data['metadata']['document_id']}")
            print(f"Content: {data['page_content'][:300]}...\n")




