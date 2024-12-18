import os
import re
import requests
import fitz  # PyMuPDF


# def is_url(path: str) -> bool:
#     """Check if the given path is a URL.
    
#     Args:
#         path (str): Path or URL to check.
    
#     Returns:
#         bool: True if the path is a URL, False otherwise.
#     """
#     url_pattern = re.compile(r'^(http|https)://')
#     return bool(url_pattern.match(path))


# def load_pdf(source: str) -> fitz.Document:
#     """Load a PDF document from a local file or URL.
    
#     Args:
#         source (str): Path to a local PDF file or URL to the PDF file.
    
#     Returns:
#         fitz.Document: A PyMuPDF Document object if successful, None otherwise.
#     """
#     pdf_document = None
    
#     try:
#         if is_url(source):
#             response = requests.get(source, timeout=10)  
#             response.raise_for_status()  
#             pdf_document = fitz.open(stream=response.content, filetype="pdf")  
#             print(f"Successfully loaded PDF from URL: {source}")
#         elif os.path.isfile(source):
#             pdf_document = fitz.open(source)
#             print(f"Successfully loaded PDF from local file path: {source}")
#         else:
#             raise ValueError(f"Invalid source: {source}. Please provide a valid URL or local file path.")
#     except requests.exceptions.RequestException as e:
#         print(f"Failed to load PDF from URL. Error: {e}")
#     except Exception as e:
#         print(f"Error while loading PDF. Error: {e}")
    
#     return pdf_document


# def extract_text_from_pdf(pdf_document: fitz.Document) -> str:
#     """Extract and clean the text content from a PDF document.
    
#     Args:
#         pdf_document (fitz.Document): A PyMuPDF Document object.
    
#     Returns:
#         str: Extracted and cleaned text from the PDF document.
#     """
#     if not pdf_document:
#         raise ValueError("Provided object is not a valid PDF document.")
    
#     full_text = ""
#     try:
#         for page in pdf_document:
#             page_text = page.get_text()
#             full_text += page_text + "\n"
#         print("Successfully extracted text from PDF.")
#     except Exception as e:
#         print(f"Error while extracting text from PDF. Error: {e}")
    
#     return clean_text(full_text)


# def clean_text(text: str) -> str:
#     """Clean up extracted text for RAG purposes by removing extra whitespace, newlines, special characters, and common page elements like headers and footers.
    
#     Args:
#         text (str): Raw text extracted from the PDF.
    
#     Returns:
#         str: Cleaned and normalized text.
#     """
#     # Remove page numbers and repeated headers/footers (common for PDFs)
#     text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Remove standalone page numbers
    
#     # Remove multiple newlines and extra whitespace
#     text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a single space
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
#     # Remove non-alphanumeric characters except basic punctuation
#     text = re.sub(r'[^\w\s.,]', '', text)  # Remove non-alphanumeric characters (except ., and spaces)
    
#     # Remove common unwanted sections if necessary (like 'References', 'Table of Contents')
#     text = re.sub(r'References.*', '', text, flags=re.IGNORECASE)  # Remove References section
#     text = re.sub(r'Table of Contents.*', '', text, flags=re.IGNORECASE)  # Remove Table of Contents section
    
#     return text.strip()


# def load_document(path: str) -> str:
#     """Main function to load and extract text from a PDF document.
    
#     Args:
#         path (str): URL or file path to the PDF document.
    
#     Returns:
#         str: Cleaned and extracted text from the PDF document.
#     """
#     pdf_document = load_pdf(path)
    
#     if pdf_document:
#         full_text = extract_text_from_pdf(pdf_document)
#         pdf_document.close()
#         return full_text
    
#     return ""


# if __name__ == "__main__": 
#     url_pdf_path = 'https://sherlock-holm.es/stories/pdf/a4/2-sided/fina.pdf'  
    
#     print("\nExtracting from url PDF...")
#     local_text = load_document(url_pdf_path)
#     print(f"Extracted text from url PDF:\n{local_text}\n")



#******************************
import os
import re
import requests
import fitz  
import uuid  


def is_url(path: str) -> bool:
    url_pattern = re.compile(r'^(http|https)://')
    return bool(url_pattern.match(path))


def load_pdf(source: str) -> fitz.Document:
    pdf_document = None
    try:
        if is_url(source):
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            print(f"Successfully loaded PDF from URL: {source}")
        elif os.path.isfile(source):
            pdf_document = fitz.open(source)
            print(f"Successfully loaded PDF from local file path: {source}")
        else:
            raise ValueError(f"Invalid source: {source}.")
    except Exception as e:
        print(f"Error while loading PDF. Error: {e}")
    return pdf_document


def extract_text_from_pdf(pdf_document: fitz.Document) -> list:
    """Extract and analyze the PDF content with font size, position, and font name."""
    full_content = []
    try:
        for page_num, page in enumerate(pdf_document):
            page_dict = page.get_text('dict') 
            blocks = page_dict.get('blocks', [])
            
            for block in blocks:
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        text = span.get('text', '').strip()
                        font_size = span.get('size', 0)
                        font = span.get('font', '')
                        
                        if text:
                            full_content.append({
                                'page_num': page_num + 1,
                                'text': text,
                                'font_size': font_size,
                                'font': font
                            })
        print("Successfully extracted text from PDF with font information.")
    except Exception as e:
        print(f"Error while extracting text from PDF. Error: {e}")
    
    return full_content


def extract_headers_and_content_from_fonts(content: list):
    font_sizes = [item['font_size'] for item in content]
    unique_font_sizes = sorted(set(font_sizes), reverse=True)  
    print(unique_font_sizes)
    if not unique_font_sizes:
        print("No font sizes found in the content.")
        return []

    largest_font_size = unique_font_sizes[0]
    second_largest_font_size = unique_font_sizes[1] if len(unique_font_sizes) > 1 else largest_font_size
    third_largest_font_size = unique_font_sizes[2] if len(unique_font_sizes) > 2 else second_largest_font_size

    structured_data = []
    current_header1 = None
    current_header2 = None
    current_header3 = None
    content_buffer = []

    for item in content:
        font_size = item['font_size']
        text = item['text']

        if font_size == largest_font_size:
            if content_buffer:
                structured_data.append({
                    "header1": current_header1,
                    "header2": current_header2,
                    "header3": current_header3,
                    "content": " ".join(content_buffer)
                })
                content_buffer = []

            current_header1 = text
            current_header2 = None
            current_header3 = None

        elif font_size == second_largest_font_size:
            if content_buffer:
                structured_data.append({
                    "header1": current_header1,
                    "header2": current_header2,
                    "header3": current_header3,
                    "content": " ".join(content_buffer)
                })
                content_buffer = []

            current_header2 = text
            current_header3 = None

        elif font_size == third_largest_font_size:
            if content_buffer:
                structured_data.append({
                    "header1": current_header1,
                    "header2": current_header2,
                    "header3": current_header3,
                    "content": " ".join(content_buffer)
                })
                content_buffer = []

            current_header3 = text
        else:
            content_buffer.append(text)

    if content_buffer:
        structured_data.append({
            "header1": current_header1,
            "header2": current_header2,
            "header3": current_header3,
            "content": " ".join(content_buffer)
        })

    return structured_data


def load_document(path: str):
    """Load the PDF document and return structured metadata and content."""
    pdf_document = load_pdf(path)
    if pdf_document:
        content = extract_text_from_pdf(pdf_document)
        print(content)
        #print(content)
        pdf_document.close()
        
        structured_data = extract_headers_and_content_from_fonts(content)
        return structured_data
    
    return []


if __name__ == "__main__": 
    url_pdf_path = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
    
    document_id = str(uuid.uuid4())
    document_name = "attention_is_all_you_need.pdf"
    
    extracted_data = load_document(url_pdf_path)
    
    if not extracted_data:
        print("\nNo sections were extracted.")
    else:
        for i, data in enumerate(extracted_data[:10]): 
            print(f"\n--- Section {i + 1} ---")
            print(f"Header 1: {data['header1']}")
            print(f"Header 2: {data['header2']}")
            print(f"Header 3: {data['header3']}")
            print(f"Content: {data['content']}\n")  
