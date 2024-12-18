import os
import re
import requests
import fitz 
#from day2 import split_text_into_sections,  preprocess_sections, spacy_tokenizer

# checkking url
def is_url(path: str) -> bool:
    """_summary_
    find https in the source path
    Args:
        path (str): _description_path

    Returns:
        bool: _description_bool
    """
    url_pattern = re.compile(r'^(http|https)://')
    return bool(url_pattern.match(path))

# load document
def load_pdf(source: str):
    """_summary_
    checking type of source and load it
    Args:
        source (str): _description_path 

    Returns:
        _type_: _description_path 
    """
    pdf_document = None
    
    if is_url(source):
        try:
            response = requests.get(source, timeout=10)  
            response.raise_for_status() 
            pdf_document = fitz.open(stream=response.content, filetype="pdf")  
            print(f"Successfully loaded PDF from URL: {source}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to load PDF from URL. Error: {e}")
        except Exception as e:
            print(f"Error while loading PDF from URL. Error: {e}")
            
    elif os.path.isfile(source):
        try:
            pdf_document = fitz.open(source)
            print(f"Successfully loaded PDF from local file path: {source}")
        except Exception as e:
            print(f"Failed to load PDF from local path. Error: {e}")
            
    else:
        print(f"Invalid source: {source}. Please provide a valid URL or local file path.")

    return pdf_document

# read document
def read_pdf_content(pdf_document: str):
    """_summary_
    Extract and returns the text content from pdf document
    Args:
        pdf_document (_type_): _description_
        
    Returns:
        _type_: _description_
    
    detect html type, (get_text())  
    pdf miner font size     
    """
    try:
        if not pdf_document:
            raise ValueError ("Provided object is not iterable like PDF document")
        
        #dictionary = {}
        full_text = ""
        for page in pdf_document:
            page_text = page.get_text()
            full_text += page_text + "\n" 
        
        # for page_number in range(len(pdf_document)):
        #     page = pdf_document.load_page(page_number)  
        #     page_text = page.get_text()  
        #     full_text += page_text + "\n"  
        #     dictionary[page_number] = full_text
        print("Successfully extracted text from PDF.")
        
    except (ValueError, TypeError) as ve:
        print(f"{ve} not a valid pdf")
    return full_text

def extract_text_from_pdf(pdf_document: fitz.Document, start_page: int = 0, end_page: int = None) -> str:
    """Extract and clean the text content from a PDF document within a page range.
    
    Args:
        pdf_document (fitz.Document): A PyMuPDF Document object.
        start_page (int): The starting page number (0-indexed).
        end_page (int): The ending page number (0-indexed, exclusive). If None, it extracts until the last page.
    
    Returns:
        str: Extracted and cleaned text from the specified range of the PDF document.
    """
    if not pdf_document:
        raise ValueError("Provided object is not a valid PDF document.")
    
    if end_page is None or end_page > len(pdf_document):
        end_page = len(pdf_document)
    
    full_text = ""
    try:
        for page_number in range(start_page, end_page):
            page = pdf_document.load_page(page_number)
            page_text = page.get_text()
            full_text += page_text + "\n"
        print(f"Successfully extracted text from pages {start_page} to {end_page - 1}.")
    except Exception as e:
        print(f"Error while extracting text from PDF. Error: {e}")
    
    return full_text


def load_document(path: str):
    """_summary_
    function main to read the text from pdf document
    Args:
        path: _description_
    """
    
    pdf_from_local = load_pdf(path)
    print(pdf_from_local)
    if pdf_from_local:
        full_text = read_pdf_content(pdf_from_local)
        # print(full_text)
        # headers = [ 
        #     'Summary', 
        #     'Experience', 
        #     'Education', 
        #     'Licenses & Certifications', 
        #     'Licenses & Certiﬁcations',
        #     'Skills', 
        # ]
        # text_split = split_text_into_sections(full_text, headers)
        # text_preprocessed = preprocess_sections(text_split)
        # print(text_preprocessed)
        pdf_from_local.close()
    return full_text



if __name__ == "__main__": 
    local_pdf_path = r'C:\Users\fawwaz\Downloads\Resume_Fawwaz Atha Rohmatullah_Nov.pdf'
    url_pdf_path = 'https://api.features-v2.zetta-demo.space/fileuploads/AI-Intern---Glints---Josephine-Diva-0c38af74-db36-4436-b290-6f28e56de774.pdf?'  
    texts = load_document(url_pdf_path)
    print(texts)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    