import os
import re
import requests
import fitz 

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

def main(key: str, path_local: str, path_url: str):
    """_summary_
    function main to read the text from pdf document
    Args:
        key (_type_): _description_
        path_local (_type_): _description_
        path_url (_type_): _description_
    """
    if key=='local':
        pdf_from_local = load_pdf(path_local)
        print(pdf_from_local)
        if pdf_from_local:
            full_text = read_pdf_content(pdf_from_local)
            print(full_text)
            pdf_from_local.close()
    else:
        pdf_from_url = load_pdf(path_url)
        print(pdf_from_url)
        if pdf_from_url:
            full_text = read_pdf_content(pdf_from_url)
            print(full_text)
            pdf_from_url.close()
        

if __name__ == "__main__": 
    local_pdf_path = r'C:\Users\fawwaz\Downloads\Resume_Fawwaz Atha Rohmatullah_Nov.pdf'
    url_pdf_path = 'https://media.neliti.com/media/publications/249244-none-837c3dfb.pdf'  
    main('local', local_pdf_path, url_pdf_path)