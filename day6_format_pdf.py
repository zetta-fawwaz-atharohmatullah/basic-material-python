# ********** IMPORT FRAMEWORKS **************
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
# ********** IMPORT LIBRARY **************
from typing import Dict, List
import os
import re
import requests
import json
import fitz 
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from load_pdf import load_pdf
import logging

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_model(model_name: str) -> ChatOpenAI:
        key = os.getenv("OPENAI_API_KEY")
        model = ChatOpenAI(
            api_key=key,
            model_name=model_name,
            temperature=0.9,
            max_tokens=256
        )
        return model
    
llm = get_model("gpt-4o-mini")

def extract_html_from_pdf(pdf_document: fitz.Document) -> str:
    
    """_summary_
    Extract the full HTML content from the PDF using PyMuPDF (fitz)
    Args:
        source (str): _pdf doc 

    Returns:
        _type_: _html format 
    """
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

def clean_html_chunk(html_doc: str) -> str:
    soup = BeautifulSoup(html_doc, "html.parser")
    return soup.get_text()

# Define prompts
header_prompt = PromptTemplate.from_template("""
        Analyze this HTML text and determine if it contains a header or title.
        Headers can be in any format including HTML tags like h1-h6 or other formatting like  
        <p><b>number or even only h3 with number.

         For section numbers:
            - Single number (1, 2): Level 2
            - Two numbers (1.1, 5.4): Level 3
            - Three numbers (3.2.1): Level 4
    
        Text: {text}

        If this is a header, respond in format: HEADER|level(only the number!)|text
        Expected output example = 'HEADER', '2', 'Background' or NOT_HEADER dont add other values besides these in the output!

        If not a header, respond: NOT_HEADER
        """)

content_prompt = PromptTemplate.from_template("""
        Determine if the following text content belongs to the header section "{header}".
        Consider the context and relevance of the content.

        Content: {text}

        Return only YES or NO.
        """)

# Define chains
header_chain = header_prompt | llm
content_chain = content_prompt | llm

def format_document_section(content: str, headers: Dict, metadata: Dict) -> Dict:
    cleaned_content = clean_html_text(content)
    return {
        "page_content": cleaned_content,
        "metadata": {
            "header1": headers.get("header1", ""),
            "header2": headers.get("header2", ""),
            "header3": headers.get("header3", ""),
            "header4": headers.get("header4", ""),
            "document_name": metadata["document_name"],
            "document_id": metadata["document_id"]
        }
    }

def split_content(html_content: str, chunk_size: int = 500) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        # separators=["</div>", "</p>", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=200,
        # length_function=len,
        # is_separator_regex=False
    )
    return splitter.split_text(html_content)

def clean_html_text(text: str) -> str:
    """Clean HTML tags and normalize whitespace"""
    # First use BeautifulSoup to handle HTML entities and complex tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def process_document(pdf_path: str, document_metadata: Dict, chunk_size: int = 1000) -> List[Dict]:
    pdf = load_pdf(pdf_path)
    html_content = extract_html_from_pdf(pdf)
    chunks = split_content(html_content, chunk_size=chunk_size)
    
    current_headers = {}
    content_buffer = []
    formatted_sections = []
    found_title = False
    last_chunk_ended_with_header = False
    
    for i, chunk in enumerate(chunks[:10]):
        #clean_chunk = clean_html_chunk(chunk)
        header_result = header_chain.invoke({"text": chunk})

        chunk_info = {"type": "content", "text": chunk}
        print(f"\nProcessing chunk {i+1}/{len(chunks)}:")
        print(f"Chunk content: {chunk}")
        print(f"\nheader invoke: {header_result.content}")
        
        # Check if chunk starts with a header
        chunk_has_header = "NOT_HEADER" not in header_result.content
        
        if chunk_has_header:
            _, level, header_text = header_result.content.split("|")
            level = int(level)
            #header_text = clean_html_text(header_text)
            
            if not found_title and "abstract" not in header_text.lower():
                level = 1
                found_title = True
            elif header_text.strip()[0].isdigit():
                dots = header_text.split()[0].count('.')
                level = dots + 2
            
            # If previous chunk didn't end with header, this content belongs to previous section
            if not last_chunk_ended_with_header:
                # Split chunk at header position
                header_position = chunk.find(header_text)
                if header_position > 0:
                    # Add text before header to previous section
                    content_buffer.append(chunk[:header_position])
                    
            # Save previous section if exists
            if content_buffer:
                formatted_sections.append(format_document_section(
                    "".join(content_buffer),
                    current_headers,
                    document_metadata
                ))
                content_buffer = []
            
            # Update headers
            for key in list(current_headers.keys()):
                if int(key.replace("header", "")) >= level:
                    current_headers.pop(key)
            current_headers[f"header{level}"] = header_text.strip()
            
            # Add remaining content after header
            header_position = chunk.find(header_text)
            remaining_content = chunk[header_position + len(header_text):].strip()
            if remaining_content:
                content_buffer.append(remaining_content)
                
            last_chunk_ended_with_header = False
        else:
            # No header in chunk, add entire content to buffer
            content_buffer.append(chunk_info["text"])
            last_chunk_ended_with_header = False
    
    # Save final section
    if content_buffer:
        formatted_sections.append(format_document_section(
            "".join(content_buffer),
            current_headers,
            document_metadata
        ))
    formatted_sections = [
        section for section in formatted_sections 
        if section["page_content"].strip()
    ]
    return formatted_sections


if __name__ == "__main__":
    url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
    document_metadata = {
        "document_name": "attention.pdf",
        "document_id": "12345"
    }
    
    results = process_document(url, document_metadata)
    output_file_path = "output.json"

    for i in results[:5]:
        print(i)
    
    # Save to JSON file
    # with open(output_file_path, "w", encoding="utf-8") as json_file:
    #     json.dump(results, json_file, ensure_ascii=False, indent=4)

