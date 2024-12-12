import re
from typing import List, Dict

def compile_section_pattern(headers: List[str]) -> re.Pattern:
    """
    Compile a regex pattern for all section headers.
    
    Args:
    - headers (List[str]): List of section headers to be matched.
    
    Returns:
    - re.Pattern: Compiled regex pattern for section headers.
    """
    escaped_headers = [re.escape(header) for header in headers]
    pattern = r'(' + r'|'.join(escaped_headers) + r')'
    return re.compile(pattern)


def initialize_section_dict(first_chunk: str, default_section: str = "Introduction") -> Dict[str, str]:
    """
    Initialize the section dictionary by assigning the first chunk of text to a default section.
    
    Args:
    - first_chunk (str): The text before the first section header.
    - default_section (str): The name of the default section.
    
    Returns:
    - Dict[str, str]: Initial section dictionary with the first chunk assigned to the default section.
    """
    section_dict = {}
    if first_chunk.strip():
        section_dict[default_section] = first_chunk.strip()
    return section_dict


def merge_sections(split_sections: List[str], headers: List[str]) -> Dict[str, str]:
    """
    Merge text chunks into a structured dictionary where the keys are section titles.
    
    Args:
    - split_sections (List[str]): List of text chunks split by the section pattern.
    - headers (List[str]): List of valid section headers.
    
    Returns:
    - Dict[str, str]: A dictionary with section names as keys and their corresponding content as values.
    """
    section_dict = initialize_section_dict(split_sections[0])
    current_section = None

    for i in range(1, len(split_sections)):
        chunk = split_sections[i].strip()
        #print(chunk)
        if chunk in headers:
            current_section = chunk
            if current_section not in section_dict:
                section_dict[current_section] = ""
        elif current_section:
            section_dict[current_section] += chunk + "\n"

    return {section: content.strip() for section, content in section_dict.items()}


def split_text_into_sections(text: str, headers: List[str] = None) -> Dict[str, str]:
    """
    Split the text into structured sections based on section titles.
    
    Args:
    - text (str): The entire raw text from the PDF.
    - headers (List[str], optional): List of section headers to split the text. Defaults to common headers.
    
    Returns:
    - Dict[str, str]: A dictionary where keys are section titles and values are the corresponding section content.
    """
    if headers is None:
        headers = [ 
            'Skills', 
            'Experience', 
            'Education', 
            'Projects', 
            'Certifications', 
            'Certiﬁcations',  # Handle "fi" ligatures in PDFs
            'Others'
        ]
    
    # Compile the pattern once for efficiency
    section_pattern = compile_section_pattern(headers)
    #print(section_pattern)
    # Split text into sections
    split_sections = section_pattern.split(text)
    #print(split_sections)
    # Merge split sections into a structured dictionary
    section_dict = merge_sections(split_sections, headers)
    #print(section_dict)   
    return section_dict
