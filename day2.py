import re
import spacy
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from split_section import split_text_into_sections
from load_cloud import main
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import logging
from concurrent.futures import ThreadPoolExecutor
import os
from flask import jsonify
from load_pdf import load_document
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nlp = spacy.load('en_core_web_sm')

def get_text() -> str:
    """Load the text from the cloud or PDF."""
    try:
        logging.info('Starting function: get_text')
        text = main()
        logging.info('Finished function: get_text')
        return text
    except Exception as e:
        logging.exception('Error in get_text')
        raise
        
def compute_similarity(query: str, processed_data: Dict[str, Dict[str, any]], glove_embeddings: Dict[str, np.ndarray], top_k: int = 3) -> None:
    """Compute similarity for each section using GloVe, TF-IDF, or Word2Vec embeddings."""
    try:
        logging.info('Starting function: compute_similarity')
        query_tokens = [token.text for token in nlp(query.lower())]
        token_vectors = [glove_embeddings[token] for token in query_tokens if token in glove_embeddings]
        query_vector = np.mean(token_vectors, axis=0).reshape(1, -1) if token_vectors else np.zeros(50).reshape(1, -1)

        similarity_scores = []

        for section_name, section_data in processed_data.items():
            section_vector = section_data['glove_embedding'].reshape(1, -1)
            similarity_score = cosine_similarity(section_vector, query_vector)[0][0]
            similarity_scores.append((section_name, similarity_score))
        
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        logging.info('Finished function: compute_similarity')
        print(f"\nTop {top_k} Embedding for Query: '{query}'\n")
        for i, (section_name, score) in enumerate(sorted_scores[:top_k]):
            print(f"Rank {i+1} - Section: {section_name}")
            print(f"  Similarity: {score:.4f}")
            print("-" * 40)
    except Exception as e:
        logging.exception('Error in compute_similarity')
        raise
    
def compute_similarity_return(query: str, processed_data: Dict[str, Dict[str, any]], glove_embeddings: Dict[str, np.ndarray], top_k: int = 3) -> List[Dict[str, any]]:
    """Compute similarity for each section using GloVe, TF-IDF, or Word2Vec embeddings."""
    try:
        logging.info('Starting function: compute_similarity')
        
        # Tokenize the query and get its GloVe embedding
        query_tokens = [token.text for token in nlp(query.lower())]
        token_vectors = [glove_embeddings[token] for token in query_tokens if token in glove_embeddings]
        query_vector = np.mean(token_vectors, axis=0).reshape(1, -1) if token_vectors else np.zeros(50).reshape(1, -1)

        similarity_scores = []

        for section_name, section_data in processed_data.items():
            if 'glove_embedding' in section_data:
                section_vector = section_data['glove_embedding'].reshape(1, -1)
                similarity_score = cosine_similarity(section_vector, query_vector)[0][0]
                similarity_scores.append({
                    "section": section_name,
                    "similarity_score": round(float(similarity_score), 4)
                })
        
        sorted_scores = sorted(similarity_scores, key=lambda x: x['similarity_score'], reverse=True)
        
        logging.info('Finished function: compute_similarity')
        
        print(f"\nTop {top_k} Embedding for Query: '{query}'\n")
        for i, result in enumerate(sorted_scores[:top_k]):
            print(f"Rank {i+1} - Section: {result['section']}")
            print(f"  Similarity: {result['similarity_score']:.4f}")
            print("-" * 40)

        return sorted_scores[:top_k]

    except Exception as e:
        logging.exception('Error in compute_similarity')
        raise


def load_glove(glove_path: str) -> Dict[str, np.ndarray]:
    """Load GloVe embeddings from a file and return as a dictionary."""
    try:
        logging.info("Start function: load_glove")
        embeddings_dict = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = vector
        logging.info("finished function: load_glove")
        return embeddings_dict
    except Exception as e:
        logging.exception('Error in load_glove')
        raise

def clean_section(text: str) -> str:
    """Clean and normalize the text of a section."""
    try:
        logging.info("Start function: clean_section")
        patterns = {
            'remove_emails': re.compile(r'\S+@\S+'),      # Remove email addresses 
            'remove_extra_symbols': re.compile(r'[|*•]'),   # Remove symbols like '|', '*', '•'
            'remove_special_chars': re.compile(r'[^a-zA-Z0-9\s%/.-]'),   # Keep necessary symbols: %, /, ., -
            'remove_parentheses_for_dates': re.compile(r'\((\d{2}/\d{4})\)'),   # Remove parentheses for dates "(06/2024)" -> "06/2024"
            'remove_extra_spaces': re.compile(r'\s+'),  # Replace multiple spaces with a single space
        }
        
        text = text.lower()
        text = patterns['remove_emails'].sub(' ', text)  
        text = patterns['remove_extra_symbols'].sub(' ', text)  
        text = patterns['remove_parentheses_for_dates'].sub(r'\1', text)  
        text = patterns['remove_special_chars'].sub(' ', text)  
        text = patterns['remove_extra_spaces'].sub(' ', text)  
        logging.info("finished function: clean_section")
        return text.strip()
    except Exception as e:
        logging.exception('Error in clean_section')
        raise

def spacy_tokenizer(text: str) -> Dict[str, int]:
    """Tokenize the text using spaCy and count token frequencies."""
    try:
        logging.info("Start function spacy_tokenizer")
        doc = nlp(text)
        clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token not in STOP_WORDS]
        logging.info("finished function: spacy_tokenizer")
        return clean_tokens
    except Exception as e:
        logging.error("Error in function spacy_tokenizer")
        raise
    
def preprocess_sections(section_dict: Dict[str, str]) -> Dict[str, Dict[str, any]]:
    """Preprocess all sections to compute NLTK tokens and spaCy tokens."""
    try:
        logging.info("Start function preprocess section")
        processed_data = {}

        for section_name, content in section_dict.items():
            # clean each section
            cleaned_text = clean_section(content)
            # SpaCy
            spacy_tokens = spacy_tokenizer(cleaned_text)
            # store in dictionary
            processed_data[section_name] = {
                'content': cleaned_text,
                'spacy_tokens_len': len(spacy_tokens),
                'spacy_tokens': spacy_tokens
            }
        logging.info("finished function: preprocess sections")
        return processed_data
    except Exception as e:
        logging.error("Error in function preprocess section")
        raise

def tfidf(processed_data: Dict[str, Dict[str, any]]) -> Dict[str, Dict[str, any]]:
    """_summary_
    Add TF-IDF and Word2Vec embeddings to each section.
    Args:
        processed_data (Dict[str, Dict[str, any]]): _description_

    Returns:
        Dict[str, Dict[str, any]]: _description_
    """
    try:
        logging.info("Start function tfidf")
        # Compute TF-IDF embeddings
        vectorizer = TfidfVectorizer()
        all_texts = [section['content'] for section in processed_data.values()]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        for i, section_name in enumerate(processed_data):
            processed_data[section_name]['tfidf_embedding'] = tfidf_matrix[i].toarray()[0]
        logging.info("finished function: tfidf vectorizer")
        return processed_data
    except Exception as e:
        logging.error("Error in function tfidf")

def word2vect(processed_data):
    """_summary_

    Args:
        processed_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        all_tokens = [section['spacy_tokens'] for section in processed_data.values()]
        word2vec_model = Word2Vec(all_tokens, vector_size=100, window=5, min_count=1, workers=4)
        
        for section_name, section_data in processed_data.items():
            token_vectors = [word2vec_model.wv[token] for token in section_data['spacy_tokens'] if token in word2vec_model.wv]
            if token_vectors:
                section_vector = np.mean(token_vectors, axis=0)
                processed_data[section_name]['word2vec_embedding'] = section_vector
            else:
                processed_data[section_name]['word2vec_embedding'] = np.zeros(100)
        logging.info("finished function: word2vec embedding")
        return processed_data
    except Exception as e:
        logging.error("Error in function word2vect")
        raise 
    
def add_glove_embeddings(processed_data: Dict[str, Dict[str, any]], glove_embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict[str, any]]:
    """Add GloVe embeddings to each section using NLTK tokens."""
    try:
        logging.info("Start function add glove embeddings")
        for section_name, section_data in processed_data.items():
            token_vectors = [glove_embeddings[token] for token in section_data['spacy_tokens'] if token in glove_embeddings]
            if token_vectors:
                section_vector = np.mean(token_vectors, axis=0)
                processed_data[section_name]['glove_embedding'] = section_vector
            else:
                processed_data[section_name]['glove_embedding'] = np.zeros(50)  # GloVe is 50D
        logging.info("finished function: add glove embeddings")
        return processed_data
    except Exception as e:
        logging.error("Error in function add glove embedding")
        
def procces_similarity(query, text_data):
    executor = ThreadPoolExecutor(max_workers=4)
    embedding = os.getenv("glove_embedding")
    if not os.path.exists(embedding):
        logging.error(f"GloVe embeddings file '{embedding}' does not exist.")
        return jsonify({"error": "GloVe embeddings file not found"}), 500
        
    glove_embeddings = load_glove(embedding)
    processed_data = {section_name: {'spacy_tokens': tokens} for section_name, tokens in text_data.items()}
    processed_data_with_embeddings = add_glove_embeddings(processed_data, glove_embeddings)

    future = executor.submit(compute_similarity_return, query, processed_data_with_embeddings, glove_embeddings, 3)
    similarity_scores = future.result()
    return similarity_scores

def process_pdf(file_url):
    text = load_document(file_url)
    headers = [ 
            'Summary', 
            'Experience', 
            'Education', 
            'Licenses & Certifications', 
            'Licenses & Certiﬁcations',
            'Skills', 
        ]
    text_split = split_text_into_sections(text, headers)
    text_preprocessed = preprocess_sections(text_split)
    spacyload = {section_name: content["spacy_tokens"] for section_name, content in text_preprocessed.items()}
    return spacyload


    
if __name__ == "__main__":
    # get file from cloud
    text = get_text()
    # split text based on section
    section_dict = split_text_into_sections(text)
    # execute cleaning, and tokenization
    processed_data = preprocess_sections(section_dict)
    # TF-IDF embeddings
    processed_data = tfidf(processed_data)
    # word2vec train
    processed_data = word2vect(processed_data)
    # Glove pre - trained model
    glove_embeddings = load_glove('glove.6B.50d.txt')
    processed_data = add_glove_embeddings(processed_data, glove_embeddings)
    # Display each section
    # for section_name, section_data in processed_data.items():
    #     print(f"\n\n--- {section_name.upper()} ---\n")
    #     print(f"Content Preview: {section_data['content'][:200]} \n")
    #     print(f"Total spaCy Tokens: {section_data['spacy_tokens_len']}")
    #     print(f"TF-IDF Embedding (first 5 values): {section_data['tfidf_embedding'][:5]}")
    #     print(f"Word2Vec Embedding (first 5 values): {section_data['word2vec_embedding'][:5]}")
    #     print(f"GloVe Embedding (first 5 values): {section_data['glove_embedding'][:5]}")

    # query = 'ant cat dog'
    # compute_similarity(query, processed_data, glove_embeddings, top_k=3)
    
    url_pdf_path = 'https://api.features-v2.zetta-demo.space/fileuploads/AI-Intern---Glints---Josephine-Diva-0c38af74-db36-4436-b290-6f28e56de774.pdf'
    spacyload = process_pdf(url_pdf_path)
    print(spacyload)
    logging.info("finished main script")

   