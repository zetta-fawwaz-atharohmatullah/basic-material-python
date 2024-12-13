from flask import Flask, request, jsonify, send_from_directory
import requests
from day2 import procces_similarity, process_pdf
                  
import logging
from detect import LanguageTranslator
from concurrent.futures import ThreadPoolExecutor
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()
#CORS(app)
executor = ThreadPoolExecutor(max_workers=4)

@app.route('/load_doc', methods=['POST'])
def load_doc():
    """ 
    Expects: 
        - JSON body with 'file_url': URL of the PDF file to process
    Returns: 
        - 200: JSON message indicating document processed and sent to webhook
        - 400: JSON error message if the request is invalid
        - 500: JSON error message if an internal error occurs
    """
    try:
        # Extract input data and validate
        data = request.json
        if not isinstance(data, dict) or 'file_url' not in data:
            return jsonify({"error": "Missing 'file_url' in request"}), 400

        file_url = data.get('file_url')

        logging.info(f"Processing Document")
        # Extract text from PDF URL
        try:
            response = requests.get(file_url, stream=True, timeout=10)  # 10-second timeout for file download
            if int(response.headers.get('content-length', 0)) > 10 * 1024 * 1024:  # 10 MB limit
                return jsonify({"error": "File size exceeds the limit"}), 400
            
            text = process_pdf(file_url)  
        except Exception as e:
            logging.exception(f"Failed to extract text from PDF: {file_url}")
            return jsonify({"error": "Failed to extract text from the PDF file"}), 500

        # Asynchronous POST request to the webhook
        webhook_url = os.getenv("WEBHOOK_URL") 
        logging.info(f"Sending processed document to webhook: {webhook_url}")
        try:
            future = executor.submit(requests.post, webhook_url, json={"text": text})
            response = future.result(timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.exception('Failed to send data to the webhook.')
            return jsonify({"error": "Failed to send the processed document to the webhook"}), 500
        except Exception as e:
            logging.exception("An unexpected error occurred while executing the thread.")
            return jsonify({"error": "An unexpected error occurred"}), 500
        
        logging.info("Finished processing document and successfully sent to webhook.")
        return jsonify({"message": "Document processed and sent to webhook"}), 200

    except Exception as e:
        logging.exception('An error occurred in /load_doc endpoint.')
        return jsonify({"error": "Internal server error"}), 500

@app.route('/compute_similarity', methods=['POST'])
def compute_similarity_endpoint():
    """ 
    Expects: 
        - JSON body with 'query': The search query and 'text': list dict with tokens
    Returns: 
        - 200: JSON with query and similarity scores
        - 400: JSON error message if the request is invalid
        - 500: JSON error message if an internal error occurs
    """
    try:
        data = request.json

        if 'query' not in data or 'text' not in data:
            return jsonify({"error": "Missing 'query' or 'text' in request"}), 400

        query = data['query']
        text_data = data['text']
        
        similarity_scores = procces_similarity(query, text_data)

        logging.info("Finished computing similarity scores.")
        return jsonify({
            "query": query,
            "similarity_scores": similarity_scores
        }), 200

    except Exception as e:
        logging.exception('An error occurred in /compute_similarity endpoint.')
        return jsonify({"error": "An internal server error occurred"}), 500

@app.route("/translate", methods=['POST'])
def translate():
    """
    API Endpoint to translate a given text to the specified target language.
    Expects:
    {
        "text": "The text to be translated",
        "target_lang": "es"  // target language (e.g., "es" for Spanish)
    }
    Returns: 
        - 200: JSON with the original text, translation process, and translated text
        - 400: JSON error message if the request is invalid
        - 500: JSON error message if an internal error occurs
    """
    try:
        data = request.json
        text = data.get('text')
        target = data.get('target_lang')
        
        # Validate input
        if 'text' not in data or 'target_lang' not in data:
            return jsonify({"error": "Missing 'text' or 'target_lang' in request"}), 400
        
        translator = LanguageTranslator(num_threads=4)
        full_target, full_detected, translated = translator.process_lang(text, target)
        
        logging.info("Finished translations.")
        return jsonify({
            "original_text": text,
            "Process Language": f"{full_detected} -> {full_target}",
            "translated_text": translated
        }), 200
        
    except Exception as e:
        logging.exception(f'An error occurred while translating the text {e}.')
        return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == "__main__":
    app.run(debug=False)