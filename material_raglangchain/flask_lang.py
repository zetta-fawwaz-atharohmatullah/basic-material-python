from flask import Flask, request, jsonify
import logging
from test import qa_system

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/qa_system', methods=['POST'])
def qa_systems():
    try:
        # Extract input data and validate
        data = request.json
        if 'Query' not in data:
            return jsonify({"error": "Missing 'Query' in request"}), 400
        
        query = data.get("Query")
        results = qa_system(query)
        
        # Check the result is structured correctly
        if not all(key in results for key in ("Question", "Answer", "Tokens_in", "Tokens_out")):
            return jsonify({"error": "Unexpected response from qa_system"}), 500

        logging.info("Finished computing similarity scores.")
        return jsonify({
            "question": results.get("question"),
            "answer": results.get("answer"),
            "tokens_in": results.get("tokens_in"),
            "tokens_out": results.get("tokens_out")
        }), 200
        
    except Exception as e:
        logging.exception(f'An error occurred in /qa_system endpoint: {e}')
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
