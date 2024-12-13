from flask import jsonify, Flask, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# **********  SWAGGER documentation
@app.route('/documentation/<path:path>')
def send_static(path):
    return send_from_directory('documentation', path)

SWAGGER_URL = '/swagger'
API_URL = '/documentation/sample.yaml'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name' : 'FLASK API '
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route("/", methods=[ 'GET'])
def home():
    return jsonify({"message": "FLASK API RUNNING"}), 200

if __name__ == '__main__':
    app.run(debug=True)
