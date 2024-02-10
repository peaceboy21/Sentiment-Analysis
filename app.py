from flask import Flask, request, jsonify, render_template
from sentimentanalysis.pipeline.prediction import PredictionPipeline

app = Flask(__name__, template_folder='template')

pipeline = PredictionPipeline()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
    
@app.route("/predict", methods=["POST"])
def predict():
    text_input = request.json["text"]
    result = pipeline.sentiment_analysis(text_input)
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
