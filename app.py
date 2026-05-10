from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load(
    "penguin_species_model.pkl"
)

EXPECTED_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island",
    "sex"
]

@app.route("/health", methods=["GET"])
def health():
    
    return jsonify({
        "status": "ok"
    })

@app.route("/predict", methods=["POST"])
def predict():
    
    try:
        
        data = request.get_json()
        
        # JSON yoxdursa
        if data is None:
            return jsonify({
                "error": "No JSON received"
            }), 400
        
        # Missing feature check
        missing = [
            feature
            for feature in EXPECTED_FEATURES
            if feature not in data
        ]
        
        if missing:
            return jsonify({
                "error": f"Missing features: {missing}"
            }), 400
        
        # DataFrame yarat
        input_df = pd.DataFrame([data])
        
        # Prediction
        prediction = model.predict(input_df)[0]
        
        # Probabilities
        probabilities = model.predict_proba(input_df)[0]
        
        class_probs = {
            class_name: float(prob)
            for class_name, prob in zip(
                model.classes_,
                probabilities
            )
        }
        
        return jsonify({
            "prediction": prediction,
            "probabilities": class_probs
        })
    
    except Exception as e:
        
        return jsonify({
            "error": str(e)
        }), 500
    

if __name__ == "__main__":
    
    app.run(debug=True)

