from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib

class SymptomInput(BaseModel):
    symptoms: str

class DiseasePredictor:
    def __init__(self):
        try:
            # Load model, tokenizer, and label encoder
            self.model = AutoModelForSequenceClassification.from_pretrained("model")
            self.tokenizer = AutoTokenizer.from_pretrained("tokenizer")
            self.label_encoder = joblib.load("label_encoder.pkl")

            # Device setup (GPU if available)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise RuntimeError("Failed to load model, tokenizer, or label encoder.")

    def predict(self, symptoms: str):
        """Make a disease prediction based on input symptoms."""
        try:
            if not symptoms.strip():
                return "Error: Symptoms cannot be empty."

            # Encode the input symptoms
            encoded_input = self.tokenizer.encode_plus(
                symptoms,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            # Move input data to the correct device
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            predicted_index = torch.argmax(logits, dim=1).cpu().numpy()[0]
            predicted_disease = self.label_encoder.inverse_transform([predicted_index])[0]

            return predicted_disease

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return "Error: Could not generate prediction."

# Initialize FastAPI app and disease predictor
app = FastAPI()
predictor = DiseasePredictor()

@app.post("/predict")
def predict_disease(data: SymptomInput):
    """API endpoint to predict disease from symptoms."""
    if not data.symptoms.strip():
        raise HTTPException(status_code=400, detail="Invalid input. Symptoms cannot be empty.")

    predicted_disease = predictor.predict(data.symptoms)

    if "Error" in predicted_disease:
        raise HTTPException(status_code=500, detail=predicted_disease)

    return {"disease": predicted_disease}

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Disease Prediction API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
