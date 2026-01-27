from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_loader import ModelLoader
import uvicorn
import os

app = FastAPI(title="Innovillage AI Service", description="Songket Motif Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "./model.pth/songket_model.pth.zip")

loader = ModelLoader(MODEL_PATH)

@app.on_event("startup")
async def startup_event():
    try:
        loader.load()
    except Exception as e:
        print(f"WARNING: Failed to load model at startup: {e}")
        print("Ensure the model path is correct and dependencies are installed.")

@app.get("/")
def read_root():
    return {"status": "online", "service": "Innovillage Songket Recognizer"}

@app.post("/predict")
async def predict_motif(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        result = loader.predict(file.file)
        
        motif_name = result["motif"]
        result["philosophy"] = f"Filosofi untuk {motif_name} belum tersedia."
        
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
