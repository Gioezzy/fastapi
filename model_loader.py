import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import shutil
import tempfile
import zipfile

class ModelLoader:
    def __init__(self, model_path, classes=None):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = classes or [
            "apel", 
            "baragi", 
            "bungoStangkai", 
            "itiakPulangPatang", 
            "pucuakRabung", 
            "rangkiang", 
            "saikGalamai", 
            "taratai", 
            "tulip"
        ]

        self.db_mapping = {
            "apel": "Bungo Apel",
            "baragi": "Baragi",
            "bungotangkai": "Bungo Satangkai",
            "bungostangkai": "Bungo Satangkai",
            "itiakpulangpatang": "Itiak Pulang Patang",
            "pucuakrabung": "Pucuak Rabung",
            "rangkiang": "Rangkiang",
            "saikgalamai": "Saik Galamai",
            "taratai": "Bungo Taratai",
            "tulip": "Bungo Tulip"
        }

    def _zip_directory(self, dir_path, zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dir_path)
                    
                    try:
                        zinfo = zipfile.ZipInfo(arcname)
                        zinfo.date_time = (2025, 1, 1, 12, 0, 0) 
                        
                        with open(file_path, "rb") as f:
                            zipf.writestr(zinfo, f.read())
                    except Exception as e:
                        print(f"Warning: Could not zip {file_path}: {e}")

    def load(self):
        print(f"Loading model from {self.model_path}...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                print("Detected state_dict/checkpoint. Attempting to load into architecture...")
                
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                
                try:
                    from torchvision import models
                    print("Attempting to load as MobileNetV2...")
                    
                    self.model = models.mobilenet_v2(pretrained=False)
                    
                    num_ftrs = self.model.classifier[1].in_features
                    self.model.classifier[1] = nn.Linear(num_ftrs, len(self.classes))
                    
                    self.model.load_state_dict(state_dict)
                    print("Success! Loaded as MobileNetV2.")
                        
                except Exception as e:
                    print(f"Architecture matching failed. Keys in state_dict: {list(state_dict.keys())[:5]}")
                    raise RuntimeError(f"Could not match state_dict to standard architectures (MobileNetV2). Error: {e}")

            else:
                self.model = checkpoint

            self.model.to(self.device).eval()
            print("Model loaded successfully!")
            
            if hasattr(self.model, 'classes'):
                self.classes = self.model.classes
                print(f"Loaded classes from model: {self.classes}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict(self, image_file):
        if not self.model:
            raise RuntimeError("Model not loaded")

        image = Image.open(image_file).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            predicted_class = "Unknown"
            if predicted_idx < len(self.classes):
                predicted_class = self.classes[predicted_idx]
            
            db_name = self.db_mapping.get(predicted_class.lower()) or self.db_mapping.get(predicted_class) or predicted_class

            return {
                "motif": db_name,
                "raw_motif": predicted_class,
                "confidence": float(confidence),
                "all_scores": {self.classes[i]: float(probabilities[i]) for i in range(min(len(self.classes), len(probabilities)))} 
                               if self.classes else {}
            }
