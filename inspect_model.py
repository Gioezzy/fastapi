
import torch
import os
import sys

MODEL_PATH = "model.pth/songket_model.pth.zip"

def inspect():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: File {MODEL_PATH} not found.")
        return

    try:
        print(f"Loading {MODEL_PATH}...")
        # Load without map_location first to see structure, or map to cpu
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        if isinstance(checkpoint, dict):
            print("\n✅ File is a Dictionary (Checkpoint).")
            print("Found keys:", list(checkpoint.keys()))
            
            # Check for common class keys
            possible_keys = ['classes', 'class_names', 'labels', 'idx_to_class', 'categories']
            found_classes = None
            
            for key in possible_keys:
                if key in checkpoint:
                    found_classes = checkpoint[key]
                    print(f"\n🎉 FOUND CLASSES under key '{key}':")
                    print(found_classes)
                    break
            
            if not found_classes:
                print("\n❌ No obvious class metadata found in the dictionary keys.")
                print("This means the file only contains mathematical weights, not names.")
                print("Current keys in state_dict (sample):")
                if 'state_dict' in checkpoint:
                   print(list(checkpoint['state_dict'].keys())[:5])
                else:
                   print(list(checkpoint.keys())[:5])
        else:
            print("\n⚠️ File is a full Model Object (not a dict).")
            if hasattr(checkpoint, 'classes'):
                 print(f"🎉 Found 'classes' attribute: {checkpoint.classes}")
            else:
                 print("Object does not have 'classes' attribute.")

    except Exception as e:
        print(f"Error inspecting file: {e}")

if __name__ == "__main__":
    inspect()
