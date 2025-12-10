import subprocess
import sys
import importlib
import os
from flask import render_template
pipe.enable_model_cpu_offload() 
# --- Dependency Management ---
# Automatically install packages if they are missing
packages = {
    "flask": "flask",
    "flask_cors": "flask-cors",
    "diffusers": "git+https://github.com/huggingface/diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "google.protobuf": "protobuf",
    "sentencepiece": "sentencepiece"
}

def install_packages():
    print("--- Checking and Installing Dependencies ---")
    for module, package in packages.items():
        try:
            importlib.import_module(module)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("--- Dependencies Ready ---")

try:
    install_packages()
except Exception as e:
    print(f"Error installing packages: {e}")
    sys.exit(1)

# --- Application ---
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from diffusers import ZImagePipeline
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


print(f"\n[INIT] Loading Z-Image Pipeline on {DEVICE}...")

try:
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
    )
    pipe.to(DEVICE)
    if DEVICE == "cuda":
        try:
            pipe.transformer.set_attention_backend("flash")
        except:
            pass
    print("[INIT] Pipeline Loaded Successfully.\n")
except Exception as e:
    print(f"[ERROR] Failed to load pipeline: {e}")
    pipe = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "loaded": pipe is not None})

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate():
    if not pipe:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.json
    print(f"[REQ] Generating: {data.get('prompt')[:50]}...")
    
    try:
        generator = torch.Generator(DEVICE).manual_seed(int(data.get("seed", 42)))
        image = pipe(
            prompt=data.get("prompt"),
            height=int(data.get("height", 1024)),
            width=int(data.get("width", 1024)),
            num_inference_steps=int(data.get("steps", 9)),
            guidance_scale=float(data.get("guidance", 0.0)),
            generator=generator
        ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({"image": f"data:image/png;base64,{img_str}"})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Server running on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)