
<img width="400" height="400" alt="logo" src="https://github.com/user-attachments/assets/3d3314af-74e6-49b0-8a7f-1eefdca75feb" />



# Z-Image Controller WEBUI + API Backend

A lightweight local web UI for running the **Z-Image Turbo** pipeline
(Tongyi-MAI / Diffusers).\
Includes automatic dependency installation, a modern Tailwind UI, and a
simple Flask backend.

## Features

-   Zero-setup frontend (HTML + Tailwind + Lucide icons)
-   Flask backend that auto-installs needed Python packages
-   Health-check polling and connection status display
-   Real-time image generation with custom dimensions, seed, steps, and
    guidance scale
-   Downloadable output image
-   Works with CPU or CUDA automatically


<img width="900" height="800" alt="demo" src="https://github.com/user-attachments/assets/7e2f74e4-d41d-4dd1-97fb-a030b60e6472" />



## Requirements

-   **Python 3.10**
-   **Windows or Linux**
-   **NVIDIA GPU optional**, CUDA 12.1 recommended

## Important note for CUDA users (3090 Ti)

All dependencies are auto-installed **except one**.

**You MUST manually install torch with the CUDA 12.1 wheel first:**

    pip install torch --index-url https://download.pytorch.org/whl/cu121

**Works fine for 3090 Ti + Python 3.10.**

Everything else (Flask, CORS, Diffusers, Accelerate, Transformers,
etc.)\
is downloaded and installed **automatically** when you run `app.py`.

## Project Structure

    /
    ├── app.py
    ├── index.html
    ├── index.css
    └── static/

## Setup & Running the Server

### 1. Install Torch (CUDA users only)

    pip install torch --index-url https://download.pytorch.org/whl/cu121

### 2. Run the backend

    python app.py

Starts server at:

    http://127.0.0.1:5000

### 3. Open frontend

Open `index.html` in browser.

## API Endpoints

### GET /health

Health check route.

### POST /generate

Body:

``` json
{
  "prompt": "A red dragon flying through neon fog",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "guidance": 0.0,
  "seed": 42
}
```

Returns base64 PNG.

## Notes

-   Flash attention auto-enabled if available
-   CPU mode supported but slow
-   Model loads once at startup
