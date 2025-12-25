# Tech Stack

## Backend
- Python 3.10+
- Flask + Flask-CORS (web framework)
- PyTorch (deep learning)
- Pillow, OpenCV, NumPy (image processing)

## Frontend (Legacy)
- HTML5/CSS3/JavaScript (ES6 Modules)
- jQuery
- JSZip, jsPDF

## Frontend (Vue Migration - In Progress)
- Vue 3.5 + TypeScript
- Vite 7
- Pinia (state management)
- Vue Router 4
- Axios

## AI/ML Models
- YOLOv5 / CTD / DBNet (text detection)
- MangaOCR (Japanese OCR)
- PaddleOCR ONNX (multi-language OCR)
- LAMA (inpainting)

## Build & Packaging
- PyInstaller (desktop distribution)

## Common Commands

### Backend
```bash
# Install dependencies (CPU)
pip install -r requirements-cpu.txt

# Install dependencies (GPU with CUDA)
pip install -r requirements-gpu.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Run development server
python app.py
# Server runs at http://127.0.0.1:5000/
```

### Vue Frontend
```bash
cd vue-frontend

# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build
```

## Key Dependencies
- `openai` - OpenAI-compatible API client
- `manga-ocr` - Japanese manga OCR
- `rapidocr-onnxruntime` - PaddleOCR ONNX runtime
- `litelama` - LAMA inpainting model
- `ultralytics` - YOLO detection
- `chromadb` - Vector database for manga insight
- `edge-tts` - Text-to-speech
