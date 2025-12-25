# Project Structure

```
saber-translator/
├── app.py                 # Flask application entry point
├── requirements-*.txt     # Python dependencies (cpu/gpu variants)
│
├── src/                   # Main source code
│   ├── app/               # Flask application layer
│   │   ├── api/           # REST API endpoints (blueprints)
│   │   │   ├── translation/   # Translation-related APIs
│   │   │   ├── system/        # System/config APIs
│   │   │   └── manga_insight/ # Manga analysis APIs
│   │   ├── static/        # Static assets (CSS, JS, fonts)
│   │   ├── templates/     # Jinja2 HTML templates
│   │   ├── routes.py      # Main page routes
│   │   └── error_handlers.py
│   │
│   ├── core/              # Core business logic
│   │   ├── detection.py       # Text region detection
│   │   ├── ocr.py             # OCR orchestration
│   │   ├── translation.py     # Translation orchestration
│   │   ├── inpainting.py      # Background inpainting
│   │   ├── rendering.py       # Text rendering
│   │   ├── processing.py      # Main processing pipeline
│   │   ├── session_manager.py # Session persistence
│   │   ├── bookshelf_manager.py # Book/manga management
│   │   └── manga_insight/     # Manga analysis features
│   │
│   ├── interfaces/        # External service adapters
│   │   ├── manga_ocr_interface.py
│   │   ├── paddle_ocr_interface.py
│   │   ├── baidu_ocr_interface.py
│   │   ├── lama_interface.py
│   │   ├── vision_interface.py
│   │   └── *_translate_interface.py
│   │
│   ├── plugins/           # Plugin system
│   │   ├── base.py        # PluginBase class
│   │   ├── manager.py     # Plugin lifecycle management
│   │   └── hooks.py       # Hook definitions
│   │
│   ├── shared/            # Shared utilities
│   │   ├── constants.py   # Application constants
│   │   ├── types.py       # Type definitions
│   │   ├── config_loader.py
│   │   └── image_helpers.py
│   │
│   └── utils/             # Utility modules
│
├── vue-frontend/          # Vue 3 frontend (migration in progress)
│   ├── src/
│   │   ├── components/
│   │   ├── App.vue
│   │   └── main.ts
│   └── vite.config.ts
│
├── models/                # ML model files
│   ├── ctd/               # Comic Text Detector
│   ├── lama/              # LAMA inpainting
│   ├── manga_ocr/         # MangaOCR weights
│   ├── paddle_ocr_onnx/   # PaddleOCR ONNX models
│   └── yolo/              # YOLO detection
│
├── config/                # Runtime configuration
├── data/                  # User data (sessions, bookshelf, temp)
├── plugins/               # User-installed plugins
└── logs/                  # Application logs
```

## Architecture Patterns

### API Layer
- Flask Blueprints organize endpoints by domain
- APIs return JSON responses
- Error handling via centralized error_handlers.py

### Core Layer
- `processing.py` orchestrates the full pipeline
- Each step (detection, OCR, translation, inpainting, rendering) is modular
- Interfaces abstract external services (OCR engines, translation APIs)

### Plugin System
- Plugins extend `PluginBase` from `src/plugins/base.py`
- Hook methods: `before_processing`, `after_detection`, `after_ocr`, `before_translation`, `after_translation`, `before_rendering`, `after_processing`
- Plugin configs stored in `config/plugin_configs/`

### Data Flow
1. Image upload → base64 encoding (frontend)
2. Detection → bubble coordinates
3. OCR → original text extraction
4. Translation → translated text
5. Inpainting → clean background
6. Rendering → final image with translated text
