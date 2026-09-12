"""Reject incomplete release model archives before packaging (no inference)."""

from pathlib import Path
import sys


# Standard offline model pack. LaMA Manga is an optional separate download.
REQUIRED_FILES = """
default/detect-20241225.ckpt
ctd/comictextdetector.pt
ctd/comictextdetector.pt.onnx
yolo/ysgyolo_1.2_OS1.0.pt
saber_yolo/saber_yolo.pt
lama/inpainting_lama_mpe.ckpt
lama/big-lama.safetensors
manga_ocr/config.json
manga_ocr/preprocessor_config.json
manga_ocr/pytorch_model.bin
manga_ocr/tokenizer_config.json
manga_ocr/vocab.txt
ocr_48px/ocr_ar_48px.ckpt
ocr_48px/alphabet-all-v7.txt
paddle_ocr_onnx_v6/det.onnx
paddle_ocr_onnx_v6/rec.onnx
paddle_ocr_onnx_v6/ppocrv6_dict.txt
paddleocr_vl_1_6/config.json
paddleocr_vl_1_6/generation_config.json
paddleocr_vl_1_6/model.safetensors
paddleocr_vl_1_6/preprocessor_config.json
paddleocr_vl_1_6/processor_config.json
paddleocr_vl_1_6/added_tokens.json
paddleocr_vl_1_6/special_tokens_map.json
paddleocr_vl_1_6/tokenizer.json
paddleocr_vl_1_6/tokenizer.model
paddleocr_vl_1_6/tokenizer_config.json
paddleocr_vl_1_6/chat_template.jinja
""".split()


def main() -> None:
    root = Path(sys.argv[1])
    missing = [name for name in REQUIRED_FILES
               if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        raise SystemExit("Incomplete model pack; missing or empty files:\n" + "\n".join(missing))
    print(f"Verified {len(REQUIRED_FILES)} required model files")


if __name__ == "__main__":
    main()
