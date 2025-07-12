# Seber-Translator-linux

Self-use version adapted for Linux; after installing the necessary dependencies, you can run it with python app.py.


## requirement
1. 'sd-webui-cleaner' from original project's release package.
2. 'ultralytics_yolov5_master' from original project's release package.
3. 'weights' from from original project's release package.

## change
1. WebP is used as the default intermediate storage medium globally to save cache space.
2. Added the tag qwen no_think
3. In order to read the YOLO files trained by the original project on Windows, this program forcibly uses Path conversion. 
It is currently unclear whether any bugs will occur.