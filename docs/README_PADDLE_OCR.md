# PaddleOCR 多语言识别功能文档

## 功能简介

PaddleOCR 引擎支持多语言文本识别，目前在本应用中实现了以下语言的支持：

- 日语 (Japanese)
- 英语 (English)
- 韩语 (Korean)

PaddleOCR 具有较高的识别准确率和处理速度，可以有效识别图像中的文字内容，包括漫画、游戏截图等各种场景。

## 安装依赖

确保已安装所有必要的依赖项：

```bash
pip install -r requirements.txt
```

这将安装 PaddleOCR 及其依赖项，包括 PaddlePaddle。

## 使用方法

1. 在应用界面中选择"源语言"（日语、英语或韩语）
2. 上传需要翻译的图片
3. 应用会自动使用 PaddleOCR 识别图片中的文字
4. 识别结果会显示在界面上，并自动进行翻译

当识别日语文本时，系统会优先使用 PaddleOCR 引擎，如果遇到初始化问题会自动降级使用 MangaOCR。

## 注意事项

- 首次使用 PaddleOCR 时，系统会自动下载相应语言的模型文件
- 模型文件下载可能需要一些时间，请耐心等待
- 识别准确率受图片质量、字体清晰度和文字方向等因素影响
- 如果使用了自定义字体或特殊字体，识别效果可能会受到影响

## 调试信息

如果遇到 PaddleOCR 识别问题，可以查看以下调试信息：

- 控制台日志信息，查找带有 `PaddleOCR` 或 `BubbleDetection` 标签的日志
- 检查是否存在 PaddleOCR 初始化失败的错误
- 临时文件夹中可能会保存识别过程中的中间图像，方便排查问题

## 模型文件结构

PaddleOCR 模型文件保存在用户目录下的 `.paddleocr` 文件夹中：

```
HOME_DIR/.paddleocr/
    ├── whl    # 模型文件存储位置
    │   ├── det
    │   │   └── ch_PP-OCRv3_det_infer  # 检测模型 (~3MB)
    │   ├── rec
    │   │   ├── japan_PP-OCRv3_rec_infer  # 日语识别模型 (~16MB)
    │   │   ├── korean_PP-OCRv3_rec_infer  # 韩语识别模型 (~11MB)
    │   │   └── en_PP-OCRv3_rec_infer  # 英语识别模型 (~8MB)
    │   └── cls
    │       └── ch_ppocr_mobile_v2.0_cls_infer  # 分类模型 (~0.7MB)
```

## 打包说明

在使用 PyInstaller 打包应用程序时，PaddleOCR 可能会遇到一些特殊的目录结构问题。以下是打包时的常见问题及解决方法：

### 路径冲突问题

打包时可能会遇到 `Pyinstaller needs to create a directory at '...\tools\__init__.py', but there already exists a file at that path!` 类型的错误。这是因为 PyInstaller 尝试将文件同时作为目录和文件处理。

解决方法：
1. 在 `app.spec` 文件中，不要简单地添加整个目录，而是使用 `os.walk()` 遍历文件，并保持目录结构：

```python
# 创建单独的PaddleOCR相关文件列表，以便可以控制添加顺序
paddleocr_datas = []
if paddleocr_path:
    # 添加主paddleocr包
    paddleocr_datas.append((paddleocr_path, 'paddleocr'))
    
    # 查找并添加所有需要的文件，使用合适的目标路径
    for root, dirs, files in os.walk(paddleocr_path):
        for file in files:
            if file.endswith('.py') or file.endswith('.txt'):
                source_file = os.path.join(root, file)
                # 获取相对于paddleocr_path的路径
                rel_path = os.path.relpath(root, paddleocr_path)
                if rel_path == '.':
                    target_path = 'paddleocr'
                else:
                    target_path = os.path.join('paddleocr', rel_path)
                paddleocr_datas.append((source_file, target_path))
```

2. 在 `datas` 列表中添加这些文件:

```python
datas=[
    # 其他数据文件...
    
    # 添加PaddleOCR特定文件
    *paddleocr_datas,
]
```

### rapidfuzz 警告

打包时可能会遇到 `Failed to process hook entry point 'EntryPoint(name='hook-dirs', value='rapidfuzz.__pyinstaller:get_hook_dirs'...` 警告。

解决方法是创建一个修复的钩子文件：

```python
# 尝试在运行前删除任何现有的rapidfuzz.__pyinstaller.py文件
rapidfuzz_pyinstaller_path = None
try:
    import rapidfuzz
    rapidfuzz_dir = os.path.dirname(rapidfuzz.__file__)
    rapidfuzz_pyinstaller_path = os.path.join(rapidfuzz_dir, "__pyinstaller.py")
    if os.path.exists(rapidfuzz_pyinstaller_path):
        print(f"正在备份并修改问题文件: {rapidfuzz_pyinstaller_path}")
        # 备份文件
        shutil.copy2(rapidfuzz_pyinstaller_path, rapidfuzz_pyinstaller_path + ".bak")
        # 创建正确的钩子文件
        with open(rapidfuzz_pyinstaller_path, 'w') as f:
            f.write("# Fixed pyinstaller hook file\n")
            f.write("def get_hook_dirs():\n")
            f.write("    return []\n")
        print("已修复rapidfuzz.__pyinstaller.py文件")
except Exception as e:
    print(f"修复rapidfuzz钩子文件失败: {e}")
```

经过以上修复后，PaddleOCR 在打包的应用程序中应该能够正常工作。如果出现初始化问题，系统会自动降级使用 MangaOCR 来确保应用功能不受影响。 