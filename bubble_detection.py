import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import manga_ocr
import os
from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_dir, "weights", "best.pt")

repo_dir = os.path.join(current_dir, "ultralytics_yolov5_master")

if os.path.exists(repo_dir):
    model = torch.hub.load(repo_or_dir=repo_dir, source='local', model='custom', path=weights_path)
else:
    raise FileNotFoundError(f'ultralytics/yolov5 项目文件未找到：{repo_dir}。请确保已下载项目文件并将其放在正确的位置。'
                            f'Checkout https://github.com/ultralytics/yolov5 for available files.')

model.conf = 0.6

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "manga_ocr_model")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Manga OCR 模型文件夹未找到：{model_path}。请确保已下载模型文件并将其放在正确的位置。"
                            f"Checkout 'https://huggingface.co/kha-white/manga-ocr-base/tree/main' for available files.")

ocr = manga_ocr.MangaOcr(model_path)

DEFAULT_PROMPT = "你是一个好用的翻译助手。请将我的日文翻译成中文，我发给你所有的话都是需要翻译的内容，你只需要回答翻译结果。特别注意：翻译结果字数不能超过原文字数！翻译结果请符合中文的语言习惯。"

def translate_text(text, target_language, model_provider, custom_base_url=None, api_key=None, model_name=None, prompt_content=None):
    if prompt_content is None:
        prompt_content = DEFAULT_PROMPT

    if model_provider == 'siliconflow':
        client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_content},
                    {"role": "user", "content": text},
                ],
                timeout=10
            )
            translated_text = response.choices[0].message.content.strip()
            return translated_text
        except Exception as e:
            print(f"翻译 API 请求失败: {e}")
            return "翻译失败"
    elif model_provider == 'deepseek':
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_content},
                    {"role": "user", "content": text},
                ],
                timeout=10
            )
            translated_text = response.choices[0].message.content.strip()
            return translated_text
        except Exception as e:
            print(f"DeepSeek 翻译 API 请求失败: {e}")
            return "翻译失败"
    elif model_provider == 'custom':
        if not custom_base_url:
            print(f"自定义服务商需要提供对应的URL！")
            return "翻译失败"
        client = OpenAI(api_key=api_key, base_url=custom_base_url)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_content},
                    {"role": "user", "content": text},
                ],
                timeout=10
            )
            translated_text = response.choices[0].message.content.strip()
            return translated_text
        except Exception as e:
            print(f"自定义服务商翻译 API 请求失败: {e}")
            return "翻译失败"
    else:
        print(f"未知的翻译模型提供商: {model_provider}")
        return text

def draw_multiline_text_vertical_right_to_left(draw, text, font, x, y, max_height, fill='black'):
    if not text:
        return

    lines = []
    current_line = ""
    current_column_height = 0
    line_height = font.size + 5

    for char in text:
        bbox = font.getbbox(char)
        char_height = bbox[3] - bbox[1]

        if current_column_height + line_height <= max_height:
            current_line += char
            current_column_height += line_height
        else:
            lines.append(current_line)
            current_line = char
            current_column_height = line_height

    lines.append(current_line)

    current_x = x
    column_width = font.size + 5

    for line in lines:
        current_y = y
        for char in line:
            draw.text((current_x, current_y), char, font=font, fill=fill, anchor="rt")
            current_y += line_height
        current_x -= column_width

def draw_multiline_text_horizontal(draw, text, font, x, y, max_width, fill='black'):
    if not text:
        return

    lines = []
    current_line = ""
    current_line_width = 0

    for char in text:
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        space_width = font.getbbox(' ')[2] - font.getbbox(' ')[0]

        if current_line_width + char_width <= max_width:
            current_line += char
            current_line_width += char_width
        else:
            lines.append(current_line)
            current_line = char
            current_line_width = char_width

    lines.append(current_line)

    current_y = y
    line_height = font.size + 5

    for line in lines:
        current_x = x
        for char in line:
            draw.text((current_x, current_y), char, font=font, fill=fill, anchor="lt")
            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            current_x += char_width
        current_y += line_height

def detect_text_in_bubbles(image, target_language='zh', text_direction='vertical', fontSize=30, model_provider='siliconflow', custom_base_url=None, api_key=None, model_name=None, fontFamily="static/STSONG.TTF", prompt_content=None):
    try:
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        results = model(img_cv)

        boxes = results.xyxy[0][:, :4].cpu().numpy()
        scores = results.xyxy[0][:, 4].cpu().numpy()
        class_ids = results.xyxy[0][:, 5].cpu().numpy()
        valid_detections = scores > model.conf
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        class_ids = class_ids[valid_detections]
        bubble_texts = []
        bubble_coords = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            bubble_coords.append((x1, y1, x2, y2))

        bubble_coords.sort(key=lambda x: x[2], reverse=True)

        for x1, y1, x2, y2 in bubble_coords:
            bubble_img = img_cv[y1:y2, x1:x2]
            bubble_img_pil = Image.fromarray(cv2.cvtColor(bubble_img, cv2.COLOR_BGR2RGB))
            text = ocr(bubble_img_pil)
            translated_text = translate_text(text, target_language=target_language, model_provider=model_provider, custom_base_url=custom_base_url, api_key=api_key, model_name=model_name, prompt_content=prompt_content)
            bubble_texts.append(translated_text)

        img_pil = image.copy()
        draw = ImageDraw.Draw(img_pil)

        # 修改字体路径获取方式
        font_path = os.path.join(os.path.dirname(__file__), fontFamily)
        print(f"尝试加载字体 (detect_text_in_bubbles): {font_path}")

        try:
            font = ImageFont.truetype(font_path, fontSize, encoding="utf-8")
        except IOError as e:
            font = ImageFont.load_default()
            print(f"使用默认字体,因为发生以下错误：{e}")
        except Exception as e:
            print(f"加载字体时发生未知错误: {e}")

        if font is None:
            print("警告：未能成功加载字体，请检查字体路径和文件是否正确")

        for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
            draw.rectangle(((x1, y1), (x2, y2)), fill='white')

            text_x = x2 - 10
            text_y = y1 + 10
            max_text_height = y2 - y1 - 20
            max_text_width = x2 - x1 - 20

            if text_direction == 'vertical':
                draw_multiline_text_vertical_right_to_left(draw, bubble_texts[i], font, text_x, text_y, max_text_height)
            elif text_direction == 'horizontal':
                draw_multiline_text_horizontal(draw, bubble_texts[i], font, x1 + 10, y1 + 10, max_text_width)

        img_with_bubbles_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_with_bubbles_pil = Image.fromarray(cv2.cvtColor(img_with_bubbles_cv, cv2.COLOR_BGR2RGB))

        return img_with_bubbles_pil, bubble_texts, bubble_coords

    except Exception as e:
        print(f"Error in detect_text_in_bubbles: {e}")
        return image, [], []

def re_render_text_in_bubbles(image, translated_texts, bubble_coords, fontSize=30, fontFamily="static/STSONG.TTF", text_direction='vertical'):
    try:
        img_pil = image.copy()
        draw = ImageDraw.Draw(img_pil)

        # 修改字体路径获取方式
        font_path = os.path.join(os.path.dirname(__file__), fontFamily)
        print(f"尝试加载字体 (re_render_text_in_bubbles): {font_path}")

        try:
            font = ImageFont.truetype(font_path, fontSize, encoding="utf-8")
        except IOError as e:
            font = ImageFont.load_default()
            print(f"使用默认字体,因为发生以下错误：{e}")
        except Exception as e:
            print(f"加载字体时发生未知错误: {e}")

        if font is None:
            print("警告：未能成功加载字体，请检查字体路径和文件是否正确")
            return image

        if not translated_texts or not bubble_coords:
            print("警告：没有翻译文本或气泡坐标，无法重新渲染。")
            return image

        for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
            if i < len(translated_texts):
                draw.rectangle(((x1, y1), (x2, y2)), fill='white')

                text_x = x2 - 10
                text_y = y1 + 10
                max_text_height = y2 - y1 - 20
                max_text_width = x2 - x1 - 20

                if text_direction == 'vertical':
                    draw_multiline_text_vertical_right_to_left(draw, translated_texts[i], font, text_x, text_y, max_text_height)
                elif text_direction == 'horizontal':
                    draw_multiline_text_horizontal(draw, translated_texts[i], font, x1 + 10, y1 + 10, max_text_width)
            else:
                print(f"警告：气泡坐标数量多于翻译文本数量，索引 {i} 超出范围。")

        img_with_bubbles_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_with_bubbles_pil = Image.fromarray(cv2.cvtColor(img_with_bubbles_cv, cv2.COLOR_BGR2RGB))

        return img_with_bubbles_pil

    except Exception as e:
        print(f"Error in re_render_text_in_bubbles: {e}")
        return image