import os
from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
import io
from bubble_detection import detect_text_in_bubbles, DEFAULT_PROMPT, re_render_text_in_bubbles
import webbrowser
import threading
from openai import OpenAI
import json
import PyPDF2
from flask_cors import CORS

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='')
CORS(app)

MODEL_INFO_FILE = 'model_info.json'
PROMPT_INFO_FILE = 'prompt_info.json'
MAX_MODEL_HISTORY = 5
DEFAULT_PROMPT_NAME = "默认提示词"

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

@app.route('/')
def index():
    prompts = load_prompts()
    prompt_names = [prompt['name'] for prompt in prompts['saved_prompts']]
    default_prompt_content = get_default_prompt_content()
    return render_template('index.html', prompt_names=prompt_names, default_prompt_content=default_prompt_content)

@app.route('/translate_image', methods=['POST'])
def translate_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400

        image_data = data.get('image')
        target_language = data.get('target_language')
        text_direction = data.get('textDirection')
        fontSize_str = data.get('fontSize')
        custom_base_url = data.get('custom_base_url')
        api_key = data.get('api_key')
        model_name = data.get('model_name')
        model_provider = data.get('model_provider')
        fontFamily = data.get('fontFamily')
        prompt_content = data.get('prompt_content')

        if not all([image_data, target_language, text_direction, fontSize_str, api_key, model_name, model_provider, fontFamily]):
            return jsonify({'error': '缺少必要的参数'}), 400

        try:
            fontSize = int(fontSize_str)
            if fontSize <= 0:
                return jsonify({'error': '字号大小必须是正整数'}), 400
        except ValueError:
            return jsonify({'error': '字号大小必须是整数'}), 400

        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        translated_image, bubble_texts, bubble_coords = detect_text_in_bubbles(
            img,
            target_language,
            text_direction=text_direction,
            fontSize=fontSize,
            model_provider=model_provider,
            custom_base_url=custom_base_url,
            api_key=api_key,
            model_name=model_name,
            fontFamily=fontFamily,
            prompt_content=prompt_content
        )

        buffered = io.BytesIO()
        translated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        save_model_info(model_provider, model_name)

        return jsonify({
            'translated_image': img_str,
            'bubble_texts': bubble_texts,
            'bubble_coords': bubble_coords
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/re_render_image', methods=['POST'])
def re_render_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400

        image_data = data.get('image')
        translated_text = data.get('translated_text')
        bubble_coords = data.get('bubble_coords')
        fontSize_str = data.get('fontSize')
        fontFamily = data.get('fontFamily')
        text_direction = data.get('textDirection')

        if not all([image_data, translated_text, bubble_coords, fontSize_str, fontFamily, text_direction]):
            return jsonify({'error': '缺少必要的参数'}), 400

        try:
            fontSize = int(fontSize_str)
            if fontSize <= 0:
                return jsonify({'error': '字号大小必须是正整数'}), 400
        except ValueError:
            return jsonify({'error': '字号大小必须是整数'}), 400

        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        rendered_image = re_render_text_in_bubbles(
            img,
            translated_text,
            bubble_coords,
            fontSize,
            fontFamily=fontFamily,
            text_direction=text_direction
        )

        buffered = io.BytesIO()
        rendered_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'rendered_image': img_str})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


def load_model_info():
    try:
        with open(MODEL_INFO_FILE, 'r') as f:
            try:
                model_info = json.load(f)
            except json.JSONDecodeError:
                print("JSON 文件为空或格式不正确，使用默认值")
                model_info = {}
        return model_info
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"加载模型信息失败: {e}")
        return {}


def save_model_info(model_provider, model_name):
    model_info = load_model_info()

    if model_provider not in model_info:
        model_info[model_provider] = []

    if model_name and model_name not in model_info[model_provider]:
        model_info[model_provider].insert(0, model_name)
        model_info[model_provider] = model_info[model_provider][:MAX_MODEL_HISTORY]

    try:
        with open(MODEL_INFO_FILE, 'w') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存模型信息失败: {e}")

@app.route('/get_used_models', methods=['GET'])
def get_used_models():
    model_provider = request.args.get('model_provider')
    if not model_provider:
        return jsonify({'error': '缺少 model_provider 参数'}), 400

    model_info = load_model_info()
    used_models = model_info.get(model_provider, [])
    return jsonify({'models': used_models})

@app.route('/get_model_info', methods=['GET'])
def get_model_info():
    model_info = load_model_info()
    return jsonify(model_info)

@app.route('/save_model_info', methods=['POST'])
def route_save_model_info():
    data = request.get_json()
    if not data or 'modelProvider' not in data or 'modelName' not in data:
        return jsonify({'error': '缺少模型供应商或模型名称'}), 400

    model_provider = data['modelProvider']
    model_name = data['modelName']
    save_model_info(model_provider, model_name)
    return jsonify({'message': '模型信息保存成功'})

def pdf_to_images(pdf_file):
    images = []
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file) 
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            if '/Resources' in page and '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()

                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        try:
                            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                            data = xObject[obj].get_data()
                            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                                mode = "RGB"
                            else:
                                mode = "P"

                            if '/Filter' in xObject[obj]:
                                if xObject[obj]['/Filter'] == '/FlateDecode':
                                    img = Image.frombytes(mode, size, data)
                                    images.append(img)
                                elif xObject[obj]['/Filter'] == '/DCTDecode':
                                    img = Image.open(io.BytesIO(data))
                                    images.append(img)
                                else:
                                    print("Unknown filter, skipping")
                            else:
                                img = Image.frombytes(mode, size, data)
                                images.append(img)
                        except Exception as image_e:
                            print(f"提取图片失败，页面 {page_num + 1}, 对象 {obj}: {image_e}")
        return images
    except Exception as e:
        print(f"PDF 转换失败: {e}")
        return []

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdfFile' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    pdf_file = request.files['pdfFile']
    if pdf_file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if pdf_file:
        try:
            images = pdf_to_images(pdf_file)
            image_data_list = []
            for i, image in enumerate(images):
                try:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_data_list.append(img_str)
                except Exception as save_e:
                    print(f"保存图片失败，跳过图片{i+1}: {save_e}")

            return jsonify({'images': image_data_list}), 200
        except Exception as e:
            print(f"处理 PDF 文件时出错: {e}")
            return jsonify({'error': f"处理 PDF 文件时出错: {e}"}), 500

    return jsonify({'error': '上传失败'}), 500

def load_prompts():
    try:
        with open(PROMPT_INFO_FILE, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        print(f"{PROMPT_INFO_FILE} 文件未找到，创建默认文件。")
        prompt_data = {"default_prompt": DEFAULT_PROMPT, "saved_prompts": []}
        save_prompts(prompt_data)
    except json.JSONDecodeError:
        print(f"{PROMPT_INFO_FILE} JSON解码错误，使用默认提示词配置。")
        prompt_data = {"default_prompt": DEFAULT_PROMPT, "saved_prompts": []}
    return prompt_data

def save_prompts(prompt_data):
    try:
        with open(PROMPT_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存提示词信息失败: {e}")

def get_default_prompt_content():
    prompts = load_prompts()
    return prompts.get('default_prompt', DEFAULT_PROMPT)

@app.route('/get_prompts', methods=['GET'])
def get_prompts():
    prompts = load_prompts()
    prompt_names = [prompt['name'] for prompt in prompts['saved_prompts']]
    default_prompt_content = prompts.get('default_prompt', DEFAULT_PROMPT)
    return jsonify({'prompt_names': prompt_names, 'default_prompt_content': default_prompt_content})

@app.route('/save_prompt', methods=['POST'])
def save_prompt():
    data = request.get_json()
    if not data or 'prompt_name' not in data or 'prompt_content' not in data:
        return jsonify({'error': '缺少提示词名称或内容'}), 400

    prompt_name = data['prompt_name']
    prompt_content = data['prompt_content']

    prompts = load_prompts()
    existing_prompt_index = next((index for (index, d) in enumerate(prompts['saved_prompts']) if d["name"] == prompt_name), None)
    if existing_prompt_index is not None:
        prompts['saved_prompts'][existing_prompt_index]['content'] = prompt_content
    else:
        prompts['saved_prompts'].append({'name': prompt_name, 'content': prompt_content})

    save_prompts(prompts)
    return jsonify({'message': '提示词保存成功'})

@app.route('/get_prompt_content', methods=['GET'])
def get_prompt_content():
    prompt_name = request.args.get('prompt_name')
    if not prompt_name:
        return jsonify({'error': '缺少提示词名称'}), 400

    prompts = load_prompts()
    if prompt_name == DEFAULT_PROMPT_NAME:
        prompt_content = prompts.get('default_prompt', DEFAULT_PROMPT)
    else:
        saved_prompt = next((prompt for prompt in prompts['saved_prompts'] if prompt['name'] == prompt_name), None)
        prompt_content = saved_prompt['content'] if saved_prompt else None

    if prompt_content:
        return jsonify({'prompt_content': prompt_content})
    else:
        return jsonify({'error': '提示词未找到'}), 404

@app.route('/reset_prompt_to_default', methods=['POST'])
def reset_prompt_to_default():
    prompts = load_prompts()
    prompts['default_prompt'] = DEFAULT_PROMPT
    save_prompts(prompts)
    return jsonify({'message': '提示词已重置为默认'})

@app.route('/delete_prompt', methods=['POST'])
def delete_prompt():
    data = request.get_json()
    if not data or 'prompt_name' not in data:
        return jsonify({'error': '缺少提示词名称'}), 400

    prompt_name = data['prompt_name']
    prompts = load_prompts()
    prompts['saved_prompts'] = [prompt for prompt in prompts['saved_prompts'] if prompt['name'] != prompt_name]
    save_prompts(prompts)
    return jsonify({'message': '提示词删除成功'})


if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    print("程序正在运行，请在浏览器中访问 http://127.0.0.1:5000/")
    app.run(debug=True, use_reloader=False)
