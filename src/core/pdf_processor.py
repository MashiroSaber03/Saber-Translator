import io
import logging
from PIL import Image
import PyPDF2 

logger = logging.getLogger("PDFProcessor")

def extract_images_from_pdf(pdf_file_stream):
    """
    从 PDF 文件流中提取图像。

    Args:
        pdf_file_stream: PDF 文件的文件流对象 (例如通过 request.files 获取)。

    Returns:
        list: 包含提取出的 PIL Image 对象的列表。
    """
    images = []
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        num_pages = len(pdf_reader.pages)
        logger.info(f"开始处理 PDF 文件，共 {num_pages} 页。")

        for page_num in range(num_pages):
            try:
                page = pdf_reader.pages[page_num]
                # PyPDF2 >= 3.0.0 使用 page.images
                if hasattr(page, 'images') and page.images:
                    logger.debug(f"页面 {page_num + 1}: 发现 {len(page.images)} 个图像对象 (新方法)。")
                    for i, img_obj in enumerate(page.images):
                        try:
                            # img_obj.data 是图像的字节数据
                            img = Image.open(io.BytesIO(img_obj.data))
                            # 尝试获取原始文件名（可能没有）
                            img_name = img_obj.name if hasattr(img_obj, 'name') else f"page{page_num+1}_img{i+1}"
                            logger.info(f"  成功提取图像: {img_name} (模式: {img.mode}, 尺寸: {img.size})")
                            images.append(img)
                        except Exception as img_e:
                            logger.warning(f"  提取页面 {page_num + 1} 的图像 {i+1} 失败: {img_e}")
                # 兼容旧版 PyPDF2 或不同结构的 PDF
                elif '/Resources' in page and '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    img_count = 0
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            img_count += 1
                            try:
                                img_data = xObject[obj].get_data() # 使用 get_data()
                                img = Image.open(io.BytesIO(img_data))
                                logger.info(f"  成功提取图像 (XObject): {obj} (模式: {img.mode}, 尺寸: {img.size})")
                                images.append(img)
                            except Exception as img_e:
                                logger.warning(f"  提取页面 {page_num + 1} 的 XObject 图像 {obj} 失败: {img_e}")
                    if img_count > 0:
                         logger.debug(f"页面 {page_num + 1}: 发现 {img_count} 个 XObject 图像对象 (旧方法)。")

            except Exception as page_e:
                logger.error(f"处理 PDF 页面 {page_num + 1} 时出错: {page_e}", exc_info=True)

        logger.info(f"PDF 处理完成，共提取 {len(images)} 张图片。")
        return images

    except PyPDF2.errors.PdfReadError as pdf_err:
         logger.error(f"无法读取 PDF 文件，可能是文件损坏或密码保护: {pdf_err}")
         return []
    except Exception as e:
        logger.error(f"处理 PDF 文件时发生未知错误: {e}", exc_info=True)
        return []

# --- 测试代码 ---
if __name__ == '__main__':
    from src.shared.path_helpers import resource_path # 需要导入
    import os

    print("--- 测试 PDF 处理器 ---")
    # 需要一个测试 PDF 文件路径
    test_pdf_path = resource_path('docs/example.pdf') # 替换为你的测试 PDF 文件路径

    if os.path.exists(test_pdf_path):
        print(f"加载测试 PDF: {test_pdf_path}")
        try:
            with open(test_pdf_path, 'rb') as f:
                extracted_images = extract_images_from_pdf(f)
            print(f"提取完成，共找到 {len(extracted_images)} 张图片。")

            # 保存提取的图片用于检查 (可选)
            if extracted_images:
                save_dir = get_debug_dir("pdf_extracted_images")
                print(f"将提取的图片保存到: {save_dir}")
                for i, img in enumerate(extracted_images):
                    try:
                        img.save(os.path.join(save_dir, f"extracted_image_{i+1}.png"))
                    except Exception as save_e:
                        print(f"  保存图片 {i+1} 失败: {save_e}")
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
    else:
        print(f"错误：测试 PDF 文件未找到 {test_pdf_path}")