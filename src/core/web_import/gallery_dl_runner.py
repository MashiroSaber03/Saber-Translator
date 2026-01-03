"""
网页漫画导入 - Gallery-DL 引擎

提供基于 gallery-dl 的漫画图片提取和下载功能：
- 支持 Pixiv、MangaDex 等主流漫画网站
- 高速下载，无需 AI 分析
- 自动处理防盗链
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from .image_processor import ImageProcessor

logger = logging.getLogger("WebImport.GalleryDL")


@dataclass
class GalleryDLExtractResult:
    """Gallery-DL 提取结果"""
    success: bool
    comic_title: str = ""
    chapter_title: str = ""
    pages: List[Dict[str, Any]] = None
    total_pages: int = 0
    source_url: str = ""
    referer: str = ""
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.pages is None:
            self.pages = []


class GalleryDLRunner:
    """Gallery-DL 运行器"""
    
    def __init__(self, config: dict = None):
        """
        初始化 Gallery-DL 运行器
        
        Args:
            config: 配置字典
                - timeout: 超时时间（秒）
                - imagePreprocess: 图片预处理配置
        """
        self.config = config or {}
        # 直接从配置获取 timeout，如果没有则使用默认值
        # 注意：不要使用嵌套的 download.timeout，因为那是给下载器用的
        raw_timeout = self.config.get('timeout')
        if isinstance(raw_timeout, (int, float)) and raw_timeout > 0:
            self.timeout = raw_timeout
        else:
            self.timeout = 600  # 默认10分钟
        logger.info(f"GalleryDLRunner 初始化，超时设置: {self.timeout} 秒")
        
        # 图片预处理器
        preprocess_config = self.config.get('imagePreprocess', {})
        self.processor = ImageProcessor(preprocess_config)
    
    @staticmethod
    def is_available() -> bool:
        """
        检查 gallery-dl 是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "gallery_dl", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.debug(f"gallery-dl 不可用: {e}")
            return False
    
    @staticmethod
    def is_supported(url: str) -> bool:
        """
        检查 URL 是否受 gallery-dl 支持
        
        Args:
            url: 要检查的 URL
        
        Returns:
            bool: 是否支持
        """
        try:
            from gallery_dl import extractor
            return extractor.find(url) is not None
        except ImportError:
            # gallery-dl 未安装，尝试命令行检查
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "gallery_dl", "--list-extractors"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                # 简单检查 URL 是否匹配已知站点
                from urllib.parse import urlparse
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                
                # 常见支持的站点列表
                supported_domains = [
                    'pixiv.net', 'mangadex.org', 'twitter.com', 'x.com',
                    'deviantart.com', 'artstation.com', 'danbooru.donmai.us',
                    'gelbooru.com', 'konachan.com', 'yande.re', 'sankakucomplex.com',
                    'nhentai.net', 'e-hentai.org', 'exhentai.org',
                    'imgur.com', 'flickr.com', 'tumblr.com',
                    'weibo.com', 'bilibili.com', 'lofter.com'
                ]
                
                for supported in supported_domains:
                    if supported in domain:
                        return True
                return False
            except Exception:
                return False
        except Exception as e:
            logger.debug(f"检查 URL 支持失败: {e}")
            return False
    
    def extract_metadata(self, url: str) -> GalleryDLExtractResult:
        """
        提取元数据（直接下载模式）
        
        gallery-dl 的 --dump-json 对某些站点不适用，
        改为直接下载到临时目录，然后返回文件列表作为预览
        
        Args:
            url: 漫画页面 URL
        
        Returns:
            GalleryDLExtractResult: 提取结果
        """
        from urllib.parse import urlparse
        
        # 从 URL 提取标题
        parsed = urlparse(url)
        comic_title = parsed.path.strip('/').split('/')[-1] or "Gallery"
        
        logger.info(f"执行 gallery-dl 提取: {url}")
        
        # 使用项目的临时目录
        import os
        project_root = Path(__file__).parent.parent.parent.parent  # src/core/web_import -> 项目根目录
        project_temp_dir = project_root / "data" / "temp" / "gallery_dl"
        project_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 清理旧的临时文件
        for old_file in project_temp_dir.glob("*"):
            try:
                if old_file.is_file():
                    old_file.unlink()
                elif old_file.is_dir():
                    import shutil
                    shutil.rmtree(old_file)
            except Exception:
                pass
        
        temp_dir = str(project_temp_dir)
        
        try:
            # 使用 gallery-dl Python API 直接下载
            from gallery_dl import job, config
            
            # 清空之前的配置
            config.clear()
            
            # 配置输出目录和文件命名
            config.set((), "base-directory", temp_dir)
            config.set((), "directory", ["."])
            config.set((), "filename", "{num:>03}.{extension}")
            
            # 设置范围限制 - 只下载前20张用于预览
            config.set(("extractor",), "range", "1-20")
            
            logger.info(f"使用 gallery-dl Python API 下载到: {temp_dir}")
            
            # 执行下载
            djob = job.DownloadJob(url)
            djob.run()
            logger.info("gallery-dl 下载完成")
            
            # 查找下载的图片
            all_images = []
            for file_path in Path(temp_dir).rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in [
                    '.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'
                ]:
                    all_images.append(file_path)
            
            # 按文件名排序
            all_images.sort(key=lambda p: p.name)
            
            if not all_images:
                return GalleryDLExtractResult(
                    success=False,
                    source_url=url,
                    error="未能下载到任何图片，请检查URL是否正确"
                )
            
            # 构建页面列表
            pages = []
            for i, file_path in enumerate(all_images):
                page_num = i + 1
                pages.append({
                    "pageNumber": page_num,
                    "imageUrl": f"/api/web-import/static/temp/gallery_dl/{file_path.name}",
                    "localPath": str(file_path)
                })
            
            logger.info(f"gallery-dl 提取成功: {len(pages)} 张图片")
            
            return GalleryDLExtractResult(
                success=True,
                comic_title=comic_title,
                chapter_title=f"共 {len(pages)} 张（预览前20张）",
                pages=pages,
                total_pages=len(pages),
                source_url=url,
                referer=url
            )
            
        except TimeoutError:
            logger.error("gallery-dl 提取超时")
            return GalleryDLExtractResult(
                success=False,
                source_url=url,
                error="提取超时，请检查网络连接或尝试 AI Agent 模式"
            )
        except Exception as e:
            logger.exception("gallery-dl 提取异常")
            return GalleryDLExtractResult(
                success=False,
                source_url=url,
                error=str(e)
            )
    
    def download(
        self,
        url: str,
        selected_indices: List[int] = None,
        on_progress: Callable[[int, int], None] = None
    ) -> List[Dict[str, Any]]:
        """
        下载图片（托管下载）
        
        使用 gallery-dl 直接下载图片到临时目录
        
        Args:
            url: 漫画页面 URL
            selected_indices: 要下载的页面索引列表 (1-based)
            on_progress: 进度回调函数
        
        Returns:
            下载结果列表
        """
        # 使用项目的临时目录
        project_root = Path(__file__).parent.parent.parent.parent
        project_temp_dir = project_root / "data" / "temp" / "gallery_dl_download"
        project_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 清理旧的临时文件
        import shutil
        for old_file in project_temp_dir.glob("*"):
            try:
                if old_file.is_file():
                    old_file.unlink()
                elif old_file.is_dir():
                    shutil.rmtree(old_file)
            except Exception:
                pass
        
        temp_dir = str(project_temp_dir)
        
        try:
            # 使用 gallery-dl Python API 直接下载
            from gallery_dl import job, config
            
            config.clear()
            config.set((), "base-directory", temp_dir)
            config.set((), "directory", ["."])
            config.set((), "filename", "{num:>03}.{extension}")
            
            logger.info(f"执行 gallery-dl 下载: {url}")
            
            djob = job.DownloadJob(url)
            djob.run()
            logger.info("gallery-dl 下载完成")
            
        except Exception as e:
            logger.error(f"gallery-dl 下载失败: {e}")
            return []
        
        # 使用 rglob 递归查找所有图片
        all_images = []
        for file_path in Path(temp_dir).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in [
                '.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'
            ]:
                all_images.append(file_path)
        
        # 按文件名排序
        all_images.sort(key=lambda p: p.name)
        
        logger.info(f"找到 {len(all_images)} 个图片文件")
        
        # 筛选并处理
        images = []
        total = len(all_images)
        
        for i, file_path in enumerate(all_images):
            page_num = i + 1
            
            # 如果指定了 selected_indices 且当前不在其中，跳过
            if selected_indices and page_num not in selected_indices:
                continue
            
            try:
                # 读取文件
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                
                # 预处理图片
                processed_data, ext = self.processor.process(raw_data, file_path.name)
                
                # 生成文件名
                filename = f"page_{page_num:03d}.{ext}"
                
                # 转换为 Data URL
                data_url = ImageProcessor.to_data_url(processed_data, ext)
                
                images.append({
                    'index': i,
                    'filename': filename,
                    'dataUrl': data_url,
                    'size': len(processed_data),
                    'success': True
                })
                
                if on_progress:
                    on_progress(len(images), total if not selected_indices else len(selected_indices))
                
            except Exception as e:
                logger.warning(f"处理图片失败 {file_path}: {e}")
                images.append({
                    'index': i,
                    'filename': f"page_{page_num:03d}.failed",
                    'dataUrl': '',
                    'size': 0,
                    'success': False,
                    'error': str(e)
                })
        
        logger.info(f"gallery-dl 下载完成: {len([i for i in images if i.get('success')])} 张成功")
        
        return images


# 便捷函数
def check_gallery_dl_support(url: str) -> Dict[str, Any]:
    """
    检查 URL 是否支持 gallery-dl
    
    Args:
        url: 要检查的 URL
    
    Returns:
        {
            "available": bool,  # gallery-dl 是否可用
            "supported": bool,  # URL 是否支持
        }
    """
    available = GalleryDLRunner.is_available()
    supported = GalleryDLRunner.is_supported(url) if available else False
    
    return {
        "available": available,
        "supported": supported
    }
