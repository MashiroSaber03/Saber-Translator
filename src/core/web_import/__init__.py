"""
网页漫画导入模块

提供漫画图片提取功能：
- MangaScraperAgent: AI Agent 核心逻辑 (AI + Firecrawl)
- GalleryDLRunner: Gallery-DL 引擎 (主流站点高速下载)
- ImageDownloader: 图片下载器
- ImageProcessor: 图片预处理器
"""

from .agent import MangaScraperAgent
from .image_downloader import ImageDownloader
from .image_processor import ImageProcessor
from .gallery_dl_runner import GalleryDLRunner, check_gallery_dl_support

__all__ = [
    'MangaScraperAgent',
    'GalleryDLRunner',
    'check_gallery_dl_support',
    'ImageDownloader', 
    'ImageProcessor'
]
