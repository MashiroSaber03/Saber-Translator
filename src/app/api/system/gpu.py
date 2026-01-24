"""
GPU 资源管理 API

提供清理 GPU 显存和卸载模型的接口
"""

from flask import jsonify
import torch
import gc
import logging

from . import system_bp

logger = logging.getLogger(__name__)


@system_bp.route('/cleanup-gpu', methods=['POST'])
def cleanup_gpu():
    """
    清理 GPU 显存并卸载所有已加载的模型
    
    在进入翻译页面时调用，确保 GPU 状态干净
    
    Returns:
        JSON: 清理结果
    """
    try:
        unloaded_models = []
        
        # 1. 卸载 LAMA MPE 模型
        try:
            from src.interfaces.lama_mpe_interface import get_lama_mpe_inpainter, LamaMPEInpainter
            if LamaMPEInpainter._model is not None:
                inpainter = get_lama_mpe_inpainter()
                inpainter.unload()
                unloaded_models.append('lama_mpe')
                logger.info("已卸载 LAMA MPE 模型")
        except Exception as e:
            logger.warning(f"卸载 LAMA MPE 时出错: {e}")
        
        # 2. 卸载 litelama 模型
        try:
            from src.interfaces.lama_interface import get_litelama_inpainter, LiteLamaInpainter
            if LiteLamaInpainter._model is not None:
                inpainter = get_litelama_inpainter()
                inpainter.unload()
                unloaded_models.append('litelama')
                logger.info("已卸载 litelama 模型")
        except Exception as e:
            logger.warning(f"卸载 litelama 时出错: {e}")
        
        # 3. 强制垃圾回收（多次确保彻底）
        for _ in range(3):
            gc.collect()
        
        # 4. 清理 CUDA 缓存
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 获取清理后的显存信息
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
            
            logger.info(f"GPU 显存清理完成 - 已分配: {memory_allocated:.1f}MB, 已预留: {memory_reserved:.1f}MB")
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        return jsonify({
            'success': True,
            'message': 'GPU 资源已清理',
            'unloaded_models': unloaded_models,
            'cuda_available': cuda_available,
            'memory_allocated_mb': round(memory_allocated, 1),
            'memory_reserved_mb': round(memory_reserved, 1)
        })
        
    except Exception as e:
        logger.error(f"清理 GPU 时出错: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@system_bp.route('/gpu-status', methods=['GET'])
def get_gpu_status():
    """
    获取 GPU 状态信息
    
    Returns:
        JSON: GPU 状态
    """
    try:
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        else:
            device_name = "N/A"
            memory_allocated = 0
            memory_reserved = 0
            memory_total = 0
        
        # 检查模型加载状态
        models_loaded = []
        
        try:
            from src.interfaces.lama_mpe_interface import LamaMPEInpainter
            if LamaMPEInpainter._model is not None:
                models_loaded.append('lama_mpe')
        except:
            pass
        
        try:
            from src.interfaces.lama_interface import LiteLamaInpainter
            if LiteLamaInpainter._model is not None:
                models_loaded.append('litelama')
        except:
            pass
        
        return jsonify({
            'success': True,
            'cuda_available': cuda_available,
            'device_name': device_name,
            'memory_allocated_mb': round(memory_allocated, 1),
            'memory_reserved_mb': round(memory_reserved, 1),
            'memory_total_mb': round(memory_total, 1),
            'models_loaded': models_loaded
        })
        
    except Exception as e:
        logger.error(f"获取 GPU 状态时出错: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
