// src/app/static/js/image_viewer.js
// 图片浏览器核心类 - 支持拖拽和缩放

/**
 * 单图片浏览器类
 */
export class ImageViewer {
    constructor(viewportId, wrapperId, options = {}) {
        this.viewport = document.getElementById(viewportId);
        this.wrapper = document.getElementById(wrapperId);
        
        if (!this.viewport || !this.wrapper) {
            console.warn(`ImageViewer: 未找到元素 viewport=${viewportId} wrapper=${wrapperId}`);
            return;
        }
        
        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;
        this.isDragging = false;
        this.lastX = 0;
        this.lastY = 0;
        
        this.minScale = options.minScale || 0.1;
        this.maxScale = options.maxScale || 5;
        this.zoomSpeed = options.zoomSpeed || 0.1;
        
        // 回调函数
        this.onScaleChangeCallback = options.onScaleChange || null;
        this.onTransformChangeCallback = options.onTransformChange || null;
        
        this.init();
    }
    
    init() {
        if (!this.viewport) return;
        
        // 鼠标滚轮缩放
        this.viewport.addEventListener('wheel', (e) => {
            // 笔刷模式下不处理滚轮缩放
            if (document.querySelector('.brush-mode-active')) {
                return;
            }
            e.preventDefault();
            const rect = this.viewport.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            const delta = e.deltaY > 0 ? (1 - this.zoomSpeed) : (1 + this.zoomSpeed);
            this.zoomAt(mouseX, mouseY, delta);
        }, { passive: false });
        
        // 鼠标拖动
        this.viewport.addEventListener('mousedown', (e) => {
            // 笔刷模式下不处理拖动
            if (document.querySelector('.brush-mode-active')) {
                return;
            }
            // 左键拖动
            if (e.button === 0) {
                // 检查是否点击了高亮框，如果是则不启动拖动
                if (e.target.closest('.bubble-highlight-box')) {
                    return;
                }
                // 检查是否处于标注模式，如果是则不启动拖动（让标注功能处理）
                if (document.querySelector('.annotation-mode-active')) {
                    return;
                }
                // 检查是否处于绘制模式（编辑模式下的添加气泡功能）
                if (document.querySelector('.drawing-mode')) {
                    return;
                }
                this.isDragging = true;
                this.lastX = e.clientX;
                this.lastY = e.clientY;
                this.viewport.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });
        
        // 使用document级别监听以确保拖动流畅
        this._mouseMoveHandler = (e) => {
            if (this.isDragging) {
                const dx = e.clientX - this.lastX;
                const dy = e.clientY - this.lastY;
                this.translateX += dx;
                this.translateY += dy;
                this.lastX = e.clientX;
                this.lastY = e.clientY;
                this.applyTransform();
            }
        };
        
        this._mouseUpHandler = () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.viewport.style.cursor = 'grab';
            }
        };
        
        document.addEventListener('mousemove', this._mouseMoveHandler);
        document.addEventListener('mouseup', this._mouseUpHandler);
        
        // 键盘控制
        this.viewport.tabIndex = 0;
        this.viewport.addEventListener('keydown', (e) => {
            const step = 50;
            let handled = true;
            switch(e.key) {
                case 'ArrowUp': this.translateY += step; break;
                case 'ArrowDown': this.translateY -= step; break;
                case 'ArrowLeft': this.translateX += step; break;
                case 'ArrowRight': this.translateX -= step; break;
                case '+': case '=': this.zoom(1.2); return;
                case '-': this.zoom(0.8); return;
                case '0': this.reset(); return;
                default: handled = false;
            }
            if (handled) {
                this.applyTransform();
                e.preventDefault();
            }
        });
        
        // 双击重置
        this.viewport.addEventListener('dblclick', (e) => {
            if (!e.target.closest('.bubble-highlight-box')) {
                this.fitToScreen();
            }
        });
    }
    
    /**
     * 在指定点缩放
     */
    zoomAt(x, y, factor) {
        const newScale = Math.min(Math.max(this.scale * factor, this.minScale), this.maxScale);
        const scaleChange = newScale / this.scale;
        
        // 以鼠标位置为中心缩放
        this.translateX = x - (x - this.translateX) * scaleChange;
        this.translateY = y - (y - this.translateY) * scaleChange;
        this.scale = newScale;
        
        this.applyTransform();
        if (this.onScaleChangeCallback) {
            this.onScaleChangeCallback(this.scale);
        }
    }
    
    /**
     * 以视口中心缩放
     */
    zoom(factor) {
        if (!this.viewport) return;
        const rect = this.viewport.getBoundingClientRect();
        this.zoomAt(rect.width / 2, rect.height / 2, factor);
    }
    
    /**
     * 设置缩放比例
     */
    setScale(scale) {
        if (!this.viewport) return;
        const rect = this.viewport.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        const factor = scale / this.scale;
        this.zoomAt(centerX, centerY, factor);
    }
    
    /**
     * 重置变换
     */
    reset() {
        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;
        this.applyTransform();
        if (this.onScaleChangeCallback) {
            this.onScaleChangeCallback(this.scale);
        }
    }
    
    /**
     * 适应屏幕
     */
    fitToScreen() {
        if (!this.wrapper || !this.viewport) return;
        
        const img = this.wrapper.querySelector('img');
        if (!img || !img.naturalWidth) {
            // 图片未加载，等待加载后重试
            img?.addEventListener('load', () => this.fitToScreen(), { once: true });
            return;
        }
        
        const viewportRect = this.viewport.getBoundingClientRect();
        const scaleX = viewportRect.width / img.naturalWidth;
        const scaleY = viewportRect.height / img.naturalHeight;
        this.scale = Math.min(scaleX, scaleY) * 0.95; // 留5%边距
        
        // 居中
        this.translateX = (viewportRect.width - img.naturalWidth * this.scale) / 2;
        this.translateY = (viewportRect.height - img.naturalHeight * this.scale) / 2;
        
        this.applyTransform();
        if (this.onScaleChangeCallback) {
            this.onScaleChangeCallback(this.scale);
        }
    }
    
    /**
     * 重置变换（用于切换图片前）
     */
    resetTransform() {
        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;
        this.applyTransform();
    }
    
    /**
     * 应用变换
     */
    applyTransform() {
        if (!this.wrapper) return;
        this.wrapper.style.transform = 
            `translate(${this.translateX}px, ${this.translateY}px) scale(${this.scale})`;
        
        if (this.onTransformChangeCallback) {
            this.onTransformChangeCallback({
                scale: this.scale,
                translateX: this.translateX,
                translateY: this.translateY
            });
        }
    }
    
    /**
     * 获取当前缩放比例
     */
    getScale() {
        return this.scale;
    }
    
    /**
     * 获取当前变换状态
     */
    getTransform() {
        return {
            scale: this.scale,
            translateX: this.translateX,
            translateY: this.translateY
        };
    }
    
    /**
     * 设置变换状态
     */
    setTransform(transform) {
        this.scale = transform.scale;
        this.translateX = transform.translateX;
        this.translateY = transform.translateY;
        this.applyTransform();
    }
    
    /**
     * 同步另一个viewer的位置和缩放
     */
    syncWith(otherViewer) {
        if (!otherViewer) return;
        this.scale = otherViewer.scale;
        this.translateX = otherViewer.translateX;
        this.translateY = otherViewer.translateY;
        this.applyTransform();
    }
    
    /**
     * 滚动到指定气泡位置
     */
    scrollToBubble(bubbleCoords, imageWidth, imageHeight) {
        if (!this.viewport || !bubbleCoords || bubbleCoords.length < 4) return;
        
        const [x1, y1, x2, y2] = bubbleCoords;
        const bubbleCenterX = (x1 + x2) / 2;
        const bubbleCenterY = (y1 + y2) / 2;
        
        const viewportRect = this.viewport.getBoundingClientRect();
        const viewportCenterX = viewportRect.width / 2;
        const viewportCenterY = viewportRect.height / 2;
        
        // 计算需要的平移量，使气泡居中
        this.translateX = viewportCenterX - bubbleCenterX * this.scale;
        this.translateY = viewportCenterY - bubbleCenterY * this.scale;
        
        this.applyTransform();
    }
    
    /**
     * 销毁实例，清理事件监听
     */
    destroy() {
        if (this._mouseMoveHandler) {
            document.removeEventListener('mousemove', this._mouseMoveHandler);
        }
        if (this._mouseUpHandler) {
            document.removeEventListener('mouseup', this._mouseUpHandler);
        }
    }
}


/**
 * 双图同步浏览器类
 */
export class DualImageViewer {
    constructor(options = {}) {
        this.syncEnabled = options.syncEnabled !== false;
        this.onScaleChange = options.onScaleChange || null;
        
        this.originalViewer = null;
        this.translatedViewer = null;
        this.isInitialized = false;
    }
    
    /**
     * 初始化双图浏览器
     */
    init() {
        // 创建原图浏览器
        this.originalViewer = new ImageViewer('originalViewport', 'originalCanvasWrapper', {
            onScaleChange: (scale) => this.handleScaleChange(scale),
            onTransformChange: (transform) => this.handleOriginalTransformChange(transform)
        });
        
        // 创建翻译图浏览器
        this.translatedViewer = new ImageViewer('translatedViewport', 'translatedCanvasWrapper', {
            onScaleChange: (scale) => this.handleScaleChange(scale),
            onTransformChange: (transform) => this.handleTranslatedTransformChange(transform)
        });
        
        this.isInitialized = true;
        console.log('DualImageViewer 初始化完成');
    }
    
    /**
     * 处理缩放变化
     */
    handleScaleChange(scale) {
        if (this.onScaleChange) {
            this.onScaleChange(scale);
        }
        // 更新显示
        const zoomLevelEl = document.getElementById('zoomLevel');
        if (zoomLevelEl) {
            zoomLevelEl.textContent = Math.round(scale * 100) + '%';
        }
    }
    
    /**
     * 处理原图变换变化
     */
    handleOriginalTransformChange(transform) {
        if (this.syncEnabled && this.translatedViewer) {
            // 暂时禁用同步以避免循环
            const wasSync = this.syncEnabled;
            this.syncEnabled = false;
            this.translatedViewer.setTransform(transform);
            this.syncEnabled = wasSync;
        }
    }
    
    /**
     * 处理翻译图变换变化
     */
    handleTranslatedTransformChange(transform) {
        if (this.syncEnabled && this.originalViewer) {
            // 暂时禁用同步以避免循环
            const wasSync = this.syncEnabled;
            this.syncEnabled = false;
            this.originalViewer.setTransform(transform);
            this.syncEnabled = wasSync;
        }
    }
    
    /**
     * 切换同步状态
     */
    toggleSync() {
        this.syncEnabled = !this.syncEnabled;
        console.log('双图同步:', this.syncEnabled ? '开启' : '关闭');
        return this.syncEnabled;
    }
    
    /**
     * 适应屏幕
     */
    fitToScreen() {
        if (this.originalViewer) this.originalViewer.fitToScreen();
        if (this.translatedViewer) this.translatedViewer.fitToScreen();
    }
    
    /**
     * 重置变换（用于切换图片前）
     */
    resetTransform() {
        if (this.originalViewer) this.originalViewer.resetTransform();
        if (this.translatedViewer) this.translatedViewer.resetTransform();
    }
    
    /**
     * 放大
     */
    zoomIn() {
        if (this.translatedViewer) {
            this.translatedViewer.zoom(1.2);
        }
    }
    
    /**
     * 缩小
     */
    zoomOut() {
        if (this.translatedViewer) {
            this.translatedViewer.zoom(0.8);
        }
    }
    
    /**
     * 重置缩放
     */
    resetZoom() {
        if (this.originalViewer) this.originalViewer.reset();
        if (this.translatedViewer) this.translatedViewer.reset();
    }
    
    /**
     * 设置缩放比例
     */
    setScale(scale) {
        if (this.originalViewer) this.originalViewer.setScale(scale);
        if (this.translatedViewer) this.translatedViewer.setScale(scale);
    }
    
    /**
     * 获取当前缩放比例
     */
    getScale() {
        return this.translatedViewer ? this.translatedViewer.getScale() : 1;
    }
    
    /**
     * 滚动到指定气泡
     */
    scrollToBubble(bubbleCoords, imageWidth, imageHeight) {
        if (this.originalViewer) {
            this.originalViewer.scrollToBubble(bubbleCoords, imageWidth, imageHeight);
        }
        if (this.translatedViewer) {
            this.translatedViewer.scrollToBubble(bubbleCoords, imageWidth, imageHeight);
        }
    }
    
    /**
     * 销毁实例
     */
    destroy() {
        if (this.originalViewer) {
            this.originalViewer.destroy();
            this.originalViewer = null;
        }
        if (this.translatedViewer) {
            this.translatedViewer.destroy();
            this.translatedViewer = null;
        }
        this.isInitialized = false;
    }
}


/**
 * 面板分隔条拖动控制器
 */
export class PanelDividerController {
    constructor(dividerId, leftPanelId, rightPanelId, options = {}) {
        this.divider = document.getElementById(dividerId);
        this.leftPanel = document.getElementById(leftPanelId);
        this.rightPanel = document.getElementById(rightPanelId);
        
        if (!this.divider || !this.leftPanel || !this.rightPanel) {
            console.warn('PanelDividerController: 未找到必要元素');
            return;
        }
        
        this.minWidth = options.minWidth || 200;
        this.isDragging = false;
        
        this.init();
    }
    
    init() {
        this.divider.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const container = this.leftPanel.parentElement;
            const containerRect = container.getBoundingClientRect();
            const mouseX = e.clientX - containerRect.left;
            
            const totalWidth = containerRect.width - this.divider.offsetWidth;
            let leftWidth = mouseX;
            
            // 限制最小宽度
            leftWidth = Math.max(this.minWidth, leftWidth);
            leftWidth = Math.min(totalWidth - this.minWidth, leftWidth);
            
            const leftPercent = (leftWidth / totalWidth) * 100;
            const rightPercent = 100 - leftPercent;
            
            this.leftPanel.style.flex = `0 0 ${leftPercent}%`;
            this.rightPanel.style.flex = `0 0 ${rightPercent}%`;
        });
        
        document.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        });
    }
}


/**
 * 底部面板高度调整控制器
 */
export class BottomPanelResizer {
    constructor(resizerId, panelId, options = {}) {
        this.resizer = document.getElementById(resizerId);
        this.panel = document.getElementById(panelId);
        
        if (!this.resizer || !this.panel) {
            console.warn('BottomPanelResizer: 未找到必要元素');
            return;
        }
        
        this.minHeight = options.minHeight || 100;
        this.maxHeight = options.maxHeight || window.innerHeight * 0.5;
        this.isDragging = false;
        
        this.init();
    }
    
    init() {
        this.resizer.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.startY = e.clientY;
            this.startHeight = this.panel.offsetHeight;
            document.body.style.cursor = 'ns-resize';
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const deltaY = this.startY - e.clientY;
            let newHeight = this.startHeight + deltaY;
            
            // 限制高度范围
            newHeight = Math.max(this.minHeight, newHeight);
            newHeight = Math.min(this.maxHeight, newHeight);
            
            this.panel.style.height = newHeight + 'px';
        });
        
        document.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        });
    }
}


/**
 * 右侧/底部面板尺寸调整控制器
 * 支持水平布局（调整宽度）和垂直布局（调整高度）
 */
export class SidePanelResizer {
    constructor(resizerId, panelId, options = {}) {
        this.resizer = document.getElementById(resizerId);
        this.panel = document.getElementById(panelId);
        
        if (!this.resizer || !this.panel) {
            console.warn('SidePanelResizer: 未找到必要元素');
            return;
        }
        
        this.minWidth = options.minWidth || 480;
        this.maxWidth = options.maxWidth || window.innerWidth * 0.7;
        this.minHeight = options.minHeight || 200;
        this.maxHeight = options.maxHeight || window.innerHeight * 0.6;
        this.isDragging = false;
        
        this.init();
    }
    
    /**
     * 检测当前是否为垂直布局模式
     */
    isVerticalLayout() {
        const workspace = document.getElementById('editWorkspace');
        return workspace && workspace.classList.contains('layout-vertical');
    }
    
    init() {
        this.resizer.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.isVertical = this.isVerticalLayout();
            
            if (this.isVertical) {
                this.startY = e.clientY;
                this.startHeight = this.panel.offsetHeight;
                document.body.style.cursor = 'ns-resize';
            } else {
                this.startX = e.clientX;
                this.startWidth = this.panel.offsetWidth;
                document.body.style.cursor = 'ew-resize';
            }
            
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            if (this.isVertical) {
                // 垂直布局：调整高度（向上拖动增加高度）
                const deltaY = this.startY - e.clientY;
                let newHeight = this.startHeight + deltaY;
                
                // 动态计算最大高度（窗口高度的60%，但不超过可用空间）
                const workspace = document.getElementById('editWorkspace');
                const availableHeight = workspace ? workspace.offsetHeight - 150 : window.innerHeight * 0.6;
                const maxH = Math.min(this.maxHeight, availableHeight);
                
                newHeight = Math.max(this.minHeight, newHeight);
                newHeight = Math.min(maxH, newHeight);
                
                this.panel.style.flex = `0 0 auto`;
                this.panel.style.height = `${newHeight}px`;
                this.panel.style.maxHeight = `${newHeight}px`;
                this.panel.style.minHeight = `${newHeight}px`;
            } else {
                // 水平布局：调整宽度
                const deltaX = this.startX - e.clientX;
                let newWidth = this.startWidth + deltaX;
                
                newWidth = Math.max(this.minWidth, newWidth);
                newWidth = Math.min(this.maxWidth, newWidth);
                
                this.panel.style.flex = `0 0 ${newWidth}px`;
                this.panel.style.minWidth = `${newWidth}px`;
            }
        });
        
        document.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        });
    }
}