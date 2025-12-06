/**
 * 阅读页面 JavaScript
 * 处理漫画阅读功能
 */

// 当前状态
let currentBookId = null;
let currentChapterId = null;
let currentBookInfo = null;
let currentChapterInfo = null;
let chaptersData = [];
let imagesData = [];
let currentViewMode = 'translated'; // 'original' 或 'translated'

// 阅读设置
let readerSettings = {
    imageWidth: 100,
    imageGap: 8,
    bgColor: '#1a1a2e'
};

// ==================== 初始化 ====================

document.addEventListener('DOMContentLoaded', () => {
    // 解析URL参数
    const urlParams = new URLSearchParams(window.location.search);
    currentBookId = urlParams.get('book');
    currentChapterId = urlParams.get('chapter');
    
    if (!currentBookId || !currentChapterId) {
        showToast('缺少书籍或章节参数', 'error');
        setTimeout(() => window.location.href = '/', 2000);
        return;
    }
    
    // 加载设置
    loadSettings();
    
    // 初始化事件监听
    initEventListeners();
    
    // 加载数据
    loadReaderData();
    
    // 主题初始化
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.body.setAttribute('data-theme', savedTheme);
});

function initEventListeners() {
    // 返回按钮
    document.getElementById('backBtn').addEventListener('click', () => {
        window.location.href = '/';
    });
    
    // 翻译按钮
    document.getElementById('translateBtn').addEventListener('click', () => {
        window.location.href = `/translate?book=${currentBookId}&chapter=${currentChapterId}`;
    });
    
    // 空状态的翻译按钮
    document.getElementById('goTranslateBtn').addEventListener('click', () => {
        window.location.href = `/translate?book=${currentBookId}&chapter=${currentChapterId}`;
    });
    
    // 查看模式切换
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const mode = e.target.dataset.mode;
            setViewMode(mode);
        });
    });
    
    // 设置按钮
    document.getElementById('settingsBtn').addEventListener('click', () => {
        document.getElementById('settingsPanel').classList.add('active');
    });
    
    // 章节导航
    document.getElementById('prevChapterBtn').addEventListener('click', () => navigateChapter('prev'));
    document.getElementById('nextChapterBtn').addEventListener('click', () => navigateChapter('next'));
    
    // 回到顶部
    document.getElementById('scrollTopBtn').addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    
    // 滚动监听
    window.addEventListener('scroll', handleScroll);
    
    // 设置滑块
    document.getElementById('imageWidthSlider').addEventListener('input', (e) => {
        readerSettings.imageWidth = parseInt(e.target.value);
        document.getElementById('imageWidthValue').textContent = `${readerSettings.imageWidth}%`;
        applySettings();
        saveSettings();
    });
    
    document.getElementById('imageGapSlider').addEventListener('input', (e) => {
        readerSettings.imageGap = parseInt(e.target.value);
        document.getElementById('imageGapValue').textContent = `${readerSettings.imageGap}px`;
        applySettings();
        saveSettings();
    });
    
    // 背景颜色选项
    document.querySelectorAll('.bg-option').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.bg-option').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            readerSettings.bgColor = e.target.dataset.bg;
            applySettings();
            saveSettings();
        });
    });
    
    // 键盘快捷键
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeSettings();
        }
        if (e.key === 'ArrowLeft') {
            navigateChapter('prev');
        }
        if (e.key === 'ArrowRight') {
            navigateChapter('next');
        }
        if (e.key === 'Home') {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        if (e.key === 'End') {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }
    });
}

// ==================== 数据加载 ====================

async function loadReaderData() {
    try {
        // 并行加载书籍信息和章节图片
        const [bookResult, imagesResult] = await Promise.all([
            apiRequest(`/api/bookshelf/books/${currentBookId}`),
            apiRequest(`/api/bookshelf/books/${currentBookId}/chapters/${currentChapterId}/images`)
        ]);
        
        currentBookInfo = bookResult.book;
        chaptersData = currentBookInfo.chapters || [];
        currentChapterInfo = chaptersData.find(c => c.id === currentChapterId);
        imagesData = imagesResult.images || [];
        
        // 更新UI
        updateHeaderInfo();
        updateChapterNav();
        renderImages();
        
        // 隐藏加载状态
        document.getElementById('loadingState').style.display = 'none';
        
        if (imagesData.length === 0) {
            document.getElementById('emptyState').style.display = 'flex';
        } else {
            document.getElementById('imagesContainer').style.display = 'flex';
            document.getElementById('chapterNav').style.display = 'flex';
        }
        
    } catch (error) {
        console.error('加载数据失败:', error);
        showToast('加载失败: ' + error.message, 'error');
    }
}

function updateHeaderInfo() {
    document.getElementById('bookTitle').textContent = currentBookInfo?.title || '未知书籍';
    document.getElementById('chapterTitle').textContent = currentChapterInfo?.title || '未知章节';
    document.title = `${currentChapterInfo?.title || '阅读'} - ${currentBookInfo?.title || 'Saber-Translator'}`;
}

function updateChapterNav() {
    const currentIndex = chaptersData.findIndex(c => c.id === currentChapterId);
    
    const prevBtn = document.getElementById('prevChapterBtn');
    const nextBtn = document.getElementById('nextChapterBtn');
    
    prevBtn.disabled = currentIndex <= 0;
    nextBtn.disabled = currentIndex >= chaptersData.length - 1;
}

function updatePageInfo() {
    const images = document.querySelectorAll('.reader-image-wrapper');
    const viewportCenter = window.innerHeight / 2;
    let currentPage = 1;
    
    images.forEach((img, index) => {
        const rect = img.getBoundingClientRect();
        if (rect.top < viewportCenter && rect.bottom > 0) {
            currentPage = index + 1;
        }
    });
    
    document.getElementById('pageInfo').textContent = `${currentPage} / ${imagesData.length}`;
}

// ==================== 图片渲染 ====================

function renderImages() {
    const container = document.getElementById('imagesContainer');
    container.innerHTML = '';
    
    imagesData.forEach((img, index) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'reader-image-wrapper';
        
        const imgElement = document.createElement('img');
        imgElement.className = 'reader-image';
        imgElement.alt = `第 ${index + 1} 页`;
        imgElement.loading = 'lazy';
        
        // 根据当前模式选择图片源
        const src = getImageSource(img);
        if (src) {
            imgElement.src = src;
        } else {
            imgElement.src = '';
            imgElement.style.minHeight = '200px';
            imgElement.style.background = 'rgba(255,255,255,0.05)';
        }
        
        // 页码标记
        const indexLabel = document.createElement('div');
        indexLabel.className = 'image-index';
        indexLabel.textContent = `${index + 1} / ${imagesData.length}`;
        
        wrapper.appendChild(imgElement);
        wrapper.appendChild(indexLabel);
        container.appendChild(wrapper);
    });
    
    updatePageInfo();
}

function getImageSource(imageData) {
    if (currentViewMode === 'translated') {
        // 优先显示翻译后的图片，如果没有则显示原图
        return imageData.translated || imageData.original;
    } else {
        return imageData.original;
    }
}

function setViewMode(mode) {
    currentViewMode = mode;
    
    // 更新按钮状态
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    
    // 更新图片
    const images = document.querySelectorAll('.reader-image');
    images.forEach((img, index) => {
        const src = getImageSource(imagesData[index]);
        if (src && img.src !== src) {
            img.src = src;
        }
    });
}

// ==================== 章节导航 ====================

function navigateChapter(direction) {
    const currentIndex = chaptersData.findIndex(c => c.id === currentChapterId);
    let newIndex;
    
    if (direction === 'prev') {
        newIndex = currentIndex - 1;
    } else {
        newIndex = currentIndex + 1;
    }
    
    if (newIndex >= 0 && newIndex < chaptersData.length) {
        const newChapter = chaptersData[newIndex];
        window.location.href = `/reader?book=${currentBookId}&chapter=${newChapter.id}`;
    }
}

// ==================== 滚动处理 ====================

function handleScroll() {
    const scrollTop = window.scrollY;
    const scrollTopBtn = document.getElementById('scrollTopBtn');
    
    // 显示/隐藏回到顶部按钮
    if (scrollTop > 500) {
        scrollTopBtn.style.display = 'block';
    } else {
        scrollTopBtn.style.display = 'none';
    }
    
    // 更新页码
    updatePageInfo();
}

// ==================== 设置管理 ====================

function loadSettings() {
    const saved = localStorage.getItem('readerSettings');
    if (saved) {
        try {
            readerSettings = { ...readerSettings, ...JSON.parse(saved) };
        } catch (e) {
            console.error('加载设置失败:', e);
        }
    }
    
    // 应用到UI
    document.getElementById('imageWidthSlider').value = readerSettings.imageWidth;
    document.getElementById('imageWidthValue').textContent = `${readerSettings.imageWidth}%`;
    
    document.getElementById('imageGapSlider').value = readerSettings.imageGap;
    document.getElementById('imageGapValue').textContent = `${readerSettings.imageGap}px`;
    
    document.querySelectorAll('.bg-option').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.bg === readerSettings.bgColor);
    });
    
    applySettings();
}

function saveSettings() {
    localStorage.setItem('readerSettings', JSON.stringify(readerSettings));
}

function applySettings() {
    document.body.style.background = readerSettings.bgColor;
    document.documentElement.style.setProperty('--reader-image-width', `${readerSettings.imageWidth}%`);
    document.documentElement.style.setProperty('--reader-gap', `${readerSettings.imageGap}px`);
}

function closeSettings() {
    document.getElementById('settingsPanel').classList.remove('active');
}

// ==================== API 请求 ====================

async function apiRequest(url, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(url, options);
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || '操作失败');
        }
        
        return result;
    } catch (error) {
        console.error('API请求失败:', error);
        throw error;
    }
}

// ==================== Toast 通知 ====================

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast ' + type;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// ==================== 全局暴露 ====================

window.closeSettings = closeSettings;

