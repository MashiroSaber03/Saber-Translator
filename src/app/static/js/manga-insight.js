/**
 * Manga Insight 漫画分析页面脚本
 */

// ==================== 全局状态 ====================
const MangaInsight = {
    currentBookId: null,
    currentTaskId: null,
    books: [],
    config: {},
    analysisStatus: null,
    bookInfo: null
};

// 挂载到 window 对象，供其他脚本访问
window.MangaInsight = MangaInsight;

// ==================== 初始化 ====================
async function initMangaInsight() {
    // 加载书籍列表
    await loadBookList();
    
    // 加载配置
    await loadConfig();
    
    // 初始化提示词编辑器
    await initPromptsEditor();
    
    // 绑定事件
    bindEvents();
    
    // 检查 URL 参数
    const urlParams = new URLSearchParams(window.location.search);
    const bookId = urlParams.get('book');
    if (bookId) {
        loadBook(bookId);
    }
}

function bindEvents() {
    // 设置按钮
    document.getElementById('settingsBtn')?.addEventListener('click', openSettingsModal);
    
    // 主题切换
    document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);
    
    // 批量分析设置变化时更新估算
    document.getElementById('pagesPerBatch')?.addEventListener('input', updateBatchEstimate);
    document.getElementById('architecturePreset')?.addEventListener('change', onArchitectureChange);
}

// ==================== 书籍管理 ====================
async function loadBookList() {
    try {
        const response = await fetch('/api/bookshelf/books');
        const data = await response.json();
        
        if (data.success) {
            MangaInsight.books = data.books || [];
            renderBookSelector();
        }
    } catch (error) {
        console.error('加载书籍列表失败:', error);
        showToast('加载书籍列表失败', 'error');
    }
}

function renderBookSelector() {
    const selector = document.getElementById('bookSelector');
    if (!selector) return;
    
    selector.innerHTML = '<option value="">-- 选择书籍 --</option>';
    
    MangaInsight.books.forEach(book => {
        const option = document.createElement('option');
        option.value = book.book_id;
        option.textContent = book.title || book.book_id;
        selector.appendChild(option);
    });
}

async function loadBook(bookId) {
    if (!bookId) return;
    
    MangaInsight.currentBookId = bookId;
    showLoading('加载书籍信息...');
    
    try {
        // 获取书籍信息
        const bookResponse = await fetch(`/api/bookshelf/books/${bookId}`);
        const bookData = await bookResponse.json();
        
        if (!bookData.success) {
            throw new Error(bookData.error || '获取书籍信息失败');
        }
        
        const book = bookData.book;
        
        // 更新 UI（添加空值检查）
        const bookTitleEl = document.getElementById('bookTitle');
        const totalPagesEl = document.getElementById('totalPages');
        if (bookTitleEl) bookTitleEl.textContent = book.title || bookId;
        if (totalPagesEl) totalPagesEl.textContent = book.total_pages || 0;
        
        // 封面
        const coverImg = document.getElementById('bookCover');
        const coverPlaceholder = document.getElementById('coverPlaceholder');
        if (book.cover && coverImg) {
            coverImg.src = book.cover;
            coverImg.style.display = 'block';
            if (coverPlaceholder) coverPlaceholder.style.display = 'none';
        } else {
            if (coverImg) coverImg.style.display = 'none';
            if (coverPlaceholder) coverPlaceholder.style.display = 'flex';
        }
        
        // 获取分析状态
        await loadAnalysisStatus();
        
        // 渲染内容导航树（章节+页面整合）
        await renderPagesTree(book);
        
        // 显示内容区
        const selectBookPrompt = document.getElementById('selectBookPrompt');
        const contentTabs = document.getElementById('contentTabs');
        if (selectBookPrompt) selectBookPrompt.style.display = 'none';
        if (contentTabs) contentTabs.style.display = 'flex';
        
        // 保存书籍信息
        MangaInsight.bookInfo = book;
        
        // 加载概览数据
        await loadOverviewData();
        
        // 加载其他数据
        if (MangaInsight.afterBookLoaded) {
            await MangaInsight.afterBookLoaded();
        }
        
        // 更新 URL
        const url = new URL(window.location);
        url.searchParams.set('book', bookId);
        window.history.pushState({}, '', url);
        
    } catch (error) {
        console.error('加载书籍失败:', error);
        showToast('加载书籍失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function loadAnalysisStatus() {
    if (!MangaInsight.currentBookId) return false;
    
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/analyze/status`);
        const data = await response.json();
        
        if (data.success) {
            MangaInsight.analysisStatus = data;
            MangaInsight._statusRetryCount = 0;  // 重置重试计数
            updateAnalysisStatusUI(data);
            return true;
        }
        return false;
    } catch (error) {
        // 网络错误时静默处理，避免刷屏
        MangaInsight._statusRetryCount = (MangaInsight._statusRetryCount || 0) + 1;
        if (MangaInsight._statusRetryCount <= 3) {
            console.warn('获取分析状态失败，稍后重试...');
        }
        // 超过3次失败后不再输出日志
        return false;
    }
}

function updateAnalysisStatusUI(status) {
    const statusDot = document.getElementById('statusDot');
    const statusLabel = document.getElementById('statusLabel');
    const statusProgress = document.getElementById('statusProgress');
    const analyzedPages = document.getElementById('analyzedPages');
    
    const analyzedCount = status.analyzed_pages_count || 0;
    if (analyzedPages) analyzedPages.textContent = analyzedCount;
    
    // 更新状态指示器（添加空值检查）
    if (statusDot) statusDot.className = 'status-dot';
    
    if (status.current_task) {
        const taskStatus = status.current_task.status;
        if (taskStatus === 'running') {
            if (statusDot) statusDot.classList.add('running');
            if (statusLabel) statusLabel.textContent = '分析中';
            const progress = status.current_task.progress;
            if (progress && statusProgress) {
                statusProgress.textContent = `${progress.analyzed_pages || 0}/${progress.total_pages || 0}`;
            }
            showAnalysisControls('running');
            updateProgress(status.current_task.progress);
        } else if (taskStatus === 'paused') {
            if (statusDot) statusDot.classList.add('paused');
            if (statusLabel) statusLabel.textContent = '已暂停';
            showAnalysisControls('paused');
        } else if (taskStatus === 'completed') {
            if (statusDot) statusDot.classList.add('completed');
            if (statusLabel) statusLabel.textContent = '已完成';
            if (statusProgress) statusProgress.textContent = '';
            showAnalysisControls('completed');
        } else if (taskStatus === 'failed') {
            if (statusDot) statusDot.classList.add('failed');
            if (statusLabel) statusLabel.textContent = '分析失败';
            if (statusProgress) statusProgress.textContent = '';
            showAnalysisControls('idle');
        } else if (taskStatus === 'cancelled') {
            if (statusLabel) statusLabel.textContent = '已取消';
            if (statusProgress) statusProgress.textContent = '';
            showAnalysisControls('idle');
        }
        MangaInsight.currentTaskId = status.current_task.task_id;
    } else if (status.analyzed) {
        if (statusDot) statusDot.classList.add('completed');
        if (statusLabel) statusLabel.textContent = '已分析';
        if (statusProgress) statusProgress.textContent = `${analyzedCount}页`;
        showAnalysisControls('completed');
    } else {
        if (statusLabel) statusLabel.textContent = '未分析';
        if (statusProgress) statusProgress.textContent = '';
        showAnalysisControls('idle');
    }
}

function showAnalysisControls(state) {
    const btnGroupIdle = document.getElementById('btnGroupIdle');
    const btnGroupRunning = document.getElementById('btnGroupRunning');
    const btnGroupPaused = document.getElementById('btnGroupPaused');
    const progressContainer = document.getElementById('progressContainer');
    const startBtn = document.getElementById('startAnalysisBtn');
    
    // 隐藏所有按钮组（添加空值检查）
    if (btnGroupIdle) btnGroupIdle.style.display = 'none';
    if (btnGroupRunning) btnGroupRunning.style.display = 'none';
    if (btnGroupPaused) btnGroupPaused.style.display = 'none';
    if (progressContainer) progressContainer.style.display = 'none';
    
    switch (state) {
        case 'idle':
        case 'completed':
            if (btnGroupIdle) btnGroupIdle.style.display = 'flex';
            // 更新按钮文字
            if (startBtn) {
                const btnSpan = startBtn.querySelector('span');
                if (btnSpan) {
                    btnSpan.textContent = state === 'completed' ? '重新分析' : '开始分析';
                }
            }
            break;
        case 'running':
            if (btnGroupRunning) btnGroupRunning.style.display = 'flex';
            if (progressContainer) progressContainer.style.display = 'block';
            break;
        case 'paused':
            if (btnGroupPaused) btnGroupPaused.style.display = 'flex';
            if (progressContainer) progressContainer.style.display = 'block';
            break;
    }
}

function updateProgress(progress) {
    if (!progress) return;
    
    const percentage = progress.percentage || 0;
    const progressFill = document.getElementById('progressFill');
    const statusProgress = document.getElementById('statusProgress');
    
    // 更新进度条
    if (progressFill) {
        progressFill.style.width = percentage + '%';
    }
    
    // 更新状态栏中的进度文本
    if (statusProgress) {
        statusProgress.textContent = `${progress.analyzed_pages || 0}/${progress.total_pages || 0}`;
    }
}

function selectPage(pageNum) {
    MangaInsight.selectedPage = pageNum;
    // 更新选中状态
    document.querySelectorAll('.tree-page-item').forEach(item => {
        item.classList.remove('selected');
    });
    document.querySelector(`.tree-page-item[data-page="${pageNum}"]`)?.classList.add('selected');
    
    // 加载页面详情
    loadPageDetail(pageNum);
}

async function loadPageDetail(pageNum) {
    const container = document.getElementById('pageDetail');
    
    // 构建图片 URL
    const imageUrl = `/api/manga-insight/${MangaInsight.currentBookId}/page-image/${pageNum}`;
    
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/pages/${pageNum}`);
        const data = await response.json();
        
        if (!data.success || !data.analysis) {
            container.innerHTML = `
                <div class="page-detail-content">
                    <h4>📄 第 ${pageNum} 页</h4>
                    <div class="page-detail-image">
                        <img src="${imageUrl}" alt="第${pageNum}页" onclick="openImagePreview('${imageUrl}')" onerror="this.parentElement.style.display='none'">
                    </div>
                    <p>此页尚未分析</p>
                    <button class="btn btn-secondary btn-sm" onclick="reanalyzePage(${pageNum})">
                        分析此页
                    </button>
                </div>
            `;
            return;
        }
        
        const analysis = data.analysis;
        
        let html = `<div class="page-detail-content">`;
        html += `<h4>📄 第 ${pageNum} 页</h4>`;
        
        // 显示页面图片
        html += `
            <div class="page-detail-image">
                <img src="${imageUrl}" alt="第${pageNum}页" onclick="openImagePreview('${imageUrl}')" onerror="this.parentElement.style.display='none'">
            </div>
        `;
        
        if (analysis.page_summary) {
            html += `<p>${analysis.page_summary}</p>`;
        }
        
        // 对话
        const dialogues = [];
        (analysis.panels || []).forEach(panel => {
            (panel.dialogues || []).forEach(d => {
                if (d.translated_text) {
                    dialogues.push(d);
                }
            });
        });
        
        if (dialogues.length > 0) {
            html += `<h4>💬 对话</h4>`;
            dialogues.forEach(d => {
                html += `
                    <div class="dialogue-item">
                        <div class="dialogue-speaker">${d.speaker_name || '未知'}</div>
                        <div class="dialogue-text">${d.translated_text}</div>
                    </div>
                `;
            });
        }
        
        html += `
            <div style="margin-top: 12px;">
                <button class="btn btn-secondary btn-sm" onclick="reanalyzePage(${pageNum})">
                    🔄 重新分析
                </button>
            </div>
        `;
        
        html += `</div>`;
        container.innerHTML = html;
        
    } catch (error) {
        console.error('加载页面详情失败:', error);
        container.innerHTML = '<div class="placeholder-text">加载失败</div>';
    }
}

// ==================== 分析控制 ====================

// 分析模式切换
function onAnalysisModeChange() {
    const analysisModeSelect = document.getElementById('analysisModeSelect');
    const mode = analysisModeSelect?.value || 'full';
    const chapterSelect = document.getElementById('chapterSelect');
    const pageNumInput = document.getElementById('pageNumInput');
    
    // 隐藏所有子选项
    if (chapterSelect) chapterSelect.style.display = 'none';
    if (pageNumInput) pageNumInput.style.display = 'none';
    
    if (mode === 'chapter') {
        if (chapterSelect) chapterSelect.style.display = 'block';
        populateChapterSelect();
    } else if (mode === 'page') {
        if (pageNumInput) pageNumInput.style.display = 'block';
    }
}

// 填充章节选择下拉框
function populateChapterSelect() {
    const select = document.getElementById('chapterSelect');
    if (!select) return;
    
    const chapters = MangaInsight.bookInfo?.chapters || [];
    
    select.innerHTML = '<option value="">选择章节...</option>';
    chapters.forEach((ch, idx) => {
        const option = document.createElement('option');
        option.value = ch.id || ch.chapter_id || idx;
        option.textContent = ch.title || `第 ${idx + 1} 章`;
        select.appendChild(option);
    });
}

async function startAnalysis() {
    if (!MangaInsight.currentBookId) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    // 防止重复启动
    if (MangaInsight.analysisStatus?.current_task?.status === 'running') {
        showToast('分析正在进行中，请等待完成', 'warning');
        return;
    }
    
    const analysisModeSelect = document.getElementById('analysisModeSelect');
    const incrementalModeCheckbox = document.getElementById('incrementalMode');
    const analysisMode = analysisModeSelect?.value || 'full';
    const incremental = incrementalModeCheckbox?.checked ?? true;
    
    let requestBody = {};
    let endpoint = `/api/manga-insight/${MangaInsight.currentBookId}/analyze/start`;
    
    if (analysisMode === 'full') {
        // 全书分析
        requestBody = { mode: incremental ? 'incremental' : 'full' };
    } else if (analysisMode === 'chapter') {
        // 单章节分析
        const chapterSelect = document.getElementById('chapterSelect');
        const chapterId = chapterSelect?.value;
        if (!chapterId) {
            showToast('请选择要分析的章节', 'error');
            return;
        }
        requestBody = { mode: 'chapters', chapters: [chapterId] };
    } else if (analysisMode === 'page') {
        // 单页分析
        const pageNumInput = document.getElementById('pageNumInput');
        const pageNum = parseInt(pageNumInput?.value);
        if (!pageNum || pageNum < 1) {
            showToast('请输入有效的页码', 'error');
            return;
        }
        requestBody = { mode: 'pages', pages: [pageNum] };
    }
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (data.success) {
            MangaInsight.currentTaskId = data.task_id;
            showToast('分析已启动', 'success');
            await loadAnalysisStatus();  // 立即更新状态
            startProgressPolling();
        } else {
            showToast('启动失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('启动分析失败:', error);
        showToast('启动分析失败', 'error');
    }
}

async function pauseAnalysis() {
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/analyze/pause`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: MangaInsight.currentTaskId })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('分析已暂停', 'success');
            await loadAnalysisStatus();
        } else {
            showToast('暂停失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('暂停分析失败:', error);
        showToast('暂停失败', 'error');
    }
}

async function resumeAnalysis() {
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/analyze/resume`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: MangaInsight.currentTaskId })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('分析已继续', 'success');
            startProgressPolling();
        } else {
            showToast('继续失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('继续分析失败:', error);
        showToast('继续失败', 'error');
    }
}

async function cancelAnalysis() {
    if (!confirm('确定要取消分析吗？')) return;
    
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/analyze/cancel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: MangaInsight.currentTaskId })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('分析已取消', 'success');
            stopProgressPolling();
            await loadAnalysisStatus();
        } else {
            showToast('取消失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('取消分析失败:', error);
        showToast('取消失败', 'error');
    }
}

let progressPollingInterval = null;

function startProgressPolling() {
    stopProgressPolling();
    
    progressPollingInterval = setInterval(async () => {
        const success = await loadAnalysisStatus();
        
        // 如果请求失败，不检查状态（保持轮询继续重试）
        if (!success) return;
        
        const status = MangaInsight.analysisStatus;
        if (status?.current_task) {
            const taskStatus = status.current_task.status;
            if (taskStatus === 'completed' || taskStatus === 'failed' || taskStatus === 'cancelled') {
                stopProgressPolling();
                await loadOverviewData();
                // 刷新内容导航树
                if (MangaInsight.bookInfo) {
                    await renderPagesTree(MangaInsight.bookInfo);
                }
            }
        } else {
            // 没有正在运行的任务，停止轮询
            stopProgressPolling();
        }
    }, 3000);  // 3秒轮询一次，减少请求频率
}

function stopProgressPolling() {
    if (progressPollingInterval) {
        clearInterval(progressPollingInterval);
        progressPollingInterval = null;
    }
}

async function reanalyzePage(pageNum) {
    try {
        showLoading('重新分析页面...');
        
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/reanalyze/page/${pageNum}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('页面分析已启动', 'success');
        } else {
            showToast('分析失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('重新分析失败:', error);
        showToast('重新分析失败', 'error');
    } finally {
        hideLoading();
    }
}

// ==================== 数据加载 ====================
async function loadOverviewData() {
    if (!MangaInsight.currentBookId) return;
    
    try {
        // 加载概述
        const overviewResponse = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/overview`);
        const overviewData = await overviewResponse.json();
        
        if (overviewData.success && overviewData.overview) {
            const overview = overviewData.overview;
            const storySummaryEl = document.getElementById('storySummary');
            if (storySummaryEl) {
                storySummaryEl.innerHTML = overview.summary || '<div class="placeholder-text">暂无概要</div>';
            }
            
            // 显示章节数
            if (overview.total_chapters !== undefined) {
                const statChaptersEl = document.getElementById('statChapters');
                if (statChaptersEl) statChaptersEl.textContent = overview.total_chapters;
            }
        }
        
        // 加载统计
        const statusResponse = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/analyze/status`);
        const statusData = await statusResponse.json();
        
        if (statusData.success) {
            const statPagesEl = document.getElementById('statPages');
            if (statPagesEl) statPagesEl.textContent = statusData.analyzed_pages_count || 0;
        }
        
        // 加载章节统计
        const chaptersResponse = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/chapters`);
        const chaptersData = await chaptersResponse.json();
        
        if (chaptersData.success && chaptersData.chapters) {
            const statChaptersEl = document.getElementById('statChapters');
            if (statChaptersEl) statChaptersEl.textContent = chaptersData.chapters.length;
        }
        
        // 加载时间线
        await loadTimeline();
        
    } catch (error) {
        console.error('加载概览数据失败:', error);
    }
}

async function loadTimeline() {
    if (!MangaInsight.currentBookId) return;
    
    const container = document.getElementById('timelineContainer');
    if (!container) return;
    
    try {
        // 从缓存加载时间线
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/timeline`);
        const data = await response.json();
        
        if (!data.success) {
            container.innerHTML = '<div class="placeholder-text">加载时间线失败</div>';
            return;
        }
        
        // 使用通用渲染函数
        renderTimeline(data);
        
    } catch (error) {
        console.error('加载时间线失败:', error);
        container.innerHTML = '<div class="placeholder-text">加载时间线失败</div>';
    }
}

function renderTimeline(data) {
    const container = document.getElementById('timelineContainer');
    if (!container) return;
    
    const mode = data.mode || 'simple';
    const stats = data.stats || {};
    const cached = data.cached;
    
    // 增强模式
    if (mode === 'enhanced') {
        renderEnhancedTimeline(data, container);
        return;
    }
    
    // 简单模式
    const groups = data.groups || [];
    
    if (groups.length === 0) {
        if (cached === false) {
            container.innerHTML = `
                <div class="timeline-empty-state">
                    <div class="empty-icon">📈</div>
                    <h4>时间线尚未生成</h4>
                    <p>完成漫画分析后会自动生成时间线，或点击下方按钮手动生成</p>
                    <button class="btn btn-primary btn-sm" onclick="regenerateTimeline()">
                        生成时间线
                    </button>
                </div>
            `;
        } else {
            container.innerHTML = '<div class="placeholder-text">暂无时间线数据，请先完成漫画分析</div>';
        }
        return;
    }
    
    // 构建简单时间线 HTML
    let html = `
        <div class="timeline-stats">
            <span class="stat-badge">📊 ${stats.total_events || 0} 个事件</span>
            <span class="stat-badge">📄 ${stats.total_pages || 0} 页</span>
        </div>
        <div class="timeline-track">
    `;
    
    groups.forEach((group, index) => {
        const pageRange = group.page_range || {};
        const startPage = pageRange.start || '?';
        const endPage = pageRange.end || '?';
        const events = group.events || [];
        const summary = group.summary || '';
        const thumbnailPage = group.thumbnail_page || startPage;
        
        html += `
            <div class="timeline-group" data-group-id="${group.id}">
                <div class="timeline-node">
                    <div class="timeline-node-dot"></div>
                    <div class="timeline-node-line"></div>
                </div>
                <div class="timeline-card">
                    <div class="timeline-card-header">
                        <img class="timeline-thumbnail" 
                             src="/api/manga-insight/${MangaInsight.currentBookId}/thumbnail/${thumbnailPage}" 
                             alt="第${startPage}页"
                             onerror="this.style.display='none'"
                             onclick="showPageDetail(${startPage})">
                        <div class="timeline-card-title">
                            <span class="timeline-page-range">第 ${startPage}-${endPage} 页</span>
                            <span class="timeline-event-count">${events.length} 个事件</span>
                        </div>
                    </div>
                    ${summary ? `<div class="timeline-summary">${summary}</div>` : ''}
                    ${events.length > 0 ? `
                        <ul class="timeline-events-list">
                            ${events.map(e => `<li class="timeline-event-item">${e}</li>`).join('')}
                        </ul>
                    ` : ''}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

function renderEnhancedTimeline(data, container) {
    const stats = data.stats || {};
    const storyArcs = data.story_arcs || [];
    const events = data.events || [];
    const characters = data.characters || [];
    const plotThreads = data.plot_threads || [];
    const summary = data.summary || {};
    
    // 检查是否有数据
    if (storyArcs.length === 0 && events.length === 0) {
        container.innerHTML = '<div class="placeholder-text">暂无时间线数据，请先完成漫画分析</div>';
        return;
    }
    
    // 构建事件映射
    const eventMap = {};
    events.forEach(e => { if (e.id) eventMap[e.id] = e; });
    
    let html = `
        <div class="enhanced-timeline">
            <!-- 统计信息 -->
            <div class="timeline-stats enhanced">
                <span class="stat-badge">🎭 ${stats.total_arcs || 0} 个剧情弧</span>
                <span class="stat-badge">📊 ${stats.total_events || 0} 个事件</span>
                <span class="stat-badge">👥 ${stats.total_characters || 0} 个角色</span>
                <span class="stat-badge">🔗 ${stats.total_threads || 0} 条线索</span>
                <span class="stat-badge">📄 ${stats.total_pages || 0} 页</span>
            </div>
            
            <!-- 故事摘要 -->
            ${summary.one_sentence ? `
                <div class="timeline-summary-card">
                    <h4>📖 故事概要</h4>
                    <p class="one-sentence">${summary.one_sentence}</p>
                    ${summary.main_conflict ? `<p class="main-conflict"><strong>主要冲突：</strong>${summary.main_conflict}</p>` : ''}
                    ${summary.themes && summary.themes.length > 0 ? `
                        <div class="themes">
                            <strong>主题：</strong>
                            ${summary.themes.map(t => `<span class="theme-tag">${t}</span>`).join('')}
                        </div>
                    ` : ''}
                </div>
            ` : ''}
            
            <!-- 剧情弧 -->
            ${storyArcs.length > 0 ? `
                <div class="timeline-section">
                    <h4>🎭 剧情发展</h4>
                    <div class="story-arcs-track">
                        ${storyArcs.map((arc, idx) => {
                            const pageRange = arc.page_range || {};
                            const arcEvents = (arc.event_ids || []).map(id => eventMap[id]).filter(e => e);
                            return `
                                <div class="story-arc-card ${arc.mood ? 'mood-' + arc.mood : ''}" data-arc-id="${arc.id || idx}">
                                    <div class="arc-header">
                                        <span class="arc-name">${arc.name || '未命名阶段'}</span>
                                        <span class="arc-pages">第 ${pageRange.start || '?'}-${pageRange.end || '?'} 页</span>
                                    </div>
                                    ${arc.description ? `<p class="arc-description">${arc.description}</p>` : ''}
                                    ${arc.mood ? `<span class="arc-mood">${arc.mood}</span>` : ''}
                                    ${arcEvents.length > 0 ? `
                                        <div class="arc-events">
                                            <strong>关键事件：</strong>
                                            <ul>
                                                ${arcEvents.slice(0, 5).map(e => `<li>${e.event || e.description || ''}</li>`).join('')}
                                                ${arcEvents.length > 5 ? `<li class="more">...还有 ${arcEvents.length - 5} 个事件</li>` : ''}
                                            </ul>
                                        </div>
                                    ` : ''}
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            ` : ''}
            
            <!-- 角色追踪 -->
            ${characters.length > 0 ? `
                <div class="timeline-section">
                    <h4>👥 主要角色</h4>
                    <div class="characters-grid">
                        ${characters.map(char => `
                            <div class="character-card">
                                <div class="character-name">${char.name || '未知角色'}</div>
                                ${char.description ? `<p class="character-desc">${char.description}</p>` : ''}
                                ${char.arc ? `<p class="character-arc"><strong>角色弧光：</strong>${char.arc}</p>` : ''}
                                ${char.first_appearance ? `<span class="first-appear">首次出场：第 ${char.first_appearance} 页</span>` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            <!-- 线索追踪 -->
            ${plotThreads.length > 0 ? `
                <div class="timeline-section">
                    <h4>🔗 伏笔与线索</h4>
                    <div class="plot-threads-list">
                        ${plotThreads.map(thread => `
                            <div class="plot-thread-item ${thread.status === '已解决' ? 'resolved' : 'pending'}">
                                <div class="thread-header">
                                    <span class="thread-name">${thread.name || '未命名线索'}</span>
                                    <span class="thread-status ${thread.status === '已解决' ? 'resolved' : ''}">${thread.status || '进行中'}</span>
                                </div>
                                ${thread.description ? `<p class="thread-desc">${thread.description}</p>` : ''}
                                ${thread.introduced_at ? `<span class="thread-intro">第 ${thread.introduced_at} 页引入</span>` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            <!-- 事件列表（折叠） -->
            ${events.length > 0 ? `
                <div class="timeline-section">
                    <h4 class="collapsible" onclick="toggleEventsSection(this)">
                        📊 全部事件 <span class="collapse-icon">▼</span>
                    </h4>
                    <div class="events-list-section collapsed">
                        ${events.map(event => {
                            const pageRange = event.page_range || {};
                            return `
                                <div class="event-item importance-${event.importance || 'normal'}">
                                    <span class="event-pages">第 ${pageRange.start || '?'}-${pageRange.end || '?'} 页</span>
                                    <span class="event-text">${event.event || event.description || ''}</span>
                                    ${event.involved_characters && event.involved_characters.length > 0 ? 
                                        `<span class="event-chars">${event.involved_characters.join(', ')}</span>` : ''}
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
    
    container.innerHTML = html;
}

function toggleEventsSection(header) {
    const section = header.nextElementSibling;
    const icon = header.querySelector('.collapse-icon');
    if (section.classList.contains('collapsed')) {
        section.classList.remove('collapsed');
        icon.textContent = '▲';
    } else {
        section.classList.add('collapsed');
        icon.textContent = '▼';
    }
}

// ==================== 重新生成功能 ====================

async function regenerateOverview() {
    if (!MangaInsight.currentBookId) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    if (!confirm('确定要重新生成故事概述吗？这可能需要一些时间。')) {
        return;
    }
    
    showToast('正在重新生成概述...', 'info');
    const storySummaryEl = document.getElementById('storySummary');
    if (storySummaryEl) storySummaryEl.innerHTML = '<div class="placeholder-text">正在生成中...</div>';
    
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/regenerate/overview`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.success) {
            showToast('概述生成完成', 'success');
            await loadOverviewData();
        } else {
            showToast(data.error || '生成失败', 'error');
        }
    } catch (error) {
        console.error('重新生成概述失败:', error);
        showToast('重新生成失败', 'error');
    }
}

async function regenerateTimeline() {
    if (!MangaInsight.currentBookId) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    showToast('正在生成时间线...', 'info');
    const container = document.getElementById('timelineContainer');
    if (container) container.innerHTML = '<div class="placeholder-text">正在生成中...</div>';
    
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/regenerate/timeline`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats || {};
            showToast(`时间线已生成: ${stats.total_events || 0} 个事件`, 'success');
            // 直接使用返回的数据渲染，避免再次请求
            renderTimeline(data);
        } else {
            showToast(data.error || '生成失败', 'error');
            if (container) container.innerHTML = '<div class="placeholder-text">生成时间线失败</div>';
        }
    } catch (error) {
        console.error('生成时间线失败:', error);
        showToast('生成时间线失败', 'error');
        if (container) container.innerHTML = '<div class="placeholder-text">生成时间线失败</div>';
    }
}

// ==================== 标签页切换 ====================
function switchTab(tabName) {
    // 更新按钮状态
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // 更新内容显示
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });
}

// ==================== 问答功能 ====================

async function rebuildEmbeddings() {
    if (!MangaInsight.currentBookId) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    if (!confirm('确定要重建向量索引吗？\n\n这将删除现有的向量数据并重新构建，可能需要一些时间。')) {
        return;
    }
    
    try {
        showLoading('正在重建向量索引...');
        
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/rebuild-embeddings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            let message = '向量索引重建完成';
            if (data.stats) {
                message += `\n页面向量: ${data.stats.pages_count || 0} 条`;
                if (data.stats.dialogues_count) {
                    message += `\n对话向量: ${data.stats.dialogues_count} 条`;
                }
            }
            showToast(message, 'success');
        } else {
            showToast('重建失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('重建向量索引失败:', error);
        showToast('重建向量索引失败', 'error');
    } finally {
        hideLoading();
    }
}

function askQuestion(question) {
    const questionInput = document.getElementById('questionInput');
    if (questionInput) {
        questionInput.value = question;
        sendQuestion();
    }
}

function handleQuestionKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendQuestion();
    }
}

// 当前问答模式：'precise' 或 'global'
let currentQAMode = 'precise';

// 切换问答模式
function setQAMode(mode) {
    currentQAMode = mode;
    
    // 更新按钮状态
    const preciseBtn = document.getElementById('qaPreciseMode');
    const globalBtn = document.getElementById('qaGlobalMode');
    const preciseOptions = document.getElementById('preciseModeOptions');
    const globalHint = document.getElementById('globalModeHint');
    
    if (mode === 'precise') {
        preciseBtn?.classList.add('active');
        globalBtn?.classList.remove('active');
        if (preciseOptions) preciseOptions.style.display = '';
        if (globalHint) globalHint.style.display = 'none';
    } else {
        preciseBtn?.classList.remove('active');
        globalBtn?.classList.add('active');
        if (preciseOptions) preciseOptions.style.display = 'none';
        if (globalHint) globalHint.style.display = '';
    }
    
    // 更新欢迎消息
    updateWelcomeMessage();
}

// 获取欢迎消息 HTML
function getWelcomeMessageHTML() {
    if (currentQAMode === 'global') {
        return `
            <div class="welcome-icon">🌐</div>
            <h3>全局模式</h3>
            <p>基于全文摘要回答问题，适合总结性问题</p>
            <div class="welcome-examples">
                <span class="example-tag" onclick="askQuestion('故事的主题是什么？')">故事的主题是什么？</span>
                <span class="example-tag" onclick="askQuestion('主角的性格有什么变化？')">主角的性格有什么变化？</span>
                <span class="example-tag" onclick="askQuestion('结局是怎样的？')">结局是怎样的？</span>
            </div>
        `;
    } else {
        return `
            <div class="welcome-icon">💬</div>
            <h3>智能问答</h3>
            <p>针对已分析的漫画内容提问，获取精准回答</p>
        `;
    }
}

// 更新欢迎消息
function updateWelcomeMessage() {
    const welcome = document.querySelector('#chatMessages .welcome-message');
    if (!welcome) return;
    welcome.innerHTML = getWelcomeMessageHTML();
}

async function sendQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    
    if (!question) return;
    if (!MangaInsight.currentBookId) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    // 清空输入
    input.value = '';
    
    // 清空之前的问答内容（单轮对话模式）
    clearChatMessages();
    
    // 添加用户消息
    addChatMessage('user', question);
    
    // 添加加载消息
    const loadingText = currentQAMode === 'global' ? '正在分析全文...' : '思考中...';
    addChatMessage('assistant', `<div class="loading-dots">${loadingText}</div>`);
    
    // 获取检索模式开关状态（仅精确模式使用）
    const useParentChild = document.getElementById('useParentChild')?.checked || false;
    const useReasoning = document.getElementById('useReasoning')?.checked || false;
    const useReranker = document.getElementById('useReranker')?.checked || false;
    const topK = parseInt(document.getElementById('topK')?.value) || 5;
    const threshold = parseFloat(document.getElementById('threshold')?.value) || 0;
    
    // 是否使用全局模式
    const useGlobalContext = currentQAMode === 'global';
    
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                use_parent_child: useParentChild,
                use_reasoning: useReasoning,
                use_reranker: useReranker,
                top_k: topK,
                threshold: threshold,
                use_global_context: useGlobalContext  // 新增：全局模式参数
            })
        });
        
        const data = await response.json();
        
        // 移除所有加载消息（确保不残留）
        removeLoadingMessages();
        
        if (data.success) {
            // 添加回答
            let answerHtml = data.answer;
            
            // 添加模式标识
            const modeLabel = data.mode === 'global' ? '🌐 全局模式' : '🎯 精确模式';
            answerHtml = `<div class="answer-mode-badge">${modeLabel}</div>` + answerHtml;
            
            // 添加引用（仅精确模式有引用）
            if (data.citations && data.citations.length > 0) {
                answerHtml += `
                    <div class="message-citations">
                        <span>📖 引用: </span>
                        ${data.citations.map(c => `
                            <span class="citation-item" onclick="selectPage(${c.page})">
                                第${c.page}页
                            </span>
                        `).join('')}
                    </div>
                `;
            }
            
            // 添加"保存为笔记"按钮
            const qaId = Date.now();
            answerHtml += `
                <button class="message-save-btn" id="saveBtn_${qaId}" onclick="saveCurrentQA(${qaId}, this)">
                    📝 保存为笔记
                </button>
            `;
            
            // 存储当前问答数据供保存使用
            window._currentQA = {
                id: qaId,
                question: question,
                answer: data.answer,
                citations: data.citations || [],
                mode: data.mode
            };
            
            addChatMessage('assistant', answerHtml);
        } else {
            addChatMessage('assistant', '抱歉，处理问题时出错: ' + data.error);
        }
    } catch (error) {
        console.error('发送问题失败:', error);
        removeLoadingMessages();
        addChatMessage('assistant', '抱歉，网络请求失败，请稍后重试。');
    }
}

function clearChatMessages() {
    const container = document.getElementById('chatMessages');
    if (!container) return;
    
    // 清空所有消息，恢复欢迎消息（根据当前模式显示不同内容）
    container.innerHTML = `<div class="welcome-message">${getWelcomeMessageHTML()}</div>`;
}

function removeLoadingMessages() {
    // 移除所有加载消息（确保不残留）
    const container = document.getElementById('chatMessages');
    container?.querySelectorAll('.loading-dots').forEach(el => {
        el.closest('.chat-message')?.remove();
    });
}

function addChatMessage(role, content) {
    const container = document.getElementById('chatMessages');
    
    // 移除欢迎消息
    const welcome = container.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    
    // 用户使用项目 logo，助手使用机器人 emoji
    const avatar = role === 'user' 
        ? '<img src="/pic/logo.png" alt="用户" class="avatar-img">'
        : '🤖';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${content}</div>
    `;
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
}

// ==================== 设置 ====================
async function loadConfig() {
    try {
        const response = await fetch('/api/manga-insight/config');
        const data = await response.json();
        
        if (data.success) {
            MangaInsight.config = data.config;
            populateSettingsForm(data.config);
        }
    } catch (error) {
        console.error('加载配置失败:', error);
    }
}

function populateSettingsForm(config) {
    // VLM
    if (config.vlm) {
        document.getElementById('vlmProvider').value = config.vlm.provider || 'gemini';
        document.getElementById('vlmApiKey').value = config.vlm.api_key || '';
        document.getElementById('vlmModel').value = config.vlm.model || '';
        document.getElementById('vlmBaseUrl').value = config.vlm.base_url || '';
        document.getElementById('vlmRpm').value = config.vlm.rpm_limit || 10;
        document.getElementById('vlmTemperature').value = config.vlm.temperature || 0.3;
        document.getElementById('vlmForceJson').checked = config.vlm.force_json || false;
        document.getElementById('vlmUseStream').checked = config.vlm.use_stream !== false;  // 默认开启
        document.getElementById('vlmImageMaxSize').value = config.vlm.image_max_size || 0;
        // 初始化 Base URL 显示状态
        const vlmBaseUrlGroup = document.getElementById('vlmBaseUrlGroup');
        vlmBaseUrlGroup.style.display = config.vlm.provider === 'custom' ? 'block' : 'none';
    }
    
    // LLM（对话模型）
    if (config.chat_llm) {
        const useSame = config.chat_llm.use_same_as_vlm !== false;  // 默认 true
        document.getElementById('llmUseSameAsVlm').checked = useSame;
        document.getElementById('llmCustomConfig').style.display = useSame ? 'none' : 'block';
        
        // 总是加载 LLM 配置（即使 useSame 为 true），以便用户切换时能看到正确的值
        // 如果配置有值则使用配置，否则从 VLM 复制
        const llmProvider = config.chat_llm.provider || (config.vlm?.provider || 'gemini');
        const llmApiKey = config.chat_llm.api_key || (config.vlm?.api_key || '');
        const llmModel = config.chat_llm.model || (config.vlm?.model || '');
        const llmBaseUrl = config.chat_llm.base_url || (config.vlm?.base_url || '');
        const llmUseStream = config.chat_llm.use_stream !== false;  // 默认 true
        
        document.getElementById('llmProvider').value = llmProvider;
        document.getElementById('llmApiKey').value = llmApiKey;
        document.getElementById('llmModel').value = llmModel;
        document.getElementById('llmBaseUrl').value = llmBaseUrl;
        document.getElementById('llmUseStream').checked = llmUseStream;
        
        // 初始化 Base URL 显示状态
        const llmBaseUrlGroup = document.getElementById('llmBaseUrlGroup');
        llmBaseUrlGroup.style.display = llmProvider === 'custom' ? 'block' : 'none';
    }
    
    // Embedding
    if (config.embedding) {
        document.getElementById('embeddingProvider').value = config.embedding.provider || 'openai';
        document.getElementById('embeddingApiKey').value = config.embedding.api_key || '';
        document.getElementById('embeddingModel').value = config.embedding.model || '';
        document.getElementById('embeddingBaseUrl').value = config.embedding.base_url || '';
        document.getElementById('embeddingRpmLimit').value = config.embedding.rpm_limit ?? 0;
        // 初始化 Base URL 显示状态（不调用 onEmbeddingProviderChange 避免覆盖模型值）
        const embeddingBaseUrlGroup = document.getElementById('embeddingBaseUrlGroup');
        embeddingBaseUrlGroup.style.display = config.embedding.provider === 'custom' ? 'block' : 'none';
    }
    
    // Reranker
    if (config.reranker) {
        document.getElementById('rerankerProvider').value = config.reranker.provider || 'jina';
        document.getElementById('rerankerApiKey').value = config.reranker.api_key || '';
        document.getElementById('rerankerModel').value = config.reranker.model || '';
        document.getElementById('rerankerBaseUrl').value = config.reranker.base_url || '';
        document.getElementById('rerankerTopK').value = config.reranker.top_k || 5;
        // 初始化 Base URL 显示状态（不调用 onRerankerProviderChange 避免覆盖模型值）
        const rerankerBaseUrlGroup = document.getElementById('rerankerBaseUrlGroup');
        rerankerBaseUrlGroup.style.display = config.reranker.provider === 'custom' ? 'block' : 'none';
    }
    
    // 批量分析设置
    const batch = (config.analysis && config.analysis.batch) ? config.analysis.batch : {};
    document.getElementById('pagesPerBatch').value = batch.pages_per_batch || 5;
    document.getElementById('contextBatchCount').value = batch.context_batch_count ?? 1;
    document.getElementById('architecturePreset').value = batch.architecture_preset || 'standard';
    
    // 加载自定义层级
    if (batch.custom_layers && batch.custom_layers.length > 0) {
        customLayers = batch.custom_layers.map(l => ({
            name: l.name,
            units: l.units_per_group,
            align: l.align_to_chapter
        }));
    }
    
    onArchitectureChange();
    updateBatchEstimate();
}

function openSettingsModal() {
    document.getElementById('settingsModal').classList.add('show');
}

function closeSettingsModal() {
    document.getElementById('settingsModal').classList.remove('show');
}

function switchSettingsTab(tabName) {
    document.querySelectorAll('.settings-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    document.querySelectorAll('.settings-content').forEach(content => {
        content.classList.toggle('active', content.id === `settings-${tabName}`);
    });
}

function onVlmProviderChange() {
    const provider = document.getElementById('vlmProvider').value;
    const baseUrlGroup = document.getElementById('vlmBaseUrlGroup');
    baseUrlGroup.style.display = provider === 'custom' ? 'block' : 'none';
    
    // 设置默认模型
    const defaultModels = {
        'gemini': 'gemini-2.0-flash',
        'openai': 'gpt-4o',
        'qwen': 'qwen-vl-max',
        'deepseek': 'deepseek-chat'
    };
    
    if (defaultModels[provider]) {
        document.getElementById('vlmModel').value = defaultModels[provider];
    }
}

function onLlmUseSameChange() {
    const useSame = document.getElementById('llmUseSameAsVlm').checked;
    const customConfig = document.getElementById('llmCustomConfig');
    customConfig.style.display = useSame ? 'none' : 'block';
    
    // 当取消勾选时，从 VLM 配置复制值到 LLM 字段（方便用户）
    if (!useSame) {
        const llmProvider = document.getElementById('llmProvider');
        const llmApiKey = document.getElementById('llmApiKey');
        const llmModel = document.getElementById('llmModel');
        const llmBaseUrl = document.getElementById('llmBaseUrl');
        
        // 如果 LLM 字段为空，则从 VLM 复制
        if (!llmApiKey.value) {
            const vlmProvider = document.getElementById('vlmProvider').value;
            const vlmApiKey = document.getElementById('vlmApiKey').value;
            const vlmModel = document.getElementById('vlmModel').value;
            const vlmBaseUrl = document.getElementById('vlmBaseUrl').value;
            
            llmProvider.value = vlmProvider;
            llmApiKey.value = vlmApiKey;
            llmModel.value = vlmModel;
            llmBaseUrl.value = vlmBaseUrl;
            
            // 更新 Base URL 显示状态
            const llmBaseUrlGroup = document.getElementById('llmBaseUrlGroup');
            llmBaseUrlGroup.style.display = vlmProvider === 'custom' ? 'block' : 'none';
        }
    }
}

function onLlmProviderChange() {
    const provider = document.getElementById('llmProvider').value;
    const baseUrlGroup = document.getElementById('llmBaseUrlGroup');
    baseUrlGroup.style.display = provider === 'custom' ? 'block' : 'none';
    
    // 设置默认模型
    const defaultModels = {
        'gemini': 'gemini-2.0-flash',
        'openai': 'gpt-4o-mini',
        'qwen': 'qwen-turbo',
        'deepseek': 'deepseek-chat'
    };
    
    if (defaultModels[provider]) {
        document.getElementById('llmModel').value = defaultModels[provider];
    }
}

function onEmbeddingProviderChange() {
    const provider = document.getElementById('embeddingProvider').value;
    const baseUrlGroup = document.getElementById('embeddingBaseUrlGroup');
    baseUrlGroup.style.display = provider === 'custom' ? 'block' : 'none';
    
    // 设置默认模型
    const defaultModels = {
        'openai': 'text-embedding-3-small',
        'siliconflow': 'BAAI/bge-m3'
    };
    
    if (defaultModels[provider]) {
        document.getElementById('embeddingModel').value = defaultModels[provider];
    }
}

function onRerankerProviderChange() {
    const provider = document.getElementById('rerankerProvider').value;
    const baseUrlGroup = document.getElementById('rerankerBaseUrlGroup');
    baseUrlGroup.style.display = provider === 'custom' ? 'block' : 'none';
    
    // 设置默认模型
    const defaultModels = {
        'jina': 'jina-reranker-v2-base-multilingual',
        'cohere': 'rerank-multilingual-v3.0',
        'siliconflow': 'BAAI/bge-reranker-v2-m3'
    };
    
    if (defaultModels[provider]) {
        document.getElementById('rerankerModel').value = defaultModels[provider];
    }
}

function updateBatchEstimate() {
    const pagesPerBatch = parseInt(document.getElementById('pagesPerBatch').value) || 5;
    const estPagesPerBatch = document.getElementById('estPagesPerBatch');
    if (estPagesPerBatch) estPagesPerBatch.textContent = pagesPerBatch;
}

// 架构预设数据
const ARCHITECTURE_PRESETS = {
    simple: {
        name: "简洁模式",
        description: "适合100页以内的短篇漫画",
        layers: [
            {name: "批量分析", units: 5, align: false},
            {name: "全书总结", units: 0, align: false}
        ]
    },
    standard: {
        name: "标准模式",
        description: "适合大多数漫画，平衡效果与速度",
        layers: [
            {name: "批量分析", units: 5, align: false},
            {name: "段落总结", units: 5, align: false},
            {name: "全书总结", units: 0, align: false}
        ]
    },
    chapter_based: {
        name: "章节模式",
        description: "适合有明确章节划分的漫画，会在章节边界处切分",
        layers: [
            {name: "批量分析", units: 5, align: true},
            {name: "章节总结", units: 0, align: true},
            {name: "全书总结", units: 0, align: false}
        ]
    },
    full: {
        name: "完整模式",
        description: "适合长篇连载，提供最详细的分层总结",
        layers: [
            {name: "批量分析", units: 5, align: false},
            {name: "小总结", units: 5, align: false},
            {name: "章节总结", units: 0, align: true},
            {name: "全书总结", units: 0, align: false}
        ]
    }
};

// 自定义层级数据
let customLayers = [
    {name: "批量分析", units: 5, align: false},
    {name: "段落总结", units: 5, align: false},
    {name: "全书总结", units: 0, align: false}
];

function onArchitectureChange() {
    const preset = document.getElementById('architecturePreset').value;
    const customEditor = document.getElementById('customLayersEditor');
    
    // 显示/隐藏自定义编辑器
    if (customEditor) {
        customEditor.style.display = preset === 'custom' ? 'block' : 'none';
    }
    
    if (preset === 'custom') {
        // 自定义模式
        updateCustomLayersUI();
        updateLayersPreview(customLayers);
        const descEl = document.getElementById('architectureDescription');
        if (descEl) descEl.textContent = '完全自定义层级架构，灵活配置分析流程';
    } else {
        // 预设模式
        const presetData = ARCHITECTURE_PRESETS[preset] || ARCHITECTURE_PRESETS.standard;
        const descEl = document.getElementById('architectureDescription');
        if (descEl) descEl.textContent = presetData.description;
        updateLayersPreview(presetData.layers);
    }
}

function updateLayersPreview(layers) {
    const layersList = document.getElementById('layersList');
    if (layersList && layers) {
        let html = '<ul style="margin: 0; padding-left: 20px;">';
        layers.forEach((layer, idx) => {
            const alignText = layer.align ? ' <span style="color: #6366f1; font-size: 12px;">(按章节对齐)</span>' : '';
            const unitsText = layer.units > 0 ? ` - 每${layer.units}个单元汇总` : ' - 汇总全部';
            html += `<li><strong>第${idx + 1}层 - ${layer.name}</strong>${unitsText}${alignText}</li>`;
        });
        html += '</ul>';
        layersList.innerHTML = html;
    }
}

function updateCustomLayersUI() {
    const container = document.getElementById('customLayersList');
    if (!container) return;
    
    let html = '';
    customLayers.forEach((layer, idx) => {
        const isFirst = idx === 0;
        const isLast = idx === customLayers.length - 1;
        const canDelete = !isFirst && !isLast && customLayers.length > 2;
        
        html += `
        <div class="custom-layer-item" style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px; padding: 8px; background: #f5f5f5; border-radius: 4px;">
            <span style="min-width: 50px; color: #666;">第${idx + 1}层</span>
            <input type="text" value="${layer.name}" onchange="updateCustomLayer(${idx}, 'name', this.value)" 
                   style="flex: 1; padding: 4px 8px;" ${isFirst || isLast ? 'disabled' : ''} placeholder="层级名称">
            <input type="number" value="${layer.units}" onchange="updateCustomLayer(${idx}, 'units', parseInt(this.value) || 0)" 
                   style="width: 60px; padding: 4px 8px;" min="1" max="20" ${isLast ? 'disabled' : ''} title="${isFirst ? '每批分析的页数' : '每组包含单元数（0=全部汇总）'}">
            <label style="display: flex; align-items: center; gap: 4px; font-size: 12px;">
                <input type="checkbox" ${layer.align ? 'checked' : ''} onchange="updateCustomLayer(${idx}, 'align', this.checked)">
                章节对齐
            </label>
            ${canDelete ? `<button type="button" onclick="removeCustomLayer(${idx})" style="padding: 4px 8px; background: #ef4444; color: white; border: none; border-radius: 4px; cursor: pointer;">删除</button>` : ''}
        </div>`;
    });
    
    container.innerHTML = html;
}

function updateCustomLayer(idx, field, value) {
    if (customLayers[idx]) {
        customLayers[idx][field] = value;
        
        // 如果是修改第一层的单元数，同步到"每批次分析页数"
        if (idx === 0 && field === 'units') {
            const pagesPerBatchInput = document.getElementById('pagesPerBatch');
            if (pagesPerBatchInput) {
                pagesPerBatchInput.value = value;
            }
        }
        
        updateLayersPreview(customLayers);
    }
}

function onPagesPerBatchChange(value) {
    const numValue = parseInt(value) || 5;
    // 同步到自定义层级的第一层
    if (customLayers.length > 0) {
        customLayers[0].units = numValue;
        // 如果当前是自定义模式，更新UI
        if (document.getElementById('architecturePreset').value === 'custom') {
            updateCustomLayersUI();
        }
        updateLayersPreview(customLayers);
    }
}

function addCustomLayer() {
    // 在倒数第二个位置插入新层级（最后一层是全书总结）
    const insertIdx = customLayers.length - 1;
    customLayers.splice(insertIdx, 0, {
        name: `汇总层${insertIdx}`,
        units: 5,
        align: false
    });
    updateCustomLayersUI();
    updateLayersPreview(customLayers);
}

function removeCustomLayer(idx) {
    if (idx > 0 && idx < customLayers.length - 1) {
        customLayers.splice(idx, 1);
        updateCustomLayersUI();
        updateLayersPreview(customLayers);
    }
}

async function saveSettings() {
    const llmUseSame = document.getElementById('llmUseSameAsVlm').checked;
    
    const config = {
        vlm: {
            provider: document.getElementById('vlmProvider').value,
            api_key: document.getElementById('vlmApiKey').value,
            model: document.getElementById('vlmModel').value,
            base_url: document.getElementById('vlmBaseUrl').value || null,
            rpm_limit: parseInt(document.getElementById('vlmRpm').value),
            temperature: parseFloat(document.getElementById('vlmTemperature').value),
            force_json: document.getElementById('vlmForceJson').checked,
            use_stream: document.getElementById('vlmUseStream').checked,
            image_max_size: parseInt(document.getElementById('vlmImageMaxSize').value) || 0
        },
        chat_llm: {
            use_same_as_vlm: llmUseSame,
            provider: llmUseSame ? '' : document.getElementById('llmProvider').value,
            api_key: llmUseSame ? '' : document.getElementById('llmApiKey').value,
            model: llmUseSame ? '' : document.getElementById('llmModel').value,
            base_url: llmUseSame ? '' : (document.getElementById('llmBaseUrl').value || null),
            use_stream: llmUseSame ? true : document.getElementById('llmUseStream').checked
        },
        embedding: {
            provider: document.getElementById('embeddingProvider').value,
            api_key: document.getElementById('embeddingApiKey').value,
            model: document.getElementById('embeddingModel').value,
            base_url: document.getElementById('embeddingBaseUrl').value || null,
            rpm_limit: Number.isNaN(parseInt(document.getElementById('embeddingRpmLimit').value)) ? 0 : parseInt(document.getElementById('embeddingRpmLimit').value)
        },
        reranker: {
            enabled: true,  // 由问答页面的开关控制是否使用
            provider: document.getElementById('rerankerProvider').value,
            api_key: document.getElementById('rerankerApiKey').value,
            model: document.getElementById('rerankerModel').value,
            base_url: document.getElementById('rerankerBaseUrl').value || null,
            top_k: parseInt(document.getElementById('rerankerTopK').value)
        },
        analysis: {
            batch: {
                pages_per_batch: parseInt(document.getElementById('pagesPerBatch').value) || 5,
                context_batch_count: parseInt(document.getElementById('contextBatchCount').value) || 1,
                architecture_preset: document.getElementById('architecturePreset').value || 'standard',
                custom_layers: document.getElementById('architecturePreset').value === 'custom' ? 
                    customLayers.map(l => ({
                        name: l.name,
                        units_per_group: l.units,
                        align_to_chapter: l.align
                    })) : []
            }
        },
        prompts: getPromptsConfig()
    };
    
    try {
        const response = await fetch('/api/manga-insight/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.success) {
            MangaInsight.config = config;
            showToast('设置已保存', 'success');
            closeSettingsModal();
        } else {
            showToast('保存失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('保存设置失败:', error);
        showToast('保存设置失败', 'error');
    }
}

async function testVlmConnection() {
    showLoading('测试连接...');
    
    try {
        const response = await fetch('/api/manga-insight/config/test/vlm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: document.getElementById('vlmProvider').value,
                api_key: document.getElementById('vlmApiKey').value,
                model: document.getElementById('vlmModel').value,
                base_url: document.getElementById('vlmBaseUrl').value
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('连接成功', 'success');
        } else {
            showToast('连接失败: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('测试失败', 'error');
    } finally {
        hideLoading();
    }
}

async function testEmbeddingConnection() {
    showLoading('测试连接...');
    
    try {
        const response = await fetch('/api/manga-insight/config/test/embedding', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: document.getElementById('embeddingProvider').value,
                api_key: document.getElementById('embeddingApiKey').value,
                model: document.getElementById('embeddingModel').value,
                base_url: document.getElementById('embeddingBaseUrl').value || null
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('连接成功', 'success');
        } else {
            showToast('连接失败: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('测试失败', 'error');
    } finally {
        hideLoading();
    }
}

// ==================== 工具函数 ====================
function showLoading(text = '加载中...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

function toggleTheme() {
    const body = document.body;
    const isDark = body.classList.toggle('dark-theme');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// 初始化主题
(function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
})();

// ==================== 笔记功能（重构版）====================

// 笔记状态管理
const NoteManager = {
    currentNoteId: null,      // 当前查看/编辑的笔记ID
    currentNoteType: 'text',  // 当前笔记类型: text | qa
    pendingQAData: null,      // 待保存的问答数据
    editMode: false           // 是否为编辑模式
};

// 获取所有笔记
function getNotes() {
    if (!MangaInsight.currentBookId) return [];
    const key = `manga_notes_${MangaInsight.currentBookId}`;
    return JSON.parse(localStorage.getItem(key) || '[]');
}

// 保存笔记列表
function saveNotes(notes) {
    if (!MangaInsight.currentBookId) return;
    const key = `manga_notes_${MangaInsight.currentBookId}`;
    localStorage.setItem(key, JSON.stringify(notes));
}

// 打开笔记模态框（添加新笔记）
function openNoteModal(type = 'text', qaData = null) {
    if (!MangaInsight.currentBookId) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    const modal = document.getElementById('noteModal');
    const modalTitle = document.getElementById('noteModalTitle');
    const typeSelector = document.getElementById('noteTypeSelector');
    
    // 重置状态
    NoteManager.currentNoteId = null;
    NoteManager.editMode = false;
    NoteManager.pendingQAData = qaData;
    NoteManager.currentNoteType = type;
    
    // 设置标题
    modalTitle.textContent = '📝 添加笔记';
    document.getElementById('saveNoteBtn').textContent = '保存笔记';
    
    // 清空表单
    clearNoteForm();
    
    // 如果是从问答保存，设置问答类型并隐藏类型选择
    if (qaData) {
        typeSelector.style.display = 'none';
        selectNoteType('qa');
        populateQAPreview(qaData);
    } else {
        typeSelector.style.display = 'flex';
        selectNoteType(type);
        // 如果有选中的页面，自动填入
        if (MangaInsight.selectedPage) {
            document.getElementById('notePageRef').value = MangaInsight.selectedPage;
        }
    }
    
    modal.classList.add('show');
}

// 关闭笔记模态框
function closeNoteModal() {
    const modal = document.getElementById('noteModal');
    modal.classList.remove('show');
    NoteManager.pendingQAData = null;
    NoteManager.editMode = false;
}

// 选择笔记类型
function selectNoteType(type) {
    // 如果选择问答类型但没有问答数据，提示用户
    if (type === 'qa' && !NoteManager.pendingQAData && !NoteManager.editMode) {
        showToast('问答笔记需要从智能问答中保存', 'info');
        return; // 不切换类型
    }
    
    NoteManager.currentNoteType = type;
    
    // 更新类型按钮状态
    document.querySelectorAll('.note-type-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === type);
    });
    
    // 显示对应表单
    document.getElementById('textNoteForm').style.display = type === 'text' ? 'flex' : 'none';
    document.getElementById('qaNoteForm').style.display = type === 'qa' ? 'block' : 'none';
}

// 填充问答预览
function populateQAPreview(qaData) {
    document.getElementById('qaPreviewQuestion').textContent = qaData.question || '';
    document.getElementById('qaPreviewAnswer').innerHTML = qaData.answer || '';
    
    const citationsSection = document.getElementById('qaPreviewCitationsSection');
    const citationsContainer = document.getElementById('qaPreviewCitations');
    
    if (qaData.citations && qaData.citations.length > 0) {
        citationsSection.style.display = 'block';
        citationsContainer.innerHTML = qaData.citations.map(c => 
            `<span class="qa-citation-badge">第${c.page}页</span>`
        ).join('');
    } else {
        citationsSection.style.display = 'none';
    }
}

// 清空笔记表单
function clearNoteForm() {
    document.getElementById('noteTitle').value = '';
    document.getElementById('noteContent').value = '';
    document.getElementById('notePageRef').value = '';
    document.getElementById('noteTags').value = '';
    document.getElementById('qaNoteTitle').value = '';
    document.getElementById('qaNoteComment').value = '';
    document.getElementById('qaPreviewQuestion').textContent = '';
    document.getElementById('qaPreviewAnswer').innerHTML = '';
    document.getElementById('qaPreviewCitations').innerHTML = '';
}

// 保存笔记
function saveNote() {
    const notes = getNotes();
    const now = new Date().toISOString();
    
    let noteData;
    
    if (NoteManager.currentNoteType === 'qa') {
        // 问答笔记
        if (!NoteManager.pendingQAData) {
            showToast('问答数据丢失，请重新从问答中保存', 'error');
            return;
        }
        
        const customTitle = document.getElementById('qaNoteTitle').value.trim();
        const comment = document.getElementById('qaNoteComment').value.trim();
        
        noteData = {
            id: NoteManager.editMode ? NoteManager.currentNoteId : Date.now(),
            type: 'qa',
            title: customTitle || NoteManager.pendingQAData.question.substring(0, 50),
            question: NoteManager.pendingQAData.question,
            answer: NoteManager.pendingQAData.answer,
            citations: NoteManager.pendingQAData.citations || [],
            comment: comment,
            pageNum: NoteManager.pendingQAData.citations?.[0]?.page || null,
            createdAt: NoteManager.editMode ? (notes.find(n => n.id === NoteManager.currentNoteId)?.createdAt || now) : now,
            updatedAt: now
        };
    } else {
        // 文本笔记
        const title = document.getElementById('noteTitle').value.trim();
        const content = document.getElementById('noteContent').value.trim();
        const pageRef = document.getElementById('notePageRef').value;
        const tagsInput = document.getElementById('noteTags').value.trim();
        
        if (!content) {
            showToast('请输入笔记内容', 'error');
            return;
        }
        
        const tags = tagsInput ? tagsInput.split(/[,，]/).map(t => t.trim()).filter(t => t) : [];
        
        noteData = {
            id: NoteManager.editMode ? NoteManager.currentNoteId : Date.now(),
            type: 'text',
            title: title || content.substring(0, 30),
            content: content,
            pageNum: pageRef ? parseInt(pageRef) : null,
            tags: tags,
            createdAt: NoteManager.editMode ? (notes.find(n => n.id === NoteManager.currentNoteId)?.createdAt || now) : now,
            updatedAt: now
        };
    }
    
    if (NoteManager.editMode) {
        // 更新现有笔记
        const index = notes.findIndex(n => n.id === NoteManager.currentNoteId);
        if (index !== -1) {
            notes[index] = noteData;
        }
        showToast('笔记已更新', 'success');
    } else {
        // 添加新笔记
        notes.unshift(noteData);
        showToast('笔记已保存', 'success');
    }
    
    saveNotes(notes);
    // 保持当前筛选状态
    const currentFilter = document.getElementById('notesFilter')?.value || 'all';
    renderNotes(currentFilter);
    closeNoteModal();
}

// 渲染笔记列表
function renderNotes(filter = 'all') {
    const container = document.getElementById('notesList');
    if (!container) return;
    
    let notes = getNotes();
    
    // 应用筛选
    if (filter !== 'all') {
        notes = notes.filter(n => n.type === filter);
    }
    
    if (notes.length === 0) {
        container.innerHTML = '<div class="placeholder-text">暂无笔记</div>';
        return;
    }
    
    container.innerHTML = notes.map(note => {
        const isQA = note.type === 'qa';
        const typeIcon = isQA ? '💬' : '✏️';
        const typeClass = isQA ? 'qa-note' : 'text-note';
        const preview = isQA 
            ? `Q: ${note.question?.substring(0, 60) || ''}...` 
            : (note.content?.substring(0, 80) || '');
        const title = note.title || (isQA ? note.question?.substring(0, 30) : note.content?.substring(0, 30)) || '无标题';
        const tags = note.tags || [];
        
        return `
        <div class="note-item ${typeClass}" onclick="openNoteDetail(${note.id})">
            <div class="note-header">
                <span class="note-type-badge">${typeIcon}</span>
                <span class="note-title">${escapeHtml(title)}</span>
            </div>
            <div class="note-preview">${escapeHtml(preview)}</div>
            ${tags.length > 0 ? `
                <div class="note-tags">
                    ${tags.slice(0, 3).map(tag => `<span class="note-tag">${escapeHtml(tag)}</span>`).join('')}
                    ${tags.length > 3 ? `<span class="note-tag">+${tags.length - 3}</span>` : ''}
                </div>
            ` : ''}
            <div class="note-meta">
                <span class="note-meta-left">
                    ${note.pageNum ? `<span class="note-page-ref" onclick="event.stopPropagation(); selectPage(${note.pageNum})">📄 第${note.pageNum}页</span>` : ''}
                    <span>${formatDate(note.createdAt)}</span>
                </span>
                <button class="btn-delete-note" onclick="event.stopPropagation(); deleteNote(${note.id})" title="删除">×</button>
            </div>
        </div>
        `;
    }).join('');
}

// 筛选笔记
function filterNotes() {
    const filter = document.getElementById('notesFilter')?.value || 'all';
    renderNotes(filter);
}

// 打开笔记详情
function openNoteDetail(noteId) {
    const notes = getNotes();
    const note = notes.find(n => n.id === noteId);
    if (!note) return;
    
    NoteManager.currentNoteId = noteId;
    
    const modal = document.getElementById('noteDetailModal');
    const titleEl = document.getElementById('noteDetailTitle');
    const contentEl = document.getElementById('noteDetailContent');
    
    const isQA = note.type === 'qa';
    const typeIcon = isQA ? '💬' : '✏️';
    titleEl.textContent = `${typeIcon} ${isQA ? '问答笔记' : '文本笔记'}`;
    
    let contentHtml = `
        <div class="note-detail-header">
            <span class="note-detail-type-icon">${typeIcon}</span>
            <div class="note-detail-info">
                <div class="note-detail-title">${escapeHtml(note.title || '无标题')}</div>
                <div class="note-detail-meta">
                    创建于 ${formatDateTime(note.createdAt)}
                    ${note.updatedAt && note.updatedAt !== note.createdAt ? ` · 更新于 ${formatDateTime(note.updatedAt)}` : ''}
                </div>
            </div>
        </div>
        <div class="note-detail-body">
    `;
    
    if (isQA) {
        // 问答笔记内容
        contentHtml += `
            <div class="note-detail-qa-section">
                <div class="note-detail-qa-label">❓ 问题</div>
                <div class="note-detail-qa-content">${escapeHtml(note.question || '')}</div>
            </div>
            <div class="note-detail-qa-section">
                <div class="note-detail-qa-label">💡 回答</div>
                <div class="note-detail-qa-content">${note.answer || ''}</div>
            </div>
        `;
        
        if (note.citations && note.citations.length > 0) {
            contentHtml += `
                <div class="note-detail-section">
                    <div class="note-detail-section-title">📖 引用页码</div>
                    <div class="note-detail-tags">
                        ${note.citations.map(c => `
                            <span class="note-detail-tag" style="cursor:pointer" onclick="selectPage(${c.page}); closeNoteDetailModal()">
                                第${c.page}页
                            </span>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        if (note.comment) {
            contentHtml += `
                <div class="note-detail-section">
                    <div class="note-detail-section-title">📝 补充说明</div>
                    <div class="note-detail-text">${escapeHtml(note.comment)}</div>
                </div>
            `;
        }
    } else {
        // 文本笔记内容
        contentHtml += `
            <div class="note-detail-section">
                <div class="note-detail-section-title">📝 内容</div>
                <div class="note-detail-text">${escapeHtml(note.content || '')}</div>
            </div>
        `;
        
        if (note.tags && note.tags.length > 0) {
            contentHtml += `
                <div class="note-detail-section">
                    <div class="note-detail-section-title">🏷️ 标签</div>
                    <div class="note-detail-tags">
                        ${note.tags.map(tag => `<span class="note-detail-tag">${escapeHtml(tag)}</span>`).join('')}
                    </div>
                </div>
            `;
        }
        
        if (note.pageNum) {
            contentHtml += `
                <div class="note-detail-section">
                    <div class="note-detail-section-title">📄 关联页面</div>
                    <span class="note-detail-page-link" onclick="selectPage(${note.pageNum}); closeNoteDetailModal()">
                        跳转到第 ${note.pageNum} 页
                    </span>
                </div>
            `;
        }
    }
    
    contentHtml += '</div>';
    contentEl.innerHTML = contentHtml;
    
    modal.classList.add('show');
}

// 关闭笔记详情模态框
function closeNoteDetailModal() {
    const modal = document.getElementById('noteDetailModal');
    modal.classList.remove('show');
}

// 编辑当前笔记
function editCurrentNote() {
    const notes = getNotes();
    const note = notes.find(n => n.id === NoteManager.currentNoteId);
    if (!note) return;
    
    closeNoteDetailModal();
    
    // 设置编辑模式
    NoteManager.editMode = true;
    NoteManager.currentNoteType = note.type;
    
    const modal = document.getElementById('noteModal');
    const modalTitle = document.getElementById('noteModalTitle');
    const typeSelector = document.getElementById('noteTypeSelector');
    
    modalTitle.textContent = '✏️ 编辑笔记';
    document.getElementById('saveNoteBtn').textContent = '保存修改';
    typeSelector.style.display = 'none';
    
    if (note.type === 'qa') {
        NoteManager.pendingQAData = {
            question: note.question,
            answer: note.answer,
            citations: note.citations
        };
        selectNoteType('qa');
        populateQAPreview(NoteManager.pendingQAData);
        document.getElementById('qaNoteTitle').value = note.title || '';
        document.getElementById('qaNoteComment').value = note.comment || '';
    } else {
        selectNoteType('text');
        document.getElementById('noteTitle').value = note.title || '';
        document.getElementById('noteContent').value = note.content || '';
        document.getElementById('notePageRef').value = note.pageNum || '';
        document.getElementById('noteTags').value = (note.tags || []).join(', ');
    }
    
    modal.classList.add('show');
}

// 删除当前查看的笔记
function deleteCurrentNote() {
    if (!NoteManager.currentNoteId) return;
    
    if (confirm('确定要删除这条笔记吗？')) {
        deleteNote(NoteManager.currentNoteId);
        closeNoteDetailModal();
    }
}

// 删除笔记
function deleteNote(noteId) {
    const notes = getNotes().filter(n => n.id !== noteId);
    saveNotes(notes);
    // 保持当前筛选状态
    const currentFilter = document.getElementById('notesFilter')?.value || 'all';
    renderNotes(currentFilter);
    showToast('笔记已删除', 'success');
}

// 从问答保存笔记（供问答功能调用）
function saveQAAsNote(question, answer, citations) {
    openNoteModal('qa', {
        question: question,
        answer: answer,
        citations: citations
    });
}

// 保存当前问答为笔记（从问答界面的按钮调用）
function saveCurrentQA(qaId, buttonElement) {
    if (!window._currentQA || window._currentQA.id !== qaId) {
        showToast('问答数据已过期，请重新提问', 'error');
        return;
    }
    
    const qa = window._currentQA;
    
    // 调用保存笔记函数
    saveQAAsNote(qa.question, qa.answer, qa.citations);
    
    // 更新按钮状态
    if (buttonElement) {
        buttonElement.classList.add('saved');
        buttonElement.innerHTML = '✅ 已保存';
        buttonElement.onclick = null;
    }
}

// 辅助函数：HTML转义
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 辅助函数：格式化日期
function formatDate(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
}

// 辅助函数：格式化日期时间
function formatDateTime(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleString('zh-CN', { 
        year: 'numeric', month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit'
    });
}

// 添加笔记（快捷入口）
function addNote() {
    openNoteModal('text');
}

// ==================== 内容导航树（章节+页面整合）====================
async function renderPagesTree(bookInfo) {
    const container = document.getElementById('pagesTree');
    const pageCountBadge = document.getElementById('pageCount');
    
    if (!container) return;
    
    const totalPages = bookInfo?.total_pages || 0;
    const chapters = bookInfo?.chapters || [];
    
    if (pageCountBadge) pageCountBadge.textContent = `${totalPages}页`;
    
    if (totalPages === 0) {
        container.innerHTML = '<div class="empty-hint">暂无页面</div>';
        return;
    }
    
    // 获取已分析页面列表
    let analyzedPages = [];
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/pages`);
        const data = await response.json();
        if (data.success) {
            analyzedPages = data.pages || [];
        }
    } catch (e) {}
    
    const analyzedSet = new Set(analyzedPages);
    MangaInsight.analyzedPages = analyzedSet;
    
    let html = '';
    
    if (chapters.length > 0) {
        // 有章节：按章节组织
        let pageOffset = 0;
        chapters.forEach((ch, idx) => {
            const chId = ch.id || ch.chapter_id || `ch_${idx + 1}`;
            const chapterPageCount = ch.page_count || ch.pages?.length || 0;
            const startPage = pageOffset + 1;
            const endPage = pageOffset + chapterPageCount;
            
            // 检查章节内页面分析状态
            let chapterAnalyzed = false;
            if (chapterPageCount > 0) {
                let analyzedInChapter = 0;
                for (let p = startPage; p <= endPage; p++) {
                    if (analyzedSet.has(p)) analyzedInChapter++;
                }
                chapterAnalyzed = analyzedInChapter === chapterPageCount;
            }
            
            html += `
            <div class="tree-chapter" data-chapter-id="${chId}">
                <div class="tree-chapter-header">
                    <span class="tree-expand-icon" onclick="toggleChapter('${chId}')">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5l8 7-8 7z"/></svg>
                    </span>
                    <div class="tree-chapter-info" onclick="toggleChapter('${chId}')">
                        <span class="tree-chapter-title">${ch.title || `第 ${idx + 1} 章`}</span>
                        <span class="tree-chapter-meta">${chapterPageCount}页</span>
                    </div>
                    <span class="tree-chapter-status ${chapterAnalyzed ? 'analyzed' : ''}"></span>
                    <button class="btn-reanalyze-chapter" onclick="event.stopPropagation(); reanalyzeChapter('${chId}')" title="重新分析此章节">
                        🔄
                    </button>
                </div>
                <div class="tree-pages-grid">
                    ${renderPagesGridHtml(startPage, endPage, analyzedSet)}
                </div>
            </div>`;
            
            pageOffset = endPage;
        });
        
        // 如果还有剩余页面（不属于任何章节）
        if (pageOffset < totalPages) {
            html += `
            <div class="tree-chapter" data-chapter-id="__uncategorized__">
                <div class="tree-chapter-header" onclick="toggleChapter('__uncategorized__')">
                    <span class="tree-expand-icon">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5l8 7-8 7z"/></svg>
                    </span>
                    <div class="tree-chapter-info">
                        <span class="tree-chapter-title">其他页面</span>
                        <span class="tree-chapter-meta">${totalPages - pageOffset}页</span>
                    </div>
                </div>
                <div class="tree-pages-grid">
                    ${renderPagesGridHtml(pageOffset + 1, totalPages, analyzedSet)}
                </div>
            </div>`;
        }
    } else {
        // 无章节：直接显示页面网格
        html = `<div class="tree-all-pages">${renderPagesGridHtml(1, Math.min(totalPages, 100), analyzedSet)}</div>`;
        
        if (totalPages > 100) {
            html += `
            <div class="tree-load-more">
                <button class="btn-load-more" onclick="loadMorePages()">加载更多 (还有 ${totalPages - 100} 页)</button>
            </div>`;
        }
    }
    
    container.innerHTML = html;
    
    // 默认展开第一个章节
    if (chapters.length > 0) {
        const firstChapter = container.querySelector('.tree-chapter');
        if (firstChapter) {
            firstChapter.classList.add('expanded');
        }
    }
}

function renderPagesGridHtml(startPage, endPage, analyzedSet) {
    let html = '';
    for (let i = startPage; i <= endPage; i++) {
        const isAnalyzed = analyzedSet.has(i);
        // 尝试获取缩略图 URL（如果有的话）
        const thumbUrl = `/api/manga-insight/${MangaInsight.currentBookId}/thumbnail/${i}`;
        html += `
        <div class="tree-page-item ${isAnalyzed ? 'analyzed' : ''}" 
             data-page="${i}"
             onclick="selectPage(${i})"
             oncontextmenu="showContextMenu(event, ${i})">
            <img class="tree-page-thumb" src="${thumbUrl}" alt="第${i}页" 
                 onerror="this.style.display='none'" loading="lazy">
            <span class="tree-page-num">${i}</span>
        </div>`;
    }
    return html;
}

function toggleChapter(chapterId) {
    const chapter = document.querySelector(`.tree-chapter[data-chapter-id="${chapterId}"]`);
    if (chapter) {
        chapter.classList.toggle('expanded');
    }
}


// 加载更多页面（无章节模式下的分页）
function loadMorePages() {
    const container = document.querySelector('.tree-all-pages');
    const loadMoreDiv = document.querySelector('.tree-load-more');
    if (!container || !MangaInsight.bookInfo) return;
    
    const totalPages = MangaInsight.bookInfo.total_pages || 0;
    const currentLoaded = container.querySelectorAll('.tree-page-item').length;
    const nextBatch = Math.min(currentLoaded + 100, totalPages);
    const analyzedSet = MangaInsight.analyzedPages || new Set();
    
    // 添加更多页面
    let html = '';
    for (let i = currentLoaded + 1; i <= nextBatch; i++) {
        const isAnalyzed = analyzedSet.has(i);
        const thumbUrl = `/api/manga-insight/${MangaInsight.currentBookId}/thumbnail/${i}`;
        html += `
        <div class="tree-page-item ${isAnalyzed ? 'analyzed' : ''}" 
             data-page="${i}"
             onclick="selectPage(${i})"
             oncontextmenu="showContextMenu(event, ${i})">
            <img class="tree-page-thumb" src="${thumbUrl}" alt="第${i}页" 
                 onerror="this.style.display='none'" loading="lazy">
            <span class="tree-page-num">${i}</span>
        </div>`;
    }
    container.insertAdjacentHTML('beforeend', html);
    
    // 更新或移除"加载更多"按钮
    if (nextBatch >= totalPages) {
        loadMoreDiv?.remove();
    } else {
        const remaining = totalPages - nextBatch;
        loadMoreDiv.innerHTML = `<button class="btn-load-more" onclick="loadMorePages()">加载更多 (还有 ${remaining} 页)</button>`;
    }
}

// ==================== 章节重新分析 ====================
async function reanalyzeChapter(chapterId) {
    if (!confirm(`确定要重新分析此章节吗？`)) return;
    
    try {
        showLoading('启动章节重新分析...');
        
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/reanalyze/chapter/${chapterId}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            MangaInsight.currentTaskId = data.task_id;
            showToast('章节分析已启动', 'success');
            startProgressPolling();
        } else {
            showToast('启动失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('重新分析章节失败:', error);
        showToast('重新分析失败', 'error');
    } finally {
        hideLoading();
    }
}

// ==================== 右键菜单 ====================
function showContextMenu(event, pageNum) {
    event.preventDefault();
    
    // 移除现有菜单
    const existingMenu = document.querySelector('.context-menu');
    if (existingMenu) existingMenu.remove();
    
    const menu = document.createElement('div');
    menu.className = 'context-menu';
    menu.innerHTML = `
        <div class="context-menu-item" onclick="viewPageDetail(${pageNum})">
            🔍 查看分析详情
        </div>
        <div class="context-menu-item" onclick="reanalyzePage(${pageNum})">
            🔄 重新分析此页
        </div>
        <div class="context-menu-item" onclick="addNoteForPage(${pageNum})">
            📝 添加笔记
        </div>
        <div class="context-menu-divider"></div>
        <div class="context-menu-item" onclick="exportPageAnalysis(${pageNum})">
            📤 导出分析结果
        </div>
    `;
    
    menu.style.left = event.pageX + 'px';
    menu.style.top = event.pageY + 'px';
    
    document.body.appendChild(menu);
    
    // 点击其他地方关闭菜单
    setTimeout(() => {
        document.addEventListener('click', closeContextMenu, { once: true });
    }, 0);
}

function closeContextMenu() {
    const menu = document.querySelector('.context-menu');
    if (menu) menu.remove();
}

function viewPageDetail(pageNum) {
    closeContextMenu();
    selectPage(pageNum);
}

function addNoteForPage(pageNum) {
    closeContextMenu();
    MangaInsight.selectedPage = pageNum;
    addNote();
}

// ==================== 导出功能 ====================
async function exportAnalysis() {
    if (!MangaInsight.currentBookId) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    try {
        showLoading('导出分析结果...');
        
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/export`);
        const data = await response.json();
        
        if (data.success) {
            // 下载 Markdown 文件
            const blob = new Blob([data.markdown], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${MangaInsight.currentBookId}_analysis.md`;
            a.click();
            URL.revokeObjectURL(url);
            
            showToast('导出成功', 'success');
        } else {
            showToast('导出失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('导出失败:', error);
        showToast('导出失败', 'error');
    } finally {
        hideLoading();
    }
}

async function exportPageAnalysis(pageNum) {
    closeContextMenu();
    
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/pages/${pageNum}`);
        const data = await response.json();
        
        if (data.success && data.analysis) {
            const json = JSON.stringify(data.analysis, null, 2);
            const blob = new Blob([json], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `page_${pageNum}_analysis.json`;
            a.click();
            URL.revokeObjectURL(url);
            
            showToast('页面分析已导出', 'success');
        }
    } catch (error) {
        showToast('导出失败', 'error');
    }
}

// ==================== 语录搜索 ====================
async function searchDialogues(query) {
    if (!MangaInsight.currentBookId || !query) return;
    
    try {
        const response = await fetch(`/api/manga-insight/${MangaInsight.currentBookId}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, type: 'dialogue' })
        });
        
        const data = await response.json();
        return data.results || [];
    } catch (error) {
        console.error('搜索失败:', error);
        return [];
    }
}


MangaInsight.afterBookLoaded = async function() {
    renderNotes();
}

// ==================== 获取模型功能 ====================
/**
 * 服务商配置映射
 */
const PROVIDER_CONFIG_MAP = {
    'vlm': { provider: 'vlmProvider', apiKey: 'vlmApiKey', baseUrl: 'vlmBaseUrl', model: 'vlmModel', select: 'vlmModelSelect', selectDiv: 'vlmModelSelectDiv', count: 'vlmModelCount', btn: 'vlmFetchModelsBtn' },
    'llm': { provider: 'llmProvider', apiKey: 'llmApiKey', baseUrl: 'llmBaseUrl', model: 'llmModel', select: 'llmModelSelect', selectDiv: 'llmModelSelectDiv', count: 'llmModelCount', btn: 'llmFetchModelsBtn' },
    'embedding': { provider: 'embeddingProvider', apiKey: 'embeddingApiKey', baseUrl: 'embeddingBaseUrl', model: 'embeddingModel', select: 'embeddingModelSelect', selectDiv: 'embeddingModelSelectDiv', count: 'embeddingModelCount', btn: 'embeddingFetchModelsBtn' }
};

/**
 * 获取模型列表
 */
async function fetchModelsFor(type) {
    const config = PROVIDER_CONFIG_MAP[type];
    if (!config) {
        console.error('未知的配置类型:', type);
        return;
    }
    
    const providerSelect = document.getElementById(config.provider);
    const apiKeyInput = document.getElementById(config.apiKey);
    const baseUrlInput = document.getElementById(config.baseUrl);
    const modelInput = document.getElementById(config.model);
    const modelSelect = document.getElementById(config.select);
    const modelSelectDiv = document.getElementById(config.selectDiv);
    const modelCount = document.getElementById(config.count);
    const fetchBtn = document.getElementById(config.btn);
    
    if (!providerSelect || !apiKeyInput) {
        console.error('获取模型: 找不到必要的元素');
        return;
    }
    
    let provider = providerSelect.value;
    const apiKey = apiKeyInput.value.trim();
    const baseUrl = baseUrlInput?.value.trim() || '';
    
    // 验证
    if (!apiKey) {
        showToast('请先填写 API Key', 'error');
        apiKeyInput.focus();
        return;
    }
    
    // 检查是否支持模型获取
    const supportedProviders = ['siliconflow', 'deepseek', 'volcano', 'gemini', 'qwen', 'openai', 'custom'];
    if (!supportedProviders.includes(provider)) {
        showToast(`${provider} 不支持自动获取模型列表`, 'warning');
        return;
    }
    
    // 自定义服务需要 base_url
    if (provider === 'custom' && !baseUrl) {
        showToast('自定义服务需要先填写 Base URL', 'error');
        baseUrlInput?.focus();
        return;
    }
    
    // 映射服务商名称
    const apiProvider = provider === 'custom' ? 'custom_openai' : provider;
    
    // 显示加载状态
    fetchBtn.disabled = true;
    const originalText = fetchBtn.textContent;
    fetchBtn.textContent = '获取中...';
    
    try {
        const response = await fetch('/api/fetch_models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: apiProvider,
                api_key: apiKey,
                base_url: baseUrl
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.models?.length > 0) {
            // 清空并填充模型列表
            modelSelect.innerHTML = '<option value="">-- 选择模型 --</option>';
            
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name || model.id;
                modelSelect.appendChild(option);
            });
            
            // 显示模型数量
            modelCount.textContent = `共 ${data.models.length} 个模型`;
            
            // 显示下拉框
            modelSelectDiv.style.display = 'flex';
            
            // 如果当前输入框有值，尝试在列表中选中
            const currentModel = modelInput?.value || '';
            if (currentModel) {
                modelSelect.value = currentModel;
            }
            
            showToast(`获取到 ${data.models.length} 个模型`, 'success');
        } else {
            showToast(data.message || '未获取到模型列表', 'warning');
            modelSelectDiv.style.display = 'none';
        }
    } catch (error) {
        console.error('获取模型列表失败:', error);
        showToast('获取模型列表失败: ' + error.message, 'error');
        modelSelectDiv.style.display = 'none';
    } finally {
        // 恢复按钮状态
        fetchBtn.disabled = false;
        fetchBtn.textContent = originalText;
    }
}

/**
 * 模型选择事件
 */
function onModelSelected(type) {
    const config = PROVIDER_CONFIG_MAP[type];
    if (!config) return;
    
    const modelSelect = document.getElementById(config.select);
    const modelInput = document.getElementById(config.model);
    
    if (modelSelect && modelInput && modelSelect.value) {
        modelInput.value = modelSelect.value;
    }
}

// ==================== 提示词管理功能 ====================

/**
 * 默认提示词模板
 */
const DEFAULT_PROMPTS = {
    batch_analysis: `你是一个专业的漫画分析师。请分析这组连续的 {page_count} 张漫画页面（第 {start_page} 页至第 {end_page} 页）。

【重要说明】
- 这是漫画原图（未翻译版本），请直接阅读原文内容
- 无论漫画原文是什么语言，你的所有输出内容必须使用中文
- 请特别关注页面之间的剧情连续性

请按以下 JSON 格式返回结果：
{
    "page_range": {
        "start": {start_page},
        "end": {end_page}
    },
    "pages": [
        {
            "page_number": <页码>,
            "page_summary": "<该页详细内容概括，包含场景描述、角色行为、重要对话和情节发展，100-200字>"
        }
    ],
    "batch_summary": "<这组页面的整体剧情概述，详细描述故事发展、角色互动和情感变化，200-400字>",
    "key_events": ["<这组页面中的关键事件>"],
    "continuity_notes": "<与上文的衔接、场景转换、剧情走向说明>"
}

注意：
1. 按正确的漫画阅读顺序分析
2. 重点关注剧情发展和角色互动
3. page_summary 要详细描述该页发生的事情
4. batch_summary 要完整概括这批页面的故事内容`,
    
    segment_summary: `【输出中文】基于以下批次的分析结果，生成一个连贯的段落总结。

请生成结构化的总结，JSON 格式：
{
    "summary": "<这段内容的主要剧情概括，3-5句话>",
    "key_events": ["<关键事件列表>"],
    "plot_progression": "<剧情进展描述>",
    "themes": ["<本段涉及的主题>"]
}

要求：
1. 整合各批次的信息，形成连贯叙述
2. 突出重要角色和关键事件
3. 注意剧情的因果关系`,
    
    chapter_summary: `【输出中文】基于以下内容，生成完整的章节总结。

请生成章节总结，JSON 格式：
{
    "summary": "<章节整体概述，5-8句话>",
    "main_plot": "<主要剧情线描述>",
    "key_events": ["<章节关键事件，按顺序>"],
    "themes": ["<本章主题>"],
    "atmosphere": "<整体氛围>"
}

要求：
1. 综合所有内容，形成完整的章节叙述
2. 理清人物关系和剧情脉络
3. 提炼章节主题和核心冲突`,
    
    qa_response: `【输出中文】根据分析结果回答用户问题，引用相关页面。
回答时请：
1. 基于提供的漫画内容回答
2. 引用具体页码作为依据
3. 如果问题超出已分析内容，请诚实说明`
};

/**
 * 提示词元数据
 */
const PROMPT_METADATA = {
    batch_analysis: { label: '📄 批量分析提示词', hint: '用于批量分析多个页面。支持变量：{page_count}, {start_page}, {end_page}' },
    segment_summary: { label: '📑 段落总结提示词', hint: '用于汇总多个批次的分析结果生成段落总结。' },
    chapter_summary: { label: '📖 章节总结提示词', hint: '用于生成章节级别的完整总结。' },
    qa_response: { label: '💬 问答响应提示词', hint: '用于回答用户关于漫画内容的问题。' }
};

// 当前编辑的提示词数据
let currentPrompts = {};
let savedPromptsLibrary = [];

/**
 * 初始化提示词编辑器
 */
async function initPromptsEditor() {
    // 加载已保存的提示词（等待完成）
    await loadPromptsFromConfig();
    await loadPromptsLibrary();
    
    // 初始化显示第一个提示词
    onPromptSelectorChange();
}

/**
 * 从配置加载提示词
 */
async function loadPromptsFromConfig() {
    try {
        const response = await fetch('/api/manga-insight/config');
        const data = await response.json();
        
        if (data.success && data.config.prompts) {
            currentPrompts = { ...data.config.prompts };
        }
    } catch (error) {
        console.error('加载提示词配置失败:', error);
    }
}

/**
 * 加载提示词库
 */
async function loadPromptsLibrary() {
    try {
        const response = await fetch('/api/manga-insight/prompts/library');
        const data = await response.json();
        
        if (data.success) {
            savedPromptsLibrary = data.library || [];
            renderPromptsLibrary();
        }
    } catch (error) {
        console.error('加载提示词库失败:', error);
        savedPromptsLibrary = [];
    }
}

// 记录当前编辑的提示词类型
let currentEditingPromptType = 'batch_analysis';

/**
 * 提示词选择器变更
 */
function onPromptSelectorChange() {
    const selector = document.getElementById('promptSelector');
    const editor = document.getElementById('promptEditor');
    const label = document.getElementById('currentPromptLabel');
    const hint = document.getElementById('promptHint');
    
    if (!selector || !editor) return;
    
    // 先保存当前编辑的内容
    if (currentEditingPromptType && editor.value) {
        currentPrompts[currentEditingPromptType] = editor.value;
    }
    
    const promptType = selector.value;
    const metadata = PROMPT_METADATA[promptType];
    
    if (!metadata) return;
    
    // 更新当前编辑类型
    currentEditingPromptType = promptType;
    
    // 更新标签和提示
    if (label) label.textContent = metadata.label;
    if (hint) hint.textContent = metadata.hint;
    
    // 加载提示词内容（优先使用用户自定义的，否则使用默认）
    const content = currentPrompts[promptType] || DEFAULT_PROMPTS[promptType] || '';
    editor.value = content;
}

/**
 * 保存当前编辑的提示词到临时存储
 */
function saveCurrentPromptToTemp() {
    const selector = document.getElementById('promptSelector');
    const editor = document.getElementById('promptEditor');
    
    if (!selector || !editor) return;
    
    const promptType = selector.value;
    if (promptType) {
        currentPrompts[promptType] = editor.value;
    }
}

/**
 * 重置当前提示词为默认值
 */
function resetCurrentPrompt() {
    const selector = document.getElementById('promptSelector');
    const editor = document.getElementById('promptEditor');
    
    if (!selector || !editor) return;
    
    const promptType = selector.value;
    const defaultContent = DEFAULT_PROMPTS[promptType] || '';
    
    if (confirm('确定要重置为默认提示词吗？当前编辑的内容将丢失。')) {
        editor.value = defaultContent;
        currentPrompts[promptType] = '';  // 清空自定义，使用默认
        showToast('已重置为默认提示词', 'success');
    }
}

/**
 * 复制提示词到剪贴板
 */
async function copyPromptToClipboard() {
    const editor = document.getElementById('promptEditor');
    
    if (!editor) return;
    
    try {
        await navigator.clipboard.writeText(editor.value);
        showToast('已复制到剪贴板', 'success');
    } catch (error) {
        showToast('复制失败', 'error');
    }
}

/**
 * 保存提示词到库
 */
async function savePromptToLibrary() {
    const editor = document.getElementById('promptEditor');
    const selector = document.getElementById('promptSelector');
    
    if (!editor || !selector) return;
    
    const content = editor.value.trim();
    if (!content) {
        showToast('提示词内容不能为空', 'error');
        return;
    }
    
    const name = prompt('请输入提示词名称：');
    if (!name || !name.trim()) return;
    
    const promptType = selector.value;
    const newPrompt = {
        id: Date.now().toString(),
        name: name,
        type: promptType,
        content: content,
        created_at: new Date().toISOString()
    };
    
    try {
        const response = await fetch('/api/manga-insight/prompts/library', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newPrompt)
        });
        
        const data = await response.json();
        
        if (data.success) {
            savedPromptsLibrary.push(newPrompt);
            renderPromptsLibrary();
            showToast('提示词已保存到库', 'success');
        } else {
            showToast('保存失败: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('保存失败', 'error');
    }
}

/**
 * 渲染提示词库列表
 */
function renderPromptsLibrary() {
    const container = document.getElementById('savedPromptsList');
    
    if (!container) return;
    
    if (!savedPromptsLibrary || savedPromptsLibrary.length === 0) {
        container.innerHTML = '<div class="placeholder-text">暂无保存的提示词</div>';
        return;
    }
    
    container.innerHTML = savedPromptsLibrary.map(prompt => `
        <div class="saved-prompt-item" onclick="loadPromptFromLibrary('${prompt.id}')">
            <span class="prompt-name">${escapeHtml(prompt.name)}</span>
            <span class="prompt-type">${PROMPT_METADATA[prompt.type]?.label || prompt.type}</span>
            <div class="prompt-actions">
                <button onclick="event.stopPropagation(); deletePromptFromLibrary('${prompt.id}')" title="删除">🗑️</button>
            </div>
        </div>
    `).join('');
}

/**
 * HTML 转义
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 从库加载提示词
 */
function loadPromptFromLibrary(id) {
    const prompt = savedPromptsLibrary.find(p => p.id === id);
    if (!prompt) return;
    
    const selector = document.getElementById('promptSelector');
    const editor = document.getElementById('promptEditor');
    
    if (!selector || !editor) return;
    
    // 切换到对应类型
    selector.value = prompt.type;
    onPromptSelectorChange();
    
    // 填入内容
    editor.value = prompt.content;
    currentPrompts[prompt.type] = prompt.content;
    
    showToast(`已加载提示词: ${prompt.name}`, 'success');
}

/**
 * 从库删除提示词
 */
async function deletePromptFromLibrary(id) {
    if (!confirm('确定要删除这个提示词吗？')) return;
    
    try {
        const response = await fetch(`/api/manga-insight/prompts/library/${id}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            savedPromptsLibrary = savedPromptsLibrary.filter(p => p.id !== id);
            renderPromptsLibrary();
            showToast('提示词已删除', 'success');
        } else {
            showToast('删除失败', 'error');
        }
    } catch (error) {
        showToast('删除失败', 'error');
    }
}

/**
 * 导出所有提示词
 */
function exportAllPrompts() {
    // 保存当前编辑的
    saveCurrentPromptToTemp();
    
    const exportData = {
        version: '1.0',
        exported_at: new Date().toISOString(),
        prompts: currentPrompts,
        library: savedPromptsLibrary
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `manga-insight-prompts-${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('提示词已导出', 'success');
}

/**
 * 触发导入文件选择
 */
function importPromptsFromFile() {
    const fileInput = document.getElementById('promptsFileInput');
    if (fileInput) {
        fileInput.click();
    }
}

/**
 * 处理导入文件
 */
async function handlePromptsFileImport(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
        const text = await file.text();
        const importData = JSON.parse(text);
        
        if (importData.prompts) {
            currentPrompts = { ...currentPrompts, ...importData.prompts };
        }
        
        if (importData.library && Array.isArray(importData.library)) {
            // 合并库，避免重复
            const existingIds = new Set(savedPromptsLibrary.map(p => p.id));
            for (const prompt of importData.library) {
                if (!existingIds.has(prompt.id)) {
                    savedPromptsLibrary.push(prompt);
                }
            }
            
            // 保存到服务器
            await fetch('/api/manga-insight/prompts/library/import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ library: savedPromptsLibrary })
            });
        }
        
        // 刷新显示
        onPromptSelectorChange();
        renderPromptsLibrary();
        
        showToast('提示词导入成功', 'success');
    } catch (error) {
        console.error('导入失败:', error);
        showToast('导入失败，请检查文件格式', 'error');
    }
    
    // 清空文件输入
    event.target.value = '';
}

/**
 * 获取当前提示词配置（用于保存设置）
 */
function getPromptsConfig() {
    // 保存当前编辑的
    saveCurrentPromptToTemp();
    return currentPrompts;
}

/**
 * 打开图片预览
 */
function openImagePreview(imageUrl) {
    // 创建预览模态框
    const modal = document.createElement('div');
    modal.className = 'image-preview-modal';
    modal.innerHTML = `
        <button class="image-preview-close" onclick="closeImagePreview()">&times;</button>
        <img src="${imageUrl}" alt="页面预览">
    `;
    
    // 点击背景关闭
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeImagePreview();
        }
    });
    
    // ESC 键关闭
    const escHandler = (e) => {
        if (e.key === 'Escape') {
            closeImagePreview();
            document.removeEventListener('keydown', escHandler);
        }
    };
    document.addEventListener('keydown', escHandler);
    
    document.body.appendChild(modal);
    document.body.style.overflow = 'hidden';
}

/**
 * 关闭图片预览
 */
function closeImagePreview() {
    const modal = document.querySelector('.image-preview-modal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = '';
    }
}
