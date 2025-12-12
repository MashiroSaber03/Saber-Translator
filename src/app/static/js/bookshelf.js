/**
 * 书架页面 JavaScript
 * 处理书籍和章节的 CRUD 操作，包括搜索、标签、批量操作和拖拽排序
 */

// ==================== 全局状态 ====================
let currentBookId = null;
let currentChapterId = null;
let confirmCallback = null;
let allTags = [];  // 所有标签缓存
let currentBookTags = [];  // 当前编辑书籍的标签
let selectedBooks = new Set();  // 批量选中的书籍
let batchMode = false;  // 是否处于批量模式
let currentSearchQuery = '';  // 当前搜索关键词
let currentFilterTags = [];  // 当前筛选的标签
let draggedChapter = null;  // 当前拖拽的章节

// 当从翻译页面返回时自动刷新数据
window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
        loadBooks();
        loadTags();
        if (currentBookId && document.getElementById('bookDetailModal').classList.contains('active')) {
            openBookDetail(currentBookId);
        }
    }
});

// ==================== API 请求 ====================

async function apiRequest(url, method = 'GET', data = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' }
    };
    if (data) options.body = JSON.stringify(data);
    
    try {
        const response = await fetch(url, options);
        const result = await response.json();
        if (!result.success) throw new Error(result.error || '操作失败');
        return result;
    } catch (error) {
        console.error('API请求失败:', error);
        throw error;
    }
}

// ==================== 搜索和筛选 ====================

async function loadBooks() {
    try {
        let url = '/api/bookshelf/books';
        const params = new URLSearchParams();
        
        if (currentSearchQuery) params.append('search', currentSearchQuery);
        if (currentFilterTags.length > 0) params.append('tags', currentFilterTags.join(','));
        
        if (params.toString()) url += '?' + params.toString();
        
        const result = await apiRequest(url);
        renderBooks(result.books);
    } catch (error) {
        showToast('加载书籍失败: ' + error.message, 'error');
    }
}

function handleSearch() {
    currentSearchQuery = document.getElementById('searchInput').value.trim();
    loadBooks();
}

function clearSearch() {
    document.getElementById('searchInput').value = '';
    document.getElementById('clearSearchBtn').style.display = 'none';
    currentSearchQuery = '';
    loadBooks();
}

function toggleTagFilter(tagName) {
    const index = currentFilterTags.indexOf(tagName);
    if (index > -1) {
        currentFilterTags.splice(index, 1);
    } else {
        currentFilterTags.push(tagName);
    }
    renderTagChips();
    loadBooks();
}

function renderTagChips() {
    const container = document.getElementById('tagChips');
    if (!container) return;
    
    container.innerHTML = allTags.map(tag => `
        <span class="tag-chip ${currentFilterTags.includes(tag.name) ? 'active' : ''}" 
              style="--tag-color: ${tag.color}"
              onclick="toggleTagFilter('${tag.name}')">
            ${tag.name}
            ${tag.book_count ? `<span class="tag-count">${tag.book_count}</span>` : ''}
        </span>
    `).join('');
    
    if (allTags.length === 0) {
        container.innerHTML = '<span class="no-tags">暂无标签</span>';
    }
}

// ==================== 书籍渲染 ====================

function renderBooks(books) {
    const grid = document.getElementById('booksGrid');
    const emptyState = document.getElementById('emptyState');
    
    if (!books || books.length === 0) {
        grid.style.display = 'none';
        emptyState.style.display = 'flex';
        return;
    }
    
    grid.style.display = 'grid';
    emptyState.style.display = 'none';
    
    grid.innerHTML = books.map(book => {
        const tagsHtml = (book.tags || []).slice(0, 3).map(tag => {
            const tagInfo = allTags.find(t => t.name === tag) || { color: '#667eea' };
            return `<span class="book-tag" style="background: ${tagInfo.color}">${tag}</span>`;
        }).join('');
        
        return `
        <div class="book-card ${batchMode ? 'batch-mode' : ''} ${selectedBooks.has(book.id) ? 'selected' : ''}" 
             data-book-id="${book.id}"
             onclick="${batchMode ? `toggleBookSelection('${book.id}', event)` : `openBookDetail('${book.id}')`}"
             oncontextmenu="handleBookContextMenu('${book.id}', event)">
            ${batchMode ? `
                <div class="book-checkbox">
                    <input type="checkbox" ${selectedBooks.has(book.id) ? 'checked' : ''} onclick="event.stopPropagation()">
                </div>
            ` : ''}
            <div class="book-cover">
                ${book.cover 
                    ? `<img src="${book.cover}" alt="${book.title}" onerror="this.parentElement.innerHTML='<div class=\\'book-cover-placeholder\\'>📖</div>'">`
                    : '<div class="book-cover-placeholder">📖</div>'
                }
            </div>
            <div class="book-info">
                <h3 class="book-title" title="${book.title}">${book.title}</h3>
                <p class="book-meta">${book.chapter_count || 0} 章节</p>
                ${tagsHtml ? `<div class="book-tags">${tagsHtml}</div>` : ''}
            </div>
        </div>
    `}).join('');
}

function showCreateBookModal() {
    document.getElementById('bookModalTitle').textContent = '新建书籍';
    document.getElementById('bookId').value = '';
    document.getElementById('bookTitle').value = '';
    document.getElementById('coverPreviewImg').style.display = 'none';
    document.getElementById('coverPlaceholder').style.display = 'flex';
    currentBookTags = [];
    renderBookSelectedTags();
    document.getElementById('bookModal').classList.add('active');
}

function showEditBookModal(book) {
    document.getElementById('bookModalTitle').textContent = '编辑书籍';
    document.getElementById('bookId').value = book.id;
    document.getElementById('bookTitle').value = book.title;
    
    if (book.cover) {
        document.getElementById('coverPreviewImg').src = book.cover;
        document.getElementById('coverPreviewImg').style.display = 'block';
        document.getElementById('coverPlaceholder').style.display = 'none';
    } else {
        document.getElementById('coverPreviewImg').style.display = 'none';
        document.getElementById('coverPlaceholder').style.display = 'flex';
    }
    
    currentBookTags = book.tags || [];
    renderBookSelectedTags();
    document.getElementById('bookModal').classList.add('active');
}

function closeBookModal() {
    document.getElementById('bookModal').classList.remove('active');
    currentBookTags = [];
}

async function saveBook() {
    const bookId = document.getElementById('bookId').value;
    const title = document.getElementById('bookTitle').value.trim();
    
    if (!title) {
        showToast('请输入书籍名称', 'error');
        return;
    }
    
    // 获取封面数据
    const coverImg = document.getElementById('coverPreviewImg');
    let coverData = null;
    if (coverImg.style.display !== 'none' && coverImg.src && !coverImg.src.startsWith('/api')) {
        coverData = coverImg.src;
    }
    
    try {
        if (bookId) {
            // 更新书籍
            await apiRequest(`/api/bookshelf/books/${bookId}`, 'PUT', { title, cover: coverData, tags: currentBookTags });
            showToast('书籍更新成功', 'success');
        } else {
            // 创建书籍
            await apiRequest('/api/bookshelf/books', 'POST', { title, cover: coverData, tags: currentBookTags });
            showToast('书籍创建成功', 'success');
        }
        
        closeBookModal();
        loadBooks();
        loadTags();
        
        // 如果在书籍详情中，刷新详情
        if (currentBookId === bookId) {
            openBookDetail(bookId);
        }
    } catch (error) {
        showToast('保存失败: ' + error.message, 'error');
    }
}

async function openBookDetail(bookId) {
    currentBookId = bookId;
    
    try {
        const result = await apiRequest(`/api/bookshelf/books/${bookId}`);
        const book = result.book;
        
        // 填充书籍信息
        document.getElementById('detailBookTitle').textContent = book.title;
        document.getElementById('detailChapterCount').textContent = book.chapters?.length || 0;
        document.getElementById('detailCreatedAt').textContent = formatDate(book.created_at);
        document.getElementById('detailUpdatedAt').textContent = formatDate(book.updated_at);
        
        // 渲染书籍标签
        renderDetailBookTags(book.tags || []);
        
        // 设置封面
        const coverImg = document.getElementById('detailCoverImg');
        if (book.cover) {
            coverImg.src = book.cover;
            coverImg.style.display = 'block';
        } else {
            coverImg.style.display = 'none';
        }
        
        // 渲染章节列表
        renderChapters(book.chapters || []);
        
        // 显示模态框
        document.getElementById('bookDetailModal').classList.add('active');
    } catch (error) {
        showToast('加载书籍详情失败: ' + error.message, 'error');
    }
}

// 渲染书籍详情中的标签
function renderDetailBookTags(tags) {
    const container = document.getElementById('detailBookTags');
    if (!container) return;
    
    if (!tags || tags.length === 0) {
        container.innerHTML = '<span class="no-tags-hint">暂无标签</span>';
        return;
    }
    
    container.innerHTML = tags.map(tag => {
        const tagInfo = allTags.find(t => t.name === tag) || { color: '#667eea' };
        return `
            <span class="detail-tag" style="background: ${tagInfo.color}">
                ${tag}
                <span class="remove-detail-tag" onclick="removeTagFromCurrentBook('${tag}')">&times;</span>
            </span>
        `;
    }).join('');
}

// 从当前书籍移除标签
async function removeTagFromCurrentBook(tagName) {
    if (!currentBookId) return;
    
    try {
        const result = await apiRequest(`/api/bookshelf/books/${currentBookId}`);
        const book = result.book;
        const newTags = (book.tags || []).filter(t => t !== tagName);
        
        await apiRequest(`/api/bookshelf/books/${currentBookId}`, 'PUT', { tags: newTags });
        showToast('标签已移除', 'success');
        
        // 刷新显示
        renderDetailBookTags(newTags);
        loadBooks();
        loadTags();
    } catch (error) {
        showToast('移除标签失败: ' + error.message, 'error');
    }
}

function closeBookDetailModal() {
    document.getElementById('bookDetailModal').classList.remove('active');
    currentBookId = null;
}

function goToInsight() {
    if (!currentBookId) return;
    window.location.href = `/insight?book=${currentBookId}`;
}

async function editCurrentBook() {
    if (!currentBookId) return;
    
    try {
        const result = await apiRequest(`/api/bookshelf/books/${currentBookId}`);
        closeBookDetailModal();
        showEditBookModal(result.book);
    } catch (error) {
        showToast('加载书籍信息失败', 'error');
    }
}

function deleteCurrentBook() {
    if (!currentBookId) return;
    
    showConfirmModal('确定要删除这本书吗？所有章节和翻译数据都将被删除，此操作不可恢复。', async () => {
        try {
            await apiRequest(`/api/bookshelf/books/${currentBookId}`, 'DELETE');
            showToast('书籍删除成功', 'success');
            closeBookDetailModal();
            loadBooks();
        } catch (error) {
            showToast('删除失败: ' + error.message, 'error');
        }
    });
}

// ==================== 章节操作 ====================

function renderChapters(chapters) {
    const list = document.getElementById('chaptersList');
    const emptyState = document.getElementById('chaptersEmpty');
    
    if (!chapters || chapters.length === 0) {
        list.style.display = 'none';
        emptyState.style.display = 'block';
        return;
    }
    
    list.style.display = 'flex';
    emptyState.style.display = 'none';
    
    list.innerHTML = chapters.map((chapter, index) => `
        <div class="chapter-item" data-id="${chapter.id}" draggable="true">
            <div class="chapter-drag-handle" title="拖拽排序">⋮⋮</div>
            <div class="chapter-info">
                <span class="chapter-order">#${index + 1}</span>
                <span class="chapter-title">${chapter.title}</span>
                <span class="chapter-meta">${chapter.image_count || 0} 张图片</span>
            </div>
            <div class="chapter-actions">
                <button class="chapter-action-btn chapter-enter-btn" onclick="enterChapter('${chapter.id}', event)">
                    进入翻译
                </button>
                <button class="chapter-action-btn chapter-read-btn" onclick="readChapter('${chapter.id}', event)" ${(chapter.image_count || 0) === 0 ? 'disabled' : ''}>
                    进入阅读
                </button>
                <button class="chapter-action-btn" onclick="editChapter('${chapter.id}', '${chapter.title}', event)">
                    编辑
                </button>
                <button class="chapter-action-btn danger" onclick="deleteChapter('${chapter.id}', event)">
                    删除
                </button>
            </div>
        </div>
    `).join('');
    
    // 初始化拖拽排序
    initChapterDragDrop();
}

function showCreateChapterModal() {
    document.getElementById('chapterModalTitle').textContent = '新建章节';
    document.getElementById('chapterId').value = '';
    document.getElementById('chapterTitle').value = '';
    document.getElementById('chapterModal').classList.add('active');
}

function editChapter(chapterId, title, event) {
    event.stopPropagation();
    currentChapterId = chapterId;
    document.getElementById('chapterModalTitle').textContent = '编辑章节';
    document.getElementById('chapterId').value = chapterId;
    document.getElementById('chapterTitle').value = title;
    document.getElementById('chapterModal').classList.add('active');
}

function closeChapterModal() {
    document.getElementById('chapterModal').classList.remove('active');
    currentChapterId = null;
}

async function saveChapter() {
    const chapterId = document.getElementById('chapterId').value;
    const title = document.getElementById('chapterTitle').value.trim();
    
    if (!title) {
        showToast('请输入章节名称', 'error');
        return;
    }
    
    if (!currentBookId) {
        showToast('未选择书籍', 'error');
        return;
    }
    
    try {
        if (chapterId) {
            // 更新章节
            await apiRequest(`/api/bookshelf/books/${currentBookId}/chapters/${chapterId}`, 'PUT', { title });
            showToast('章节更新成功', 'success');
        } else {
            // 创建章节
            await apiRequest(`/api/bookshelf/books/${currentBookId}/chapters`, 'POST', { title });
            showToast('章节创建成功', 'success');
        }
        
        closeChapterModal();
        // 刷新书籍详情
        openBookDetail(currentBookId);
    } catch (error) {
        showToast('保存失败: ' + error.message, 'error');
    }
}

function deleteChapter(chapterId, event) {
    event.stopPropagation();
    
    showConfirmModal('确定要删除这个章节吗？章节内的所有翻译数据都将被删除。', async () => {
        try {
            await apiRequest(`/api/bookshelf/books/${currentBookId}/chapters/${chapterId}`, 'DELETE');
            showToast('章节删除成功', 'success');
            openBookDetail(currentBookId);
        } catch (error) {
            showToast('删除失败: ' + error.message, 'error');
        }
    });
}

function enterChapter(chapterId, event) {
    event.stopPropagation();
    // 跳转到翻译页面，传递书籍和章节ID
    window.location.href = `/translate?book=${currentBookId}&chapter=${chapterId}`;
}

function readChapter(chapterId, event) {
    event.stopPropagation();
    // 跳转到阅读页面，传递书籍和章节ID
    window.location.href = `/reader?book=${currentBookId}&chapter=${chapterId}`;
}

// ==================== 确认模态框 ====================

function showConfirmModal(message, callback) {
    document.getElementById('confirmMessage').textContent = message;
    confirmCallback = callback;
    document.getElementById('confirmModal').classList.add('active');
}

function closeConfirmModal() {
    document.getElementById('confirmModal').classList.remove('active');
    confirmCallback = null;
}

// ==================== 封面上传 ====================

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('coverUploadArea');
    const coverInput = document.getElementById('coverInput');
    
    if (uploadArea && coverInput) {
        // 点击上传
        uploadArea.addEventListener('click', () => {
            coverInput.click();
        });
        
        // 文件选择
        coverInput.addEventListener('change', handleCoverSelect);
        
        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleCoverFile(files[0]);
            }
        });
    }
    
    // 确认删除按钮
    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', () => {
            if (confirmCallback) {
                confirmCallback();
            }
            closeConfirmModal();
        });
    }
    
    // 新建书籍按钮
    const createBookBtn = document.getElementById('createBookBtn');
    if (createBookBtn) {
        createBookBtn.addEventListener('click', showCreateBookModal);
    }
});

function handleCoverSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleCoverFile(file);
    }
}

function handleCoverFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('请选择图片文件', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById('coverPreviewImg');
        const placeholder = document.getElementById('coverPlaceholder');
        img.src = e.target.result;
        img.style.display = 'block';
        placeholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// ==================== 工具函数 ====================

function formatDate(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleDateString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast ' + type;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// ==================== 标签管理 ====================

async function loadTags() {
    try {
        const result = await apiRequest('/api/bookshelf/tags');
        allTags = result.tags || [];
        renderTagChips();
    } catch (error) {
        console.error('加载标签失败:', error);
    }
}

function renderBookSelectedTags() {
    const container = document.getElementById('bookSelectedTags');
    if (!container) return;
    
    container.innerHTML = currentBookTags.map(tag => {
        const tagInfo = allTags.find(t => t.name === tag) || { color: '#667eea' };
        return `
            <span class="selected-tag" style="background: ${tagInfo.color}">
                ${tag}
                <span class="remove-tag" onclick="removeTagFromBook('${tag}')">&times;</span>
            </span>
        `;
    }).join('');
}

function showTagSuggestions() {
    const input = document.getElementById('tagInput');
    const container = document.getElementById('tagSuggestions');
    if (!input || !container) return;
    
    const query = input.value.trim().toLowerCase();
    const availableTags = allTags.filter(t => 
        !currentBookTags.includes(t.name) && 
        (query === '' || t.name.toLowerCase().includes(query))
    );
    
    if (availableTags.length === 0 && query === '') {
        container.style.display = 'none';
        return;
    }
    
    container.innerHTML = availableTags.map(tag => `
        <div class="tag-suggestion" style="--tag-color: ${tag.color}" onclick="addTagToBook('${tag.name}')">
            <span class="tag-color" style="background: ${tag.color}"></span>
            ${tag.name}
        </div>
    `).join('');
    
    if (query && !allTags.some(t => t.name.toLowerCase() === query)) {
        container.innerHTML += `
            <div class="tag-suggestion new-tag" onclick="addTagToBook('${input.value.trim()}')">
                <span class="tag-icon">+</span>
                创建标签 "${input.value.trim()}"
            </div>
        `;
    }
    
    container.style.display = 'block';
}

function addTagToBook(tagName) {
    if (!tagName || currentBookTags.includes(tagName)) return;
    
    currentBookTags.push(tagName);
    renderBookSelectedTags();
    
    const input = document.getElementById('tagInput');
    if (input) input.value = '';
    document.getElementById('tagSuggestions').style.display = 'none';
    
    // 如果是新标签，自动创建
    if (!allTags.some(t => t.name === tagName)) {
        apiRequest('/api/bookshelf/tags', 'POST', { name: tagName })
            .then(() => loadTags())
            .catch(err => console.error('创建标签失败:', err));
    }
}

function removeTagFromBook(tagName) {
    currentBookTags = currentBookTags.filter(t => t !== tagName);
    renderBookSelectedTags();
}

// 标签管理模态框
function showTagManageModal() {
    renderTagManageList();
    document.getElementById('tagManageModal').classList.add('active');
}

function closeTagManageModal() {
    document.getElementById('tagManageModal').classList.remove('active');
}

function renderTagManageList() {
    const container = document.getElementById('tagManageList');
    if (!container) return;
    
    if (allTags.length === 0) {
        container.innerHTML = '<p class="no-tags-message">暂无标签，请在上方添加</p>';
        return;
    }
    
    container.innerHTML = allTags.map(tag => `
        <div class="tag-manage-item" data-tag="${tag.name}">
            <span class="tag-color-dot" style="background: ${tag.color}"></span>
            <span class="tag-name">${tag.name}</span>
            <span class="tag-book-count">${tag.book_count || 0} 本</span>
            <button class="tag-edit-btn" onclick="editTag('${tag.name}', '${tag.color}')">编辑</button>
            <button class="tag-delete-btn" onclick="deleteTag('${tag.name}')">删除</button>
        </div>
    `).join('');
}

async function createNewTag() {
    const nameInput = document.getElementById('newTagName');
    const colorInput = document.getElementById('newTagColor');
    
    const name = nameInput.value.trim();
    if (!name) {
        showToast('请输入标签名称', 'error');
        return;
    }
    
    try {
        await apiRequest('/api/bookshelf/tags', 'POST', { 
            name, 
            color: colorInput.value 
        });
        showToast('标签创建成功', 'success');
        nameInput.value = '';
        await loadTags();
        renderTagManageList();
    } catch (error) {
        showToast('创建失败: ' + error.message, 'error');
    }
}

function editTag(name, color) {
    const newName = prompt('输入新的标签名称:', name);
    if (newName === null) return;
    
    const newColor = prompt('输入新的颜色代码 (如 #667eea):', color);
    if (newColor === null) return;
    
    apiRequest(`/api/bookshelf/tags/${encodeURIComponent(name)}`, 'PUT', { 
        name: newName || name, 
        color: newColor || color 
    })
    .then(() => {
        showToast('标签更新成功', 'success');
        loadTags();
        renderTagManageList();
        loadBooks();
    })
    .catch(err => showToast('更新失败: ' + err.message, 'error'));
}

async function deleteTag(name) {
    if (!confirm(`确定要删除标签 "${name}" 吗？该标签将从所有书籍中移除。`)) return;
    
    try {
        await apiRequest(`/api/bookshelf/tags/${encodeURIComponent(name)}`, 'DELETE');
        showToast('标签删除成功', 'success');
        await loadTags();
        renderTagManageList();
        loadBooks();
    } catch (error) {
        showToast('删除失败: ' + error.message, 'error');
    }
}

// ==================== 批量操作 ====================

function handleBookContextMenu(bookId, event) {
    event.preventDefault();
    if (!batchMode) {
        enterBatchMode();
        toggleBookSelection(bookId, event);
    }
}

function enterBatchMode() {
    batchMode = true;
    selectedBooks.clear();
    document.getElementById('batchToolbar').style.display = 'flex';
    loadBooks();
}

function exitBatchMode() {
    batchMode = false;
    selectedBooks.clear();
    document.getElementById('batchToolbar').style.display = 'none';
    document.getElementById('selectAllBooks').checked = false;
    loadBooks();
}

function toggleBookSelection(bookId, event) {
    event.stopPropagation();
    
    if (selectedBooks.has(bookId)) {
        selectedBooks.delete(bookId);
    } else {
        selectedBooks.add(bookId);
    }
    
    updateBatchInfo();
    
    // 更新卡片视觉状态
    const card = document.querySelector(`[data-book-id="${bookId}"]`);
    if (card) {
        card.classList.toggle('selected', selectedBooks.has(bookId));
        const checkbox = card.querySelector('input[type="checkbox"]');
        if (checkbox) checkbox.checked = selectedBooks.has(bookId);
    }
}

function toggleSelectAll() {
    const selectAll = document.getElementById('selectAllBooks').checked;
    const cards = document.querySelectorAll('.book-card[data-book-id]');
    
    if (selectAll) {
        cards.forEach(card => {
            const bookId = card.dataset.bookId;
            selectedBooks.add(bookId);
            card.classList.add('selected');
            const checkbox = card.querySelector('input[type="checkbox"]');
            if (checkbox) checkbox.checked = true;
        });
    } else {
        selectedBooks.clear();
        cards.forEach(card => {
            card.classList.remove('selected');
            const checkbox = card.querySelector('input[type="checkbox"]');
            if (checkbox) checkbox.checked = false;
        });
    }
    
    updateBatchInfo();
}

function updateBatchInfo() {
    document.getElementById('selectedCount').textContent = selectedBooks.size;
}

function showBatchAddTagModal() {
    if (selectedBooks.size === 0) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    document.getElementById('batchAddTagCount').textContent = selectedBooks.size;
    renderBatchTagSelect('batchAddTagSelect');
    document.getElementById('batchAddTagModal').classList.add('active');
}

function closeBatchAddTagModal() {
    document.getElementById('batchAddTagModal').classList.remove('active');
}

function showBatchRemoveTagModal() {
    if (selectedBooks.size === 0) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    document.getElementById('batchRemoveTagCount').textContent = selectedBooks.size;
    renderBatchTagSelect('batchRemoveTagSelect');
    document.getElementById('batchRemoveTagModal').classList.add('active');
}

function closeBatchRemoveTagModal() {
    document.getElementById('batchRemoveTagModal').classList.remove('active');
}

function renderBatchTagSelect(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = allTags.map(tag => `
        <label class="batch-tag-option">
            <input type="checkbox" value="${tag.name}">
            <span class="tag-color-dot" style="background: ${tag.color}"></span>
            ${tag.name}
        </label>
    `).join('');
    
    if (allTags.length === 0) {
        container.innerHTML = '<p class="no-tags-message">暂无标签</p>';
    }
}

async function confirmBatchAddTags() {
    const checkboxes = document.querySelectorAll('#batchAddTagSelect input:checked');
    const tags = Array.from(checkboxes).map(cb => cb.value);
    
    if (tags.length === 0) {
        showToast('请选择至少一个标签', 'error');
        return;
    }
    
    try {
        await apiRequest('/api/bookshelf/books/batch/add-tags', 'POST', {
            book_ids: Array.from(selectedBooks),
            tags
        });
        showToast('批量添加标签成功', 'success');
        closeBatchAddTagModal();
        loadBooks();
        loadTags();
    } catch (error) {
        showToast('操作失败: ' + error.message, 'error');
    }
}

async function confirmBatchRemoveTags() {
    const checkboxes = document.querySelectorAll('#batchRemoveTagSelect input:checked');
    const tags = Array.from(checkboxes).map(cb => cb.value);
    
    if (tags.length === 0) {
        showToast('请选择至少一个标签', 'error');
        return;
    }
    
    try {
        await apiRequest('/api/bookshelf/books/batch/remove-tags', 'POST', {
            book_ids: Array.from(selectedBooks),
            tags
        });
        showToast('批量移除标签成功', 'success');
        closeBatchRemoveTagModal();
        loadBooks();
        loadTags();
    } catch (error) {
        showToast('操作失败: ' + error.message, 'error');
    }
}

async function batchDeleteBooks() {
    if (selectedBooks.size === 0) {
        showToast('请先选择书籍', 'error');
        return;
    }
    
    if (!confirm(`确定要删除选中的 ${selectedBooks.size} 本书籍吗？此操作不可恢复。`)) {
        return;
    }
    
    try {
        const result = await apiRequest('/api/bookshelf/books/batch/delete', 'POST', {
            book_ids: Array.from(selectedBooks)
        });
        showToast(`成功删除 ${result.success_count} 本书籍`, 'success');
        exitBatchMode();
        loadBooks();
        loadTags();
    } catch (error) {
        showToast('删除失败: ' + error.message, 'error');
    }
}

// ==================== 章节拖拽排序 ====================

function initChapterDragDrop() {
    const list = document.getElementById('chaptersList');
    if (!list) return;
    
    list.addEventListener('dragstart', handleChapterDragStart);
    list.addEventListener('dragend', handleChapterDragEnd);
    list.addEventListener('dragover', handleChapterDragOver);
    list.addEventListener('drop', handleChapterDrop);
}

function handleChapterDragStart(e) {
    const item = e.target.closest('.chapter-item');
    if (!item) return;
    
    draggedChapter = item;
    item.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
}

function handleChapterDragEnd(e) {
    const item = e.target.closest('.chapter-item');
    if (item) item.classList.remove('dragging');
    draggedChapter = null;
    
    document.querySelectorAll('.chapter-item').forEach(el => {
        el.classList.remove('drag-over');
    });
}

function handleChapterDragOver(e) {
    e.preventDefault();
    const item = e.target.closest('.chapter-item');
    if (!item || item === draggedChapter) return;
    
    document.querySelectorAll('.chapter-item').forEach(el => {
        el.classList.remove('drag-over');
    });
    item.classList.add('drag-over');
}

async function handleChapterDrop(e) {
    e.preventDefault();
    const targetItem = e.target.closest('.chapter-item');
    if (!targetItem || !draggedChapter || targetItem === draggedChapter) return;
    
    const list = document.getElementById('chaptersList');
    const items = Array.from(list.querySelectorAll('.chapter-item'));
    const draggedIndex = items.indexOf(draggedChapter);
    const targetIndex = items.indexOf(targetItem);
    
    // 重新排列DOM
    if (draggedIndex < targetIndex) {
        targetItem.after(draggedChapter);
    } else {
        targetItem.before(draggedChapter);
    }
    
    // 获取新的排序
    const newOrder = Array.from(list.querySelectorAll('.chapter-item')).map(item => item.dataset.id);
    
    // 更新序号显示
    list.querySelectorAll('.chapter-item').forEach((item, index) => {
        const orderSpan = item.querySelector('.chapter-order');
        if (orderSpan) orderSpan.textContent = `#${index + 1}`;
    });
    
    // 保存新排序到后端
    try {
        await apiRequest(`/api/bookshelf/books/${currentBookId}/chapters/reorder`, 'POST', {
            chapter_ids: newOrder
        });
        showToast('章节排序已更新', 'success');
    } catch (error) {
        showToast('排序保存失败: ' + error.message, 'error');
        // 刷新以恢复原始顺序
        openBookDetail(currentBookId);
    }
}

// 暴露给全局
window.showCreateBookModal = showCreateBookModal;
window.closeBookModal = closeBookModal;
window.saveBook = saveBook;
window.openBookDetail = openBookDetail;
window.closeBookDetailModal = closeBookDetailModal;
window.goToInsight = goToInsight;
window.editCurrentBook = editCurrentBook;
window.deleteCurrentBook = deleteCurrentBook;
window.showCreateChapterModal = showCreateChapterModal;
window.closeChapterModal = closeChapterModal;
window.saveChapter = saveChapter;
window.editChapter = editChapter;
window.deleteChapter = deleteChapter;
window.enterChapter = enterChapter;
window.readChapter = readChapter;
window.closeConfirmModal = closeConfirmModal;
window.loadBooks = loadBooks;
window.loadTags = loadTags;
window.handleSearch = handleSearch;
window.clearSearch = clearSearch;
window.toggleTagFilter = toggleTagFilter;
window.addTagToBook = addTagToBook;
window.removeTagFromBook = removeTagFromBook;
window.showTagSuggestions = showTagSuggestions;
window.showTagManageModal = showTagManageModal;
window.closeTagManageModal = closeTagManageModal;
window.createNewTag = createNewTag;
window.editTag = editTag;
window.deleteTag = deleteTag;
window.handleBookContextMenu = handleBookContextMenu;
window.enterBatchMode = enterBatchMode;
window.exitBatchMode = exitBatchMode;
window.toggleBookSelection = toggleBookSelection;
window.toggleSelectAll = toggleSelectAll;
window.showBatchAddTagModal = showBatchAddTagModal;
window.closeBatchAddTagModal = closeBatchAddTagModal;
window.showBatchRemoveTagModal = showBatchRemoveTagModal;
window.closeBatchRemoveTagModal = closeBatchRemoveTagModal;
window.confirmBatchAddTags = confirmBatchAddTags;
window.confirmBatchRemoveTags = confirmBatchRemoveTags;
window.batchDeleteBooks = batchDeleteBooks;
window.renderDetailBookTags = renderDetailBookTags;
window.removeTagFromCurrentBook = removeTagFromCurrentBook;
window.showQuickAddTagModal = showQuickAddTagModal;
window.closeQuickAddTagModal = closeQuickAddTagModal;
window.quickAddTagToBook = quickAddTagToBook;

// ==================== 快速添加标签 ====================

let quickAddCurrentTags = [];  // 缓存当前书籍标签，避免重复请求

async function showQuickAddTagModal() {
    if (!currentBookId) return;
    
    // 先获取当前书籍标签并缓存
    try {
        const result = await apiRequest(`/api/bookshelf/books/${currentBookId}`);
        quickAddCurrentTags = result.book.tags || [];
    } catch (error) {
        console.error('获取书籍标签失败:', error);
        quickAddCurrentTags = [];
    }
    
    renderQuickTagList();
    document.getElementById('quickAddTagModal').classList.add('active');
    
    // 设置输入框事件
    const input = document.getElementById('quickTagInput');
    input.value = '';
    input.focus();
    
    // 输入时过滤标签（使用缓存，无需请求）
    input.oninput = () => renderQuickTagList(input.value.trim());
    
    // 回车添加新标签
    input.onkeypress = async (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const tagName = input.value.trim();
            if (tagName) {
                await quickAddTagToBook(tagName);
                input.value = '';
                renderQuickTagList();
            }
        }
    };
}

function closeQuickAddTagModal() {
    document.getElementById('quickAddTagModal').classList.remove('active');
    quickAddCurrentTags = [];
}

function renderQuickTagList(filter = '') {
    const container = document.getElementById('quickTagList');
    if (!container) return;
    
    // 使用缓存的标签数据
    const currentTags = quickAddCurrentTags;
    
    // 过滤可用标签
    const availableTags = allTags.filter(t => 
        !currentTags.includes(t.name) &&
        (filter === '' || t.name.toLowerCase().includes(filter.toLowerCase()))
    );
    
    if (availableTags.length === 0) {
        container.innerHTML = filter 
            ? `<div class="quick-tag-item new-tag" onclick="quickAddTagToBook('${filter}')">
                   <span class="tag-icon">+</span> 创建并添加 "${filter}"
               </div>`
            : '<p class="no-tags-hint">所有标签已添加或暂无标签</p>';
        return;
    }
    
    container.innerHTML = availableTags.map(tag => `
        <div class="quick-tag-item" onclick="quickAddTagToBook('${tag.name}')">
            <span class="tag-color-dot" style="background: ${tag.color}"></span>
            <span>${tag.name}</span>
            <span class="tag-add-icon">+</span>
        </div>
    `).join('');
    
    // 如果有过滤词且不是完全匹配已有标签，显示创建选项
    if (filter && !allTags.some(t => t.name.toLowerCase() === filter.toLowerCase())) {
        container.innerHTML += `
            <div class="quick-tag-item new-tag" onclick="quickAddTagToBook('${filter}')">
                <span class="tag-icon">+</span> 创建并添加 "${filter}"
            </div>
        `;
    }
}

async function quickAddTagToBook(tagName) {
    if (!currentBookId || !tagName) return;
    
    try {
        // 如果是新标签，先创建
        if (!allTags.some(t => t.name === tagName)) {
            await apiRequest('/api/bookshelf/tags', 'POST', { name: tagName });
            await loadTags();
        }
        
        // 检查是否已存在（使用缓存）
        if (quickAddCurrentTags.includes(tagName)) {
            showToast('该标签已存在', 'info');
            return;
        }
        
        // 添加标签
        const newTags = [...quickAddCurrentTags, tagName];
        await apiRequest(`/api/bookshelf/books/${currentBookId}`, 'PUT', { tags: newTags });
        
        // 更新缓存
        quickAddCurrentTags = newTags;
        
        showToast('标签已添加', 'success');
        renderDetailBookTags(newTags);
        renderQuickTagList(document.getElementById('quickTagInput')?.value || '');
        loadBooks();
        loadTags();
    } catch (error) {
        showToast('添加标签失败: ' + error.message, 'error');
    }
}
