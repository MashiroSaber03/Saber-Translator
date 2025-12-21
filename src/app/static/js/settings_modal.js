/**
 * 设置模态框模块
 * 管理所有一次性配置的集中设置界面
 */

import * as state from './state.js';
import * as constants from './constants.js';

// 设置模态框元素的ID映射到侧边栏原始元素
const SETTINGS_FIELD_MAPPING = {
    // OCR设置
    'settingsOcrEngine': 'ocrEngine',
    'settingsSourceLanguage': 'sourceLanguage',
    'settingsBaiduApiKey': 'baiduApiKey',
    'settingsBaiduSecretKey': 'baiduSecretKey',
    'settingsBaiduVersion': 'baiduVersion',
    'settingsBaiduSourceLanguage': 'baiduSourceLanguage',
    'settingsAiVisionProvider': 'aiVisionProvider',
    'settingsAiVisionApiKey': 'aiVisionApiKey',
    'settingsCustomAiVisionBaseUrl': 'customAiVisionBaseUrl',
    'settingsAiVisionModelName': 'aiVisionModelName',
    'settingsAiVisionOcrPrompt': 'aiVisionOcrPrompt',
    'settingsAiVisionPromptMode': 'aiVisionPromptModeSelect',
    'settingsRpmAiVisionOcr': 'rpmAiVisionOcr',
    
    // 翻译服务设置
    'settingsModelProvider': 'modelProvider',
    'settingsApiKey': 'apiKey',
    'settingsCustomBaseUrl': 'customBaseUrl',
    'settingsModelName': 'modelName',
    'settingsRpmTranslation': 'rpmTranslation',
    'settingsTranslationMaxRetries': 'translationMaxRetries',  // 重试次数
    
    // 提示词设置
    'settingsPromptContent': 'promptContent',
    'settingsTranslatePromptMode': 'translatePromptModeSelect',
    'settingsEnableTextboxPrompt': 'enableTextboxPrompt',
    'settingsTextboxPromptContent': 'textboxPromptContent',
    
    // 检测设置
    'settingsTextDetector': 'textDetector',
    'settingsBoxExpandRatio': 'boxExpandRatio',
    'settingsBoxExpandTop': 'boxExpandTop',
    'settingsBoxExpandBottom': 'boxExpandBottom',
    'settingsBoxExpandLeft': 'boxExpandLeft',
    'settingsBoxExpandRight': 'boxExpandRight',
    'settingsShowDetectionDebug': 'showDetectionDebug',  // 检测框调试开关
    'settingsUsePreciseMask': 'usePreciseMask',  // 精确文字掩膜开关
    'settingsMaskDilateSize': 'maskDilateSize',  // 掩膜膨胀大小
    'settingsMaskBoxExpandRatio': 'maskBoxExpandRatio',  // 标注框区域扩大比例
    
    // 高质量翻译设置
    'settingsHqTranslateProvider': 'hqTranslateProvider',
    'settingsHqApiKey': 'hqApiKey',
    'settingsHqModelName': 'hqModelName',
    'settingsHqCustomBaseUrl': 'hqCustomBaseUrl',
    'settingsHqBatchSize': 'hqBatchSize',
    'settingsHqSessionReset': 'hqSessionReset',
    'settingsHqRpmLimit': 'hqRpmLimit',
    'settingsHqMaxRetries': 'hqMaxRetries',  // 高质量翻译重试次数
    'settingsHqLowReasoning': 'hqLowReasoning',
    'settingsHqNoThinkingMethod': 'hqNoThinkingMethod',  // 取消思考方法
    'settingsHqForceJsonOutput': 'hqForceJsonOutput',
    'settingsHqUseStream': 'hqUseStream',  // 流式调用
    'settingsHqPrompt': 'hqPrompt',
    
    // AI校对设置
    'settingsProofreadingEnabled': 'proofreadingEnabled',  // AI校对启用状态
    'settingsProofreadingMaxRetries': 'proofreadingMaxRetries',  // AI校对重试次数
    
    // 更多设置
    'settingsPdfProcessingMethod': 'pdfProcessingMethod',  // PDF处理方式
};

// 本地存储键名
const SETTINGS_STORAGE_KEY = 'saber_translator_settings';

// 不触发change事件的元素列表（这些元素的change事件会导致提示词被覆盖）
const SKIP_CHANGE_EVENT_IDS = [
    'translatePromptModeSelect',  // 会触发 handleTranslatePromptModeChange 覆盖提示词
    'aiVisionPromptModeSelect',   // 会触发 handleAiVisionPromptModeChange 覆盖提示词
    'promptContent',              // 提示词本身不需要触发change
    'textboxPromptContent',       // 文本框提示词不需要触发change
    'aiVisionOcrPrompt',          // AI视觉OCR提示词不需要触发change
    'hqPrompt',                   // 高质量翻译提示词不需要触发change
];

// ===== 服务商分组配置 =====
// 定义每个服务商选择器对应的字段分组（涵盖所有相关参数）
const PROVIDER_FIELD_GROUPS = {
    // OCR引擎选择器
    'ocrEngine': {
        'manga_ocr': [],  // 无额外参数
        'paddle_ocr': ['sourceLanguage'],
        'baidu_ocr': ['baiduApiKey', 'baiduSecretKey', 'baiduVersion', 'baiduSourceLanguage'],
        'ai_vision': ['aiVisionProvider', 'aiVisionApiKey', 'customAiVisionBaseUrl', 'aiVisionModelName', 'aiVisionOcrPrompt', 'aiVisionPromptModeSelect', 'rpmAiVisionOcr']
    },
    // AI视觉OCR服务商选择器
    'aiVisionProvider': {
        'siliconflow': ['aiVisionApiKey', 'aiVisionModelName', 'aiVisionOcrPrompt', 'aiVisionPromptModeSelect', 'rpmAiVisionOcr'],
        'volcano': ['aiVisionApiKey', 'aiVisionModelName', 'aiVisionOcrPrompt', 'aiVisionPromptModeSelect', 'rpmAiVisionOcr'],
        'gemini': ['aiVisionApiKey', 'aiVisionModelName', 'aiVisionOcrPrompt', 'aiVisionPromptModeSelect', 'rpmAiVisionOcr'],
        'custom_openai_vision': ['aiVisionApiKey', 'customAiVisionBaseUrl', 'aiVisionModelName', 'aiVisionOcrPrompt', 'aiVisionPromptModeSelect', 'rpmAiVisionOcr']
    },
    // 翻译服务商选择器（包含RPM限制和重试次数）
    'modelProvider': {
        'siliconflow': ['apiKey', 'modelName', 'rpmTranslation', 'translationMaxRetries'],
        'deepseek': ['apiKey', 'modelName', 'rpmTranslation', 'translationMaxRetries'],
        'volcano': ['apiKey', 'modelName', 'rpmTranslation', 'translationMaxRetries'],
        'caiyun': ['apiKey', 'modelName'],  // 彩云小译不需要RPM限制
        'baidu_translate': ['apiKey', 'modelName'],  // 百度翻译不需要RPM限制
        'youdao_translate': ['apiKey', 'modelName'],  // 有道翻译不需要RPM限制
        'gemini': ['apiKey', 'modelName', 'rpmTranslation', 'translationMaxRetries'],
        'ollama': ['modelName'],  // 本地服务不需要RPM限制
        'sakura': ['modelName'],  // 本地服务不需要RPM限制
        'custom_openai': ['apiKey', 'customBaseUrl', 'modelName', 'rpmTranslation', 'translationMaxRetries']
    },
    // 高质量翻译服务商选择器（包含所有参数）
    'hqTranslateProvider': {
        'siliconflow': ['hqApiKey', 'hqModelName', 'hqBatchSize', 'hqSessionReset', 'hqRpmLimit', 'hqMaxRetries', 'hqLowReasoning', 'hqForceJsonOutput', 'hqUseStream', 'hqPrompt'],
        'deepseek': ['hqApiKey', 'hqModelName', 'hqBatchSize', 'hqSessionReset', 'hqRpmLimit', 'hqMaxRetries', 'hqLowReasoning', 'hqForceJsonOutput', 'hqUseStream', 'hqPrompt'],
        'volcano': ['hqApiKey', 'hqModelName', 'hqBatchSize', 'hqSessionReset', 'hqRpmLimit', 'hqMaxRetries', 'hqLowReasoning', 'hqForceJsonOutput', 'hqUseStream', 'hqPrompt'],
        'gemini': ['hqApiKey', 'hqModelName', 'hqBatchSize', 'hqSessionReset', 'hqRpmLimit', 'hqMaxRetries', 'hqLowReasoning', 'hqForceJsonOutput', 'hqUseStream', 'hqPrompt'],
        'custom_openai': ['hqApiKey', 'hqCustomBaseUrl', 'hqModelName', 'hqBatchSize', 'hqSessionReset', 'hqRpmLimit', 'hqMaxRetries', 'hqLowReasoning', 'hqForceJsonOutput', 'hqUseStream', 'hqPrompt']
    }
};

// 服务商设置缓存（内存中的临时存储，在保存时会写入后端）
let providerSettingsCache = {
    ocrEngine: {},
    aiVisionProvider: {},
    modelProvider: {},
    hqTranslateProvider: {}
};

// 建立 sidebarId -> modalId 的反向映射（用于从JSON加载设置时同步到模态框）
const REVERSE_FIELD_MAPPING = {};
for (const [modalId, sidebarId] of Object.entries(SETTINGS_FIELD_MAPPING)) {
    REVERSE_FIELD_MAPPING[sidebarId] = modalId;
}

// AI校对轮次默认提示词已移至 constants.js 统一管理

/**
 * 初始化设置模态框
 */
export function initSettingsModal() {
    const modal = document.getElementById('settingsModal');
    const openBtn = document.getElementById('openSettingsBtn');
    const closeBtn = modal?.querySelector('.settings-modal-close');
    const cancelBtn = document.getElementById('settingsCancelBtn');
    const saveBtn = document.getElementById('settingsSaveBtn');
    
    if (!modal || !openBtn) {
        console.warn('设置模态框元素未找到');
        return;
    }
    
    // 精简侧边栏 - 隐藏已移到设置模态框的配置项
    simplifySidebar();
    
    // 打开设置
    openBtn.addEventListener('click', () => {
        openSettingsModal();
    });
    
    // 关闭设置
    closeBtn?.addEventListener('click', closeSettingsModal);
    cancelBtn?.addEventListener('click', closeSettingsModal);
    
    // 点击背景关闭
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeSettingsModal();
        }
    });
    
    // 保存设置
    saveBtn?.addEventListener('click', () => {
        saveSettings();
        closeSettingsModal();
    });
    
    // Tab切换
    initTabNavigation();
    
    // 条件显示逻辑
    initConditionalDisplay();
    
    // 初始化AI校对设置UI
    initProofreadingSettingsUI();
    
    // 初始化提示词管理功能
    initPromptLibrary();
    
    // 初始化密码显示/隐藏切换按钮
    initPasswordToggleButtons();
    
    // 初始化模型获取按钮
    initFetchModelsButtons();
    
    // 从本地存储加载设置（在初始化完成后立即加载）
    setTimeout(async () => {
        try {
            await loadSettingsFromStorage();
        } catch (err) {
            console.error('加载设置失败:', err);
        }
    }, 200);
}

/**
 * 打开设置模态框
 */
function openSettingsModal() {
    const modal = document.getElementById('settingsModal');
    if (modal) {
        syncFromSidebar();
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
    } else {
        console.error('设置模态框元素未找到');
    }
}

/**
 * 关闭设置模态框
 */
function closeSettingsModal() {
    const modal = document.getElementById('settingsModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    }
}

/**
 * 初始化Tab导航
 */
function initTabNavigation() {
    const tabs = document.querySelectorAll('.settings-tab');
    const panes = document.querySelectorAll('.settings-tab-pane');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;
            
            // 切换tab active状态
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // 切换pane显示
            panes.forEach(pane => {
                if (pane.id === `settings-${targetTab}`) {
                    pane.classList.add('active');
                } else {
                    pane.classList.remove('active');
                }
            });
        });
    });
}

// ===== 服务商配置分组存储核心函数 =====

/**
 * 保存当前服务商的配置到缓存
 * @param {string} selectorId - 服务商选择器的ID（如 'ocrEngine', 'modelProvider'）
 * @param {string} providerValue - 当前选中的服务商值
 */
function saveProviderSettings(selectorId, providerValue) {
    const fieldGroup = PROVIDER_FIELD_GROUPS[selectorId];
    if (!fieldGroup || !fieldGroup[providerValue]) return;
    
    const fields = fieldGroup[providerValue];
    if (!providerSettingsCache[selectorId]) {
        providerSettingsCache[selectorId] = {};
    }
    if (!providerSettingsCache[selectorId][providerValue]) {
        providerSettingsCache[selectorId][providerValue] = {};
    }
    
    // 保存每个字段的当前值
    fields.forEach(fieldId => {
        // 尝试从模态框元素获取值
        const modalId = REVERSE_FIELD_MAPPING[fieldId] ? null : Object.keys(SETTINGS_FIELD_MAPPING).find(k => SETTINGS_FIELD_MAPPING[k] === fieldId);
        const settingsModalId = 'settings' + fieldId.charAt(0).toUpperCase() + fieldId.slice(1);
        
        // 优先使用模态框元素
        let el = document.getElementById(settingsModalId) || document.getElementById(fieldId);
        
        if (el) {
            if (el.type === 'checkbox') {
                providerSettingsCache[selectorId][providerValue][fieldId] = el.checked;
            } else {
                providerSettingsCache[selectorId][providerValue][fieldId] = el.value;
            }
        }
    });
    
    console.log(`[Settings] 保存 ${selectorId} -> ${providerValue}:`, providerSettingsCache[selectorId][providerValue]);
}

/**
 * 恢复指定服务商的配置从缓存
 * @param {string} selectorId - 服务商选择器的ID
 * @param {string} providerValue - 要恢复的服务商值
 */
function restoreProviderSettings(selectorId, providerValue) {
    const fieldGroup = PROVIDER_FIELD_GROUPS[selectorId];
    if (!fieldGroup || !fieldGroup[providerValue]) return;
    
    const cached = providerSettingsCache[selectorId]?.[providerValue];
    if (!cached) {
        console.log(`[Settings] ${selectorId} -> ${providerValue} 无缓存，使用默认值`);
        // 清空相关字段（可选：也可以保留当前值）
        const fields = fieldGroup[providerValue];
        fields.forEach(fieldId => {
            const settingsModalId = 'settings' + fieldId.charAt(0).toUpperCase() + fieldId.slice(1);
            let el = document.getElementById(settingsModalId) || document.getElementById(fieldId);
            if (el && el.type !== 'checkbox') {
                // 不清空，保留默认值
            }
        });
        return;
    }
    
    console.log(`[Settings] 恢复 ${selectorId} -> ${providerValue}:`, cached);
    
    const fields = fieldGroup[providerValue];
    fields.forEach(fieldId => {
        if (cached[fieldId] === undefined) return;
        
        const settingsModalId = 'settings' + fieldId.charAt(0).toUpperCase() + fieldId.slice(1);
        let el = document.getElementById(settingsModalId) || document.getElementById(fieldId);
        
        if (el) {
            if (el.type === 'checkbox') {
                el.checked = cached[fieldId];
            } else {
                el.value = cached[fieldId];
            }
            // 同步到侧边栏元素
            const sidebarEl = document.getElementById(fieldId);
            if (sidebarEl && sidebarEl !== el) {
                if (sidebarEl.type === 'checkbox') {
                    sidebarEl.checked = cached[fieldId];
                } else {
                    sidebarEl.value = cached[fieldId];
                }
            }
            
            // 如果恢复的字段是一个服务商选择器，更新 currentProviderValues
            if (fieldId in currentProviderValues) {
                currentProviderValues[fieldId] = cached[fieldId];
            }
        }
    });
}

/**
 * 处理服务商切换事件 - 通用处理器
 * @param {string} selectorId - 服务商选择器的ID
 * @param {string} oldValue - 切换前的服务商值
 * @param {string} newValue - 切换后的服务商值
 */
function handleProviderChange(selectorId, oldValue, newValue) {
    if (oldValue === newValue) return;
    
    // 1. 保存旧服务商的配置
    if (oldValue) {
        saveProviderSettings(selectorId, oldValue);
    }
    
    // 2. 恢复新服务商的配置
    restoreProviderSettings(selectorId, newValue);
}

// 记录每个选择器的当前值（用于检测切换）
let currentProviderValues = {
    ocrEngine: null,
    aiVisionProvider: null,
    modelProvider: null,
    hqTranslateProvider: null
};

/**
 * 初始化条件显示逻辑
 */
function initConditionalDisplay() {
    // OCR引擎切换
    const ocrEngineSelect = document.getElementById('settingsOcrEngine');
    if (ocrEngineSelect) {
        currentProviderValues.ocrEngine = ocrEngineSelect.value;
        ocrEngineSelect.addEventListener('change', () => {
            const oldValue = currentProviderValues.ocrEngine;
            const newValue = ocrEngineSelect.value;
            handleProviderChange('ocrEngine', oldValue, newValue);
            currentProviderValues.ocrEngine = newValue;
            updateOcrOptionsDisplay();
        });
        updateOcrOptionsDisplay();
    }
    
    // 翻译服务商切换
    const providerSelect = document.getElementById('settingsModelProvider');
    if (providerSelect) {
        currentProviderValues.modelProvider = providerSelect.value;
        providerSelect.addEventListener('change', () => {
            const oldValue = currentProviderValues.modelProvider;
            const newValue = providerSelect.value;
            handleProviderChange('modelProvider', oldValue, newValue);
            currentProviderValues.modelProvider = newValue;
            updateTranslateOptionsDisplay();
        });
        updateTranslateOptionsDisplay();
    }
    
    // AI视觉服务商切换
    const aiVisionProviderSelect = document.getElementById('settingsAiVisionProvider');
    if (aiVisionProviderSelect) {
        currentProviderValues.aiVisionProvider = aiVisionProviderSelect.value;
        aiVisionProviderSelect.addEventListener('change', () => {
            const oldValue = currentProviderValues.aiVisionProvider;
            const newValue = aiVisionProviderSelect.value;
            handleProviderChange('aiVisionProvider', oldValue, newValue);
            currentProviderValues.aiVisionProvider = newValue;
            updateAiVisionOptionsDisplay();
        });
        updateAiVisionOptionsDisplay();
    }
    
    // 高质量翻译服务商切换
    const hqProviderSelect = document.getElementById('settingsHqTranslateProvider');
    if (hqProviderSelect) {
        currentProviderValues.hqTranslateProvider = hqProviderSelect.value;
        hqProviderSelect.addEventListener('change', () => {
            const oldValue = currentProviderValues.hqTranslateProvider;
            const newValue = hqProviderSelect.value;
            handleProviderChange('hqTranslateProvider', oldValue, newValue);
            currentProviderValues.hqTranslateProvider = newValue;
            updateHqOptionsDisplay();
        });
        updateHqOptionsDisplay();
    }
    
    // 文本框提示词启用切换
    const enableTextboxPrompt = document.getElementById('settingsEnableTextboxPrompt');
    if (enableTextboxPrompt) {
        enableTextboxPrompt.addEventListener('change', () => {
            const area = document.getElementById('settingsTextboxPromptArea');
            if (area) {
                area.style.display = enableTextboxPrompt.checked ? 'block' : 'none';
            }
        });
    }
    
    // 检测器切换 - 控制精确文字掩膜选项的显示/隐藏
    const textDetectorSelect = document.getElementById('settingsTextDetector');
    if (textDetectorSelect) {
        textDetectorSelect.addEventListener('change', () => {
            updatePreciseMaskVisibility(textDetectorSelect.value);
        });
        // 初始化时也检查一次
        updatePreciseMaskVisibility(textDetectorSelect.value);
    }
}

/**
 * 更新精确文字掩膜选项的显示/隐藏
 * 只有 CTD 和 Default 检测器支持精确掩膜
 * @param {string} detectorType - 检测器类型
 */
function updatePreciseMaskVisibility(detectorType) {
    const preciseItem = document.getElementById('usePreciseMaskItem');
    const dilateItem = document.getElementById('maskDilateSizeItem');
    const boxExpandItem = document.getElementById('maskBoxExpandRatioItem');
    
    // 只有 CTD 和 Default 支持精确掩膜
    const supportsPreciseMask = detectorType === 'ctd' || detectorType === 'default';
    
    if (preciseItem) {
        preciseItem.style.display = supportsPreciseMask ? 'block' : 'none';
        
        // 切换到不支持的检测器时，自动关闭此选项并同步 state
        if (!supportsPreciseMask) {
            const checkbox = document.getElementById('settingsUsePreciseMask');
            if (checkbox) {
                checkbox.checked = false;
                state.setUsePreciseMask(false);
            }
        }
    }
    
    // 膨胀选项和标注框扩大选项跟随精确掩膜选项显示/隐藏
    if (dilateItem) {
        dilateItem.style.display = supportsPreciseMask ? 'block' : 'none';
    }
    if (boxExpandItem) {
        boxExpandItem.style.display = supportsPreciseMask ? 'block' : 'none';
    }
}

/**
 * 更新OCR选项显示
 */
function updateOcrOptionsDisplay() {
    const engine = document.getElementById('settingsOcrEngine')?.value;
    
    const paddleGroup = document.getElementById('settingsPaddleOcrGroup');
    const baiduGroup = document.getElementById('settingsBaiduOcrGroup');
    const aiVisionGroup = document.getElementById('settingsAiVisionGroup');
    
    if (paddleGroup) paddleGroup.style.display = engine === 'paddle_ocr' ? 'block' : 'none';
    if (baiduGroup) baiduGroup.style.display = engine === 'baidu_ocr' ? 'block' : 'none';
    if (aiVisionGroup) aiVisionGroup.style.display = engine === 'ai_vision' ? 'block' : 'none';
}

/**
 * 更新翻译服务选项显示
 * 根据不同服务商显示不同的配置字段
 */
function updateTranslateOptionsDisplay() {
    const provider = document.getElementById('settingsModelProvider')?.value;
    
    // 获取所有需要控制显示的元素
    const apiKeyDiv = document.getElementById('settingsApiKeyDiv');
    const apiKeyLabel = document.getElementById('settingsApiKeyLabel');
    const apiKeyInput = document.getElementById('settingsApiKey');
    const customUrlDiv = document.getElementById('settingsCustomBaseUrlDiv');
    const modelNameDiv = document.getElementById('settingsModelNameDiv');
    const modelNameLabel = document.getElementById('settingsModelNameLabel');
    const modelNameInput = document.getElementById('settingsModelName');
    const localModelDiv = document.getElementById('settingsLocalModelDiv');
    const ollamaModelsList = document.getElementById('settingsOllamaModelsList');
    const sakuraModelsList = document.getElementById('settingsSakuraModelsList');
    const rpmDiv = document.getElementById('settingsRpmDiv');
    const retriesDiv = document.getElementById('settingsRetriesDiv');
    
    // 默认显示所有
    if (apiKeyDiv) apiKeyDiv.style.display = 'block';
    if (customUrlDiv) customUrlDiv.style.display = 'none';
    if (modelNameDiv) modelNameDiv.style.display = 'block';
    if (localModelDiv) localModelDiv.style.display = 'none';
    if (rpmDiv) rpmDiv.style.display = 'block';
    if (retriesDiv) retriesDiv.style.display = 'block';
    
    // 切换服务商时隐藏模型列表
    const modelSelectDiv = document.getElementById('settingsModelSelectDiv');
    if (modelSelectDiv) modelSelectDiv.style.display = 'none';
    
    // 获取模型按钮
    const fetchModelsBtn = document.getElementById('settingsFetchModelsBtn');
    
    // 恢复默认标签
    if (apiKeyLabel) apiKeyLabel.textContent = 'API Key:';
    if (apiKeyInput) apiKeyInput.placeholder = '请输入API Key';
    if (modelNameLabel) modelNameLabel.textContent = '模型名称:';
    if (modelNameInput) modelNameInput.placeholder = '请输入模型名称';
    
    // 支持模型获取的服务商
    const supportsFetchModels = ['siliconflow', 'deepseek', 'volcano', 'gemini', 'custom_openai'];
    if (fetchModelsBtn) {
        fetchModelsBtn.style.display = supportsFetchModels.includes(provider) ? 'flex' : 'none';
    }
    
    switch (provider) {
        case 'baidu_translate':
            // 百度翻译: App ID + App Key
            if (apiKeyLabel) apiKeyLabel.textContent = 'App ID:';
            if (apiKeyInput) apiKeyInput.placeholder = '请输入百度翻译App ID';
            if (modelNameLabel) modelNameLabel.textContent = 'App Key:';
            if (modelNameInput) modelNameInput.placeholder = '请输入百度翻译App Key';
            if (rpmDiv) rpmDiv.style.display = 'none'; // 百度翻译不需要RPM限制
            break;
            
        case 'youdao_translate':
            // 有道翻译: App Key + App Secret
            if (apiKeyLabel) apiKeyLabel.textContent = 'App Key:';
            if (apiKeyInput) apiKeyInput.placeholder = '请输入有道翻译应用ID';
            if (modelNameLabel) modelNameLabel.textContent = 'App Secret:';
            if (modelNameInput) modelNameInput.placeholder = '请输入有道翻译应用密钥';
            if (rpmDiv) rpmDiv.style.display = 'none'; // 有道翻译不需要RPM限制
            break;
            
        case 'caiyun':
            // 彩云小译: Token + 源语言(可选)
            if (apiKeyLabel) apiKeyLabel.textContent = 'API Token:';
            if (apiKeyInput) apiKeyInput.placeholder = '请输入彩云小译Token';
            if (modelNameLabel) modelNameLabel.textContent = '源语言 (可选):';
            if (modelNameInput) {
                modelNameInput.placeholder = '可选: auto/日语/英语';
                if (!modelNameInput.value) modelNameInput.value = 'auto';
            }
            if (rpmDiv) rpmDiv.style.display = 'none'; // 彩云小译不需要RPM限制
            break;
            
        case 'ollama':
            // Ollama本地: 无需API Key，显示模型列表
            if (apiKeyDiv) apiKeyDiv.style.display = 'none';
            if (modelNameDiv) modelNameDiv.style.display = 'none';
            if (localModelDiv) localModelDiv.style.display = 'block';
            if (ollamaModelsList) ollamaModelsList.style.display = 'block';
            if (sakuraModelsList) sakuraModelsList.style.display = 'none';
            if (rpmDiv) rpmDiv.style.display = 'none'; // 本地服务不需要RPM限制
            if (retriesDiv) retriesDiv.style.display = 'none'; // 本地服务简化配置
            break;
            
        case 'sakura':
            // Sakura本地: 无需API Key，显示模型列表
            if (apiKeyDiv) apiKeyDiv.style.display = 'none';
            if (modelNameDiv) modelNameDiv.style.display = 'none';
            if (localModelDiv) localModelDiv.style.display = 'block';
            if (ollamaModelsList) ollamaModelsList.style.display = 'none';
            if (sakuraModelsList) sakuraModelsList.style.display = 'block';
            if (rpmDiv) rpmDiv.style.display = 'none'; // 本地服务不需要RPM限制
            if (retriesDiv) retriesDiv.style.display = 'none'; // 本地服务简化配置
            break;
            
        case 'custom_openai':
            // 自定义OpenAI兼容服务: 需要Base URL
            if (customUrlDiv) customUrlDiv.style.display = 'block';
            break;
            
        default:
            // SiliconFlow, DeepSeek, 火山引擎, Gemini 等云服务
            // 使用默认显示
            break;
    }
}

/**
 * 更新AI视觉选项显示
 */
function updateAiVisionOptionsDisplay() {
    const provider = document.getElementById('settingsAiVisionProvider')?.value;
    const customUrlDiv = document.getElementById('settingsCustomAiVisionBaseUrlDiv');
    
    if (customUrlDiv) {
        customUrlDiv.style.display = provider === 'custom_openai_vision' ? 'block' : 'none';
    }
    
    // 切换服务商时隐藏模型列表
    const modelSelectDiv = document.getElementById('settingsAiVisionModelSelectDiv');
    if (modelSelectDiv) modelSelectDiv.style.display = 'none';
    
    // 支持模型获取的服务商
    const fetchModelsBtn = document.getElementById('settingsAiVisionFetchModelsBtn');
    const supportsFetchModels = ['siliconflow', 'volcano', 'gemini', 'custom_openai_vision'];
    if (fetchModelsBtn) {
        fetchModelsBtn.style.display = supportsFetchModels.includes(provider) ? 'flex' : 'none';
    }
}

/**
 * 更新高质量翻译选项显示
 */
function updateHqOptionsDisplay() {
    const provider = document.getElementById('settingsHqTranslateProvider')?.value;
    const customUrlDiv = document.getElementById('settingsHqCustomBaseUrlDiv');
    
    if (customUrlDiv) {
        customUrlDiv.style.display = provider === 'custom_openai' ? 'block' : 'none';
    }
    
    // 切换服务商时隐藏模型列表
    const modelSelectDiv = document.getElementById('settingsHqModelSelectDiv');
    if (modelSelectDiv) modelSelectDiv.style.display = 'none';
}

/**
 * 从侧边栏同步值到模态框
 * 注意：如果侧边栏元素被隐藏了，则跳过同步（使用模态框已有的值）
 */
function syncFromSidebar() {
    for (const [modalId, sidebarId] of Object.entries(SETTINGS_FIELD_MAPPING)) {
        const modalEl = document.getElementById(modalId);
        const sidebarEl = document.getElementById(sidebarId);
        
        if (modalEl && sidebarEl) {
            // 检查侧边栏元素是否被隐藏
            const isHidden = sidebarEl.closest('.sidebar-hidden-config') !== null;
            if (!isHidden) {
                // 只有侧边栏元素未被隐藏时，才从侧边栏同步
                if (modalEl.type === 'checkbox') {
                    modalEl.checked = sidebarEl.checked;
                } else {
                    modalEl.value = sidebarEl.value;
                }
            }
        }
    }
    
    // 更新当前服务商值记录
    currentProviderValues.ocrEngine = document.getElementById('settingsOcrEngine')?.value || null;
    currentProviderValues.aiVisionProvider = document.getElementById('settingsAiVisionProvider')?.value || null;
    currentProviderValues.modelProvider = document.getElementById('settingsModelProvider')?.value || null;
    currentProviderValues.hqTranslateProvider = document.getElementById('settingsHqTranslateProvider')?.value || null;
    
    // 更新条件显示
    updateOcrOptionsDisplay();
    updateTranslateOptionsDisplay();
    updateAiVisionOptionsDisplay();
    updateHqOptionsDisplay();
}

/**
 * 从模态框同步值到侧边栏
 */
function syncToSidebar() {
    for (const [modalId, sidebarId] of Object.entries(SETTINGS_FIELD_MAPPING)) {
        const modalEl = document.getElementById(modalId);
        const sidebarEl = document.getElementById(sidebarId);
        
        if (modalEl && sidebarEl) {
            if (modalEl.type === 'checkbox') {
                sidebarEl.checked = modalEl.checked;
            } else {
                sidebarEl.value = modalEl.value;
            }
            
            // 只对不在跳过列表中的元素触发change事件
            if (!SKIP_CHANGE_EVENT_IDS.includes(sidebarId)) {
                sidebarEl.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    }
}

/**
 * 保存设置
 */
async function saveSettings() {
    // 保存当前所有服务商的配置到缓存
    saveAllCurrentProviderSettings();
    
    // 同步提示词到 state
    const promptContentEl = document.getElementById('settingsPromptContent');
    if (promptContentEl) {
        const newPromptValue = promptContentEl.value || '';
        state.setPromptState(newPromptValue, state.defaultPromptContent, state.savedPromptNames, state.defaultTranslateJsonPrompt);
    }
    const textboxPromptContentEl = document.getElementById('settingsTextboxPromptContent');
    if (textboxPromptContentEl && textboxPromptContentEl.value) {
        state.setTextboxPromptState(textboxPromptContentEl.value, state.defaultTextboxPromptContent, state.savedTextboxPromptNames);
    }
    const aiVisionOcrPromptEl = document.getElementById('settingsAiVisionOcrPrompt');
    if (aiVisionOcrPromptEl && aiVisionOcrPromptEl.value) {
        state.setAiVisionOcrPrompt(aiVisionOcrPromptEl.value);
    }
    
    // 同步到侧边栏
    syncToSidebar();
    
    // 同步RPM限制到 state
    const rpmTranslationEl = document.getElementById('settingsRpmTranslation');
    if (rpmTranslationEl) {
        state.setrpmLimitTranslation(parseInt(rpmTranslationEl.value) || 0);
    }
    const rpmAiVisionOcrEl = document.getElementById('settingsRpmAiVisionOcr');
    if (rpmAiVisionOcrEl) {
        state.setrpmLimitAiVisionOcr(parseInt(rpmAiVisionOcrEl.value) || 0);
    }
    
    // 同步重试次数到 state
    const translationMaxRetriesEl = document.getElementById('settingsTranslationMaxRetries');
    if (translationMaxRetriesEl) {
        state.setTranslationMaxRetries(parseInt(translationMaxRetriesEl.value) || 3);
    }
    
    // 同步检测设置到 state（直接更新 state）
    const textDetectorEl = document.getElementById('settingsTextDetector');
    if (textDetectorEl) {
        state.setTextDetector(textDetectorEl.value);
    }
    
    // 同步文本框扩展参数到 state
    const boxExpandRatioEl = document.getElementById('settingsBoxExpandRatio');
    if (boxExpandRatioEl) {
        state.setBoxExpandRatio(parseInt(boxExpandRatioEl.value) || 0);
    }
    const boxExpandTopEl = document.getElementById('settingsBoxExpandTop');
    if (boxExpandTopEl) {
        state.setBoxExpandTop(parseInt(boxExpandTopEl.value) || 0);
    }
    const boxExpandBottomEl = document.getElementById('settingsBoxExpandBottom');
    if (boxExpandBottomEl) {
        state.setBoxExpandBottom(parseInt(boxExpandBottomEl.value) || 0);
    }
    const boxExpandLeftEl = document.getElementById('settingsBoxExpandLeft');
    if (boxExpandLeftEl) {
        state.setBoxExpandLeft(parseInt(boxExpandLeftEl.value) || 0);
    }
    const boxExpandRightEl = document.getElementById('settingsBoxExpandRight');
    if (boxExpandRightEl) {
        state.setBoxExpandRight(parseInt(boxExpandRightEl.value) || 0);
    }
    
    const showDetectionDebugEl = document.getElementById('settingsShowDetectionDebug');
    if (showDetectionDebugEl) {
        state.setShowDetectionDebug(showDetectionDebugEl.checked);
    }
    
    // 同步精确文字掩膜开关到 state
    const usePreciseMaskEl = document.getElementById('settingsUsePreciseMask');
    if (usePreciseMaskEl) {
        state.setUsePreciseMask(usePreciseMaskEl.checked);
    }
    
    // 同步掩膜膨胀大小到 state
    const maskDilateSizeEl = document.getElementById('settingsMaskDilateSize');
    if (maskDilateSizeEl) {
        state.setMaskDilateSize(parseInt(maskDilateSizeEl.value) || 0);
    }
    
    // 同步标注框区域扩大比例到 state
    const maskBoxExpandRatioEl = document.getElementById('settingsMaskBoxExpandRatio');
    if (maskBoxExpandRatioEl) {
        state.setMaskBoxExpandRatio(parseInt(maskBoxExpandRatioEl.value) || 0);
    }
    
    // 同步 AI 视觉 OCR 自定义 Base URL 到 state
    const customAiVisionBaseUrlEl = document.getElementById('settingsCustomAiVisionBaseUrl');
    if (customAiVisionBaseUrlEl) {
        state.setCustomAiVisionBaseUrl(customAiVisionBaseUrlEl.value || '');
    }
    
    // 同步高质量翻译设置到 state
    const hqProviderEl = document.getElementById('settingsHqTranslateProvider');
    if (hqProviderEl) {
        state.setHqTranslateProvider(hqProviderEl.value);
    }
    const hqApiKeyEl = document.getElementById('settingsHqApiKey');
    if (hqApiKeyEl) {
        state.setHqApiKey(hqApiKeyEl.value || '');
    }
    const hqModelNameEl = document.getElementById('settingsHqModelName');
    if (hqModelNameEl) {
        state.setHqModelName(hqModelNameEl.value || '');
    }
    const hqCustomBaseUrlEl = document.getElementById('settingsHqCustomBaseUrl');
    if (hqCustomBaseUrlEl) {
        state.setHqCustomBaseUrl(hqCustomBaseUrlEl.value || '');
    }
    const hqBatchSizeEl = document.getElementById('settingsHqBatchSize');
    if (hqBatchSizeEl) {
        state.setHqBatchSize(parseInt(hqBatchSizeEl.value) || 3);
    }
    const hqSessionResetEl = document.getElementById('settingsHqSessionReset');
    if (hqSessionResetEl) {
        state.setHqSessionReset(parseInt(hqSessionResetEl.value) || 20);
    }
    const hqRpmLimitEl = document.getElementById('settingsHqRpmLimit');
    if (hqRpmLimitEl) {
        state.setHqRpmLimit(parseInt(hqRpmLimitEl.value) || 7);
    }
    const hqMaxRetriesEl = document.getElementById('settingsHqMaxRetries');
    if (hqMaxRetriesEl) {
        const value = parseInt(hqMaxRetriesEl.value);
        state.setHqTranslationMaxRetries(value);
    }
    const hqLowReasoningEl = document.getElementById('settingsHqLowReasoning');
    if (hqLowReasoningEl) {
        state.setHqLowReasoning(hqLowReasoningEl.checked);
    }
    const hqNoThinkingMethodEl = document.getElementById('settingsHqNoThinkingMethod');
    if (hqNoThinkingMethodEl) {
        state.setHqNoThinkingMethod(hqNoThinkingMethodEl.value || 'gemini');
    }
    const hqForceJsonOutputEl = document.getElementById('settingsHqForceJsonOutput');
    if (hqForceJsonOutputEl) {
        state.setHqForceJsonOutput(hqForceJsonOutputEl.checked);
    }
    const hqUseStreamEl = document.getElementById('settingsHqUseStream');
    if (hqUseStreamEl) {
        state.setHqUseStream(hqUseStreamEl.checked);
    }
    const hqPromptEl = document.getElementById('settingsHqPrompt');
    if (hqPromptEl) {
        state.setHqPrompt(hqPromptEl.value || '');
    }
    
    // 同步 PDF 处理方式到 state
    const pdfProcessingMethodEl = document.getElementById('settingsPdfProcessingMethod');
    if (pdfProcessingMethodEl) {
        state.setPdfProcessingMethod(pdfProcessingMethodEl.value || 'backend');
    }
    
    // 保存到后端
    await saveSettingsToStorage();
    
    // 显示保存成功提示
    showToast('设置已保存');
}

/**
 * 保存所有当前服务商的配置到缓存
 */
function saveAllCurrentProviderSettings() {
    // OCR引擎
    const ocrEngine = document.getElementById('settingsOcrEngine')?.value;
    if (ocrEngine) {
        saveProviderSettings('ocrEngine', ocrEngine);
    }
    
    // AI视觉服务商
    const aiVisionProvider = document.getElementById('settingsAiVisionProvider')?.value;
    if (aiVisionProvider) {
        saveProviderSettings('aiVisionProvider', aiVisionProvider);
    }
    
    // 翻译服务商
    const modelProvider = document.getElementById('settingsModelProvider')?.value;
    if (modelProvider) {
        saveProviderSettings('modelProvider', modelProvider);
    }
    
    // 高质量翻译服务商
    const hqProvider = document.getElementById('settingsHqTranslateProvider')?.value;
    if (hqProvider) {
        saveProviderSettings('hqTranslateProvider', hqProvider);
    }
}

/**
 * 保存设置到后端 JSON 文件
 */
async function saveSettingsToStorage() {
    const settings = {};
    
    for (const [modalId, sidebarId] of Object.entries(SETTINGS_FIELD_MAPPING)) {
        const el = document.getElementById(modalId);
        if (el) {
            if (el.type === 'checkbox') {
                settings[sidebarId] = el.checked;
            } else {
                settings[sidebarId] = el.value;
            }
        }
    }
    
    // 同时保存AI校对设置
    const proofreadingSettings = getProofreadingSettings();
    if (proofreadingSettings) {
        settings['proofreading'] = proofreadingSettings;
    }
    
    // ===== 保存服务商分组设置 =====
    settings['providerSettings'] = providerSettingsCache;
    
    try {
        const response = await fetch('/api/save_settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ settings: settings })
        });
        
        const result = await response.json();
        if (!result.success) {
            console.error('保存设置到后端失败:', result.error);
            // 回退到 localStorage
            localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
        }
    } catch (e) {
        console.error('保存设置到后端失败:', e);
        // 回退到 localStorage
        localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    }
}

/**
 * 辅助函数：获取AI校对设置（从 state.proofreadingRounds 读取完整配置）
 */
function getProofreadingSettings() {
    return {
        enabled: document.getElementById('settingsProofreadingEnabled')?.checked || false,
        maxRetries: document.getElementById('settingsProofreadingMaxRetries')?.value || '5',
        rounds: state.proofreadingRounds || []  // 保存完整的轮次数组
    };
}

/**
 * 辅助函数：应用AI校对设置（恢复轮次配置到 state）
 */
function applyProofreadingSettings(settings) {
    if (!settings) return;
    
    // 恢复启用状态
    const enabledEl = document.getElementById('settingsProofreadingEnabled');
    const originalEnabledEl = document.getElementById('proofreadingEnabled');
    if (enabledEl && settings.enabled !== undefined) {
        enabledEl.checked = settings.enabled;
        if (originalEnabledEl) {
            originalEnabledEl.checked = settings.enabled;
        }
    }
    
    // 恢复重试次数
    const maxRetriesEl = document.getElementById('settingsProofreadingMaxRetries');
    const originalMaxRetriesEl = document.getElementById('proofreadingMaxRetries');
    if (maxRetriesEl && settings.maxRetries !== undefined) {
        maxRetriesEl.value = settings.maxRetries;
        if (originalMaxRetriesEl) {
            originalMaxRetriesEl.value = settings.maxRetries;
        }
    }
    
    // 恢复轮次配置到 state
    if (settings.rounds && Array.isArray(settings.rounds) && settings.rounds.length > 0) {
        if (state.setProofreadingRounds) {
            state.setProofreadingRounds(settings.rounds);
        } else {
            // 如果没有 setter 函数，直接修改数组内容
            state.proofreadingRounds.length = 0;
            state.proofreadingRounds.push(...settings.rounds);
        }
    }
    
    // 重新渲染轮次 UI
    renderProofreadingRounds();
}

/**
 * 从后端加载设置
 */
async function loadSettingsFromStorage() {
    try {
        const response = await fetch('/api/get_settings');
        const result = await response.json();
        
        let settings = null;
        
        if (result.success && result.settings && Object.keys(result.settings).length > 0) {
            settings = result.settings;
        } else {
            // 尝试从 localStorage 迁移旧设置
            const stored = localStorage.getItem(SETTINGS_STORAGE_KEY);
            if (stored) {
                settings = JSON.parse(stored);
                // 迁移到后端
                await fetch('/api/save_settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ settings: settings })
                });
            }
        }
        
        if (settings) {
            applySettingsToUI(settings);
        }
    } catch (e) {
        console.error('从后端加载设置失败:', e);
        // 回退到 localStorage
        loadSettingsFromLocalStorage();
    }
}

/**
 * 应用设置到UI
 * 优先设置模态框元素，然后同步到侧边栏元素（如果存在）
 */
function applySettingsToUI(settings) {
    // ===== 先加载服务商分组设置到缓存 =====
    if (settings.providerSettings) {
        providerSettingsCache = settings.providerSettings;
        console.log('[Settings] 已加载服务商分组设置:', providerSettingsCache);
    }
    
    for (const [sidebarId, value] of Object.entries(settings)) {
        // 跳过特殊设置，单独处理
        if (sidebarId === 'proofreading' || sidebarId === 'proofreadingRounds' || sidebarId === 'providerSettings') continue;
        
        // 1. 优先设置模态框元素（通过反向映射找到对应的modalId）
        const modalId = REVERSE_FIELD_MAPPING[sidebarId];
        if (modalId) {
            const modalEl = document.getElementById(modalId);
            if (modalEl) {
                if (modalEl.type === 'checkbox') {
                    modalEl.checked = value;
                } else {
                    modalEl.value = value;
                }
            }
        }
        
        // 2. 同步提示词到 state
        if (sidebarId === 'promptContent' && value) {
            state.setPromptState(value, state.defaultPromptContent, state.savedPromptNames, state.defaultTranslateJsonPrompt);
        }
        
        // 3. 设置侧边栏元素
        const sidebarEl = document.getElementById(sidebarId);
        if (sidebarEl) {
            if (sidebarEl.type === 'checkbox') {
                sidebarEl.checked = value;
            } else {
                sidebarEl.value = value;
            }
            
            // 只对不在跳过列表中的元素触发change事件
            if (!SKIP_CHANGE_EVENT_IDS.includes(sidebarId)) {
                sidebarEl.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
        
        // 4. 其他设置同步到 state
        // 提示词相关（promptContent 已在上面处理）
        if (sidebarId === 'textboxPromptContent' && value) {
            state.setTextboxPromptState(value, state.defaultTextboxPromptContent, state.savedTextboxPromptNames);
        }
        if (sidebarId === 'aiVisionOcrPrompt' && value) {
            state.setAiVisionOcrPrompt(value);
        }
        // 高质量翻译相关
        if (sidebarId === 'hqPrompt' && value) {
            state.setHqPrompt(value);
        }
        if (sidebarId === 'hqTranslateProvider' && value) {
            state.setHqTranslateProvider(value);
        }
        if (sidebarId === 'hqApiKey') {
            state.setHqApiKey(value || '');
        }
        if (sidebarId === 'hqModelName') {
            state.setHqModelName(value || '');
        }
        if (sidebarId === 'hqCustomBaseUrl') {
            state.setHqCustomBaseUrl(value || '');
        }
        if (sidebarId === 'hqBatchSize' && value) {
            state.setHqBatchSize(value);
        }
        if (sidebarId === 'hqSessionReset' && value) {
            state.setHqSessionReset(value);
        }
        if (sidebarId === 'hqRpmLimit' && value) {
            state.setHqRpmLimit(value);
        }
        if (sidebarId === 'hqLowReasoning') {
            state.setHqLowReasoning(!!value);
        }
        if (sidebarId === 'hqNoThinkingMethod' && value) {
            state.setHqNoThinkingMethod(value);
        }
        if (sidebarId === 'hqForceJsonOutput') {
            state.setHqForceJsonOutput(!!value);
        }
        if (sidebarId === 'hqUseStream') {
            state.setHqUseStream(!!value);
        }
        if (sidebarId === 'hqMaxRetries' && value) {
            state.setHqTranslationMaxRetries(value);
        }
        // RPM限制
        if (sidebarId === 'rpmTranslation' && value) {
            state.setrpmLimitTranslation(value);
        }
        if (sidebarId === 'rpmAiVisionOcr' && value) {
            state.setrpmLimitAiVisionOcr(value);
        }
        // 重试次数
        if (sidebarId === 'translationMaxRetries' && value) {
            state.setTranslationMaxRetries(value);
        }
        // PDF处理方式
        if (sidebarId === 'pdfProcessingMethod' && value) {
            state.setPdfProcessingMethod(value);
        }
    }
    
    // 更新模态框中的条件显示
    updateOcrOptionsDisplay();
    updateTranslateOptionsDisplay();
    updateAiVisionOptionsDisplay();
    updateHqOptionsDisplay();
    
    // 更新精确掩膜选项的显示（根据检测器类型）
    const textDetectorEl = document.getElementById('settingsTextDetector');
    if (textDetectorEl) {
        updatePreciseMaskVisibility(textDetectorEl.value);
    }
    
    // 更新文本框提示词区域显示
    const enableTextboxPrompt = document.getElementById('settingsEnableTextboxPrompt');
    const textboxPromptArea = document.getElementById('settingsTextboxPromptArea');
    if (enableTextboxPrompt && textboxPromptArea) {
        textboxPromptArea.style.display = enableTextboxPrompt.checked ? 'block' : 'none';
    }
    
    // 应用AI校对设置
    if (settings.proofreading) {
        applyProofreadingSettings(settings.proofreading);
    }
    
    // 应用检测设置到 state（直接更新 state）
    if (settings.textDetector !== undefined) {
        state.setTextDetector(settings.textDetector);
    }
    // 应用文本框扩展参数到 state
    if (settings.boxExpandRatio !== undefined) {
        state.setBoxExpandRatio(parseInt(settings.boxExpandRatio) || 0);
    }
    if (settings.boxExpandTop !== undefined) {
        state.setBoxExpandTop(parseInt(settings.boxExpandTop) || 0);
    }
    if (settings.boxExpandBottom !== undefined) {
        state.setBoxExpandBottom(parseInt(settings.boxExpandBottom) || 0);
    }
    if (settings.boxExpandLeft !== undefined) {
        state.setBoxExpandLeft(parseInt(settings.boxExpandLeft) || 0);
    }
    if (settings.boxExpandRight !== undefined) {
        state.setBoxExpandRight(parseInt(settings.boxExpandRight) || 0);
    }
    if (settings.showDetectionDebug !== undefined) {
        state.setShowDetectionDebug(settings.showDetectionDebug);
    }
    // 应用精确文字掩膜设置到 state
    if (settings.usePreciseMask !== undefined) {
        state.setUsePreciseMask(settings.usePreciseMask);
    }
    // 应用掩膜膨胀系数到 state
    if (settings.maskDilateSize !== undefined) {
        state.setMaskDilateSize(parseInt(settings.maskDilateSize) || 0);
    }
    // 应用标注框区域扩大比例到 state
    if (settings.maskBoxExpandRatio !== undefined) {
        state.setMaskBoxExpandRatio(parseInt(settings.maskBoxExpandRatio) || 0);
    }
    
    // 应用 AI 视觉 OCR 自定义 Base URL 到 state
    if (settings.customAiVisionBaseUrl !== undefined) {
        state.setCustomAiVisionBaseUrl(settings.customAiVisionBaseUrl);
    }
    
    // 应用高质量翻译设置到 state
    if (settings.hqTranslateProvider !== undefined) {
        state.setHqTranslateProvider(settings.hqTranslateProvider);
    }
    if (settings.hqApiKey !== undefined) {
        state.setHqApiKey(settings.hqApiKey);
    }
    if (settings.hqModelName !== undefined) {
        state.setHqModelName(settings.hqModelName);
    }
    if (settings.hqCustomBaseUrl !== undefined) {
        state.setHqCustomBaseUrl(settings.hqCustomBaseUrl);
    }
    if (settings.hqBatchSize !== undefined) {
        state.setHqBatchSize(parseInt(settings.hqBatchSize) || 3);
    }
    if (settings.hqSessionReset !== undefined) {
        state.setHqSessionReset(parseInt(settings.hqSessionReset) || 20);
    }
    if (settings.hqRpmLimit !== undefined) {
        state.setHqRpmLimit(parseInt(settings.hqRpmLimit) || 7);
    }
    if (settings.hqMaxRetries !== undefined) {
        const value = parseInt(settings.hqMaxRetries);
        state.setHqTranslationMaxRetries(value);
    }
    if (settings.hqLowReasoning !== undefined) {
        state.setHqLowReasoning(settings.hqLowReasoning);
    }
    if (settings.hqNoThinkingMethod !== undefined) {
        state.setHqNoThinkingMethod(settings.hqNoThinkingMethod);
    }
    if (settings.hqForceJsonOutput !== undefined) {
        state.setHqForceJsonOutput(settings.hqForceJsonOutput);
    }
    if (settings.hqUseStream !== undefined) {
        state.setHqUseStream(settings.hqUseStream);
    }
    // 高质量翻译提示词：空字符串时使用默认值
    if (settings.hqPrompt !== undefined && settings.hqPrompt.trim() !== '') {
        state.setHqPrompt(settings.hqPrompt);
    } else if (settings.hqPrompt === '' || settings.hqPrompt === undefined) {
        // 确保使用默认值
        state.setHqPrompt(constants.DEFAULT_HQ_TRANSLATE_PROMPT);
    }
    
    // 应用 PDF 处理方式到 state
    if (settings.pdfProcessingMethod !== undefined) {
        state.setPdfProcessingMethod(settings.pdfProcessingMethod);
    }
    
    // ===== 初始化当前服务商值的记录 =====
    currentProviderValues.ocrEngine = document.getElementById('settingsOcrEngine')?.value || null;
    currentProviderValues.aiVisionProvider = document.getElementById('settingsAiVisionProvider')?.value || null;
    currentProviderValues.modelProvider = document.getElementById('settingsModelProvider')?.value || null;
    currentProviderValues.hqTranslateProvider = document.getElementById('settingsHqTranslateProvider')?.value || null;
}

/**
 * 从 localStorage 加载设置（回退方案）
 */
function loadSettingsFromLocalStorage() {
    try {
        const stored = localStorage.getItem(SETTINGS_STORAGE_KEY);
        if (stored) {
            const settings = JSON.parse(stored);
            applySettingsToUI(settings);
        }
    } catch (e) {
        console.error('从 localStorage 加载设置失败:', e);
    }
}

/**
 * 显示Toast提示
 * @param {string} message - 消息内容
 * @param {string} type - 类型: 'success', 'error', 'warning', 'info'
 * @param {number} duration - 显示时长(毫秒)
 */
function showToast(message, type = 'info', duration = 3000) {
    // 移除之前的toast
    const existingToast = document.getElementById('settingsToast');
    if (existingToast) {
        existingToast.remove();
    }
    
    // 创建新toast元素
    const toast = document.createElement('div');
    toast.id = 'settingsToast';
    toast.className = `settings-toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    // 自动隐藏
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, duration);
}

/**
 * 绑定测试按钮事件
 */
export function bindTestButtons() {
    // 百度OCR测试
    const testBaiduOcr = document.getElementById('settingsTestBaiduOcr');
    if (testBaiduOcr) {
        testBaiduOcr.addEventListener('click', () => {
            syncToSidebar();
            testBaiduOcrConnection();
        });
    }
    
    // AI视觉OCR测试
    const testAiVisionOcr = document.getElementById('settingsTestAiVisionOcr');
    if (testAiVisionOcr) {
        testAiVisionOcr.addEventListener('click', () => {
            syncToSidebar();
            testAiVisionOcrConnection();
        });
    }
    
    // 翻译服务测试
    const testTranslation = document.getElementById('settingsTestTranslation');
    if (testTranslation) {
        testTranslation.addEventListener('click', () => {
            syncToSidebar();
            // 根据服务商触发相应的测试
            const provider = document.getElementById('settingsModelProvider')?.value;
            testTranslationConnection(provider);
        });
    }
}

/**
 * 测试百度OCR连接
 */
async function testBaiduOcrConnection() {
    const apiKey = document.getElementById('settingsBaiduApiKey')?.value?.trim();
    const secretKey = document.getElementById('settingsBaiduSecretKey')?.value?.trim();
    
    if (!apiKey || !secretKey) {
        showToast('请填写百度OCR的API Key和Secret Key', 'warning');
        return;
    }
    
    showToast('正在测试百度OCR连接...', 'info');
    try {
        const response = await fetch('/api/test_baidu_ocr_connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: apiKey, secret_key: secretKey })
        });
        const data = await response.json();
        if (data.success) {
            showToast(data.message || '百度OCR连接成功!', 'success');
        } else {
            showToast(data.message || '百度OCR连接失败', 'error');
        }
    } catch (error) {
        showToast('百度OCR连接测试出错: ' + error.message, 'error');
    }
}

/**
 * 测试AI视觉OCR连接
 */
async function testAiVisionOcrConnection() {
    const provider = document.getElementById('settingsAiVisionProvider')?.value;
    const apiKey = document.getElementById('settingsAiVisionApiKey')?.value?.trim();
    const modelName = document.getElementById('settingsAiVisionModelName')?.value?.trim();
    const baseUrl = document.getElementById('settingsCustomAiVisionBaseUrl')?.value?.trim();
    
    if (!apiKey) {
        showToast('请填写API Key', 'warning');
        return;
    }
    if (!modelName) {
        showToast('请填写模型名称', 'warning');
        return;
    }
    if (provider === 'custom_openai_vision' && !baseUrl) {
        showToast('自定义服务需要填写Base URL', 'warning');
        return;
    }
    
    showToast('正在测试AI视觉OCR连接...', 'info');
    try {
        const response = await fetch('/api/test_ai_vision_ocr', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: provider,
                api_key: apiKey,
                model_name: modelName,
                base_url: baseUrl
            })
        });
        const data = await response.json();
        if (data.success) {
            showToast(data.message || 'AI视觉OCR连接成功!', 'success');
        } else {
            showToast(data.message || 'AI视觉OCR连接失败', 'error');
        }
    } catch (error) {
        showToast('AI视觉OCR连接测试出错: ' + error.message, 'error');
    }
}

/**
 * 统一的翻译服务连接测试
 */
async function testTranslationConnection(provider) {
    const apiKey = document.getElementById('settingsApiKey')?.value?.trim();
    const modelName = document.getElementById('settingsModelName')?.value?.trim();
    const baseUrl = document.getElementById('settingsCustomBaseUrl')?.value?.trim();
    
    // 根据服务商类型分发到不同的测试函数
    switch (provider) {
        case 'ollama':
            await testOllamaConnection();
            break;
        case 'sakura':
            await testSakuraConnection();
            break;
        case 'baidu_translate':
            await testBaiduTranslateConnection(apiKey, modelName);
            break;
        case 'youdao_translate':
            await testYoudaoTranslateConnection(apiKey, modelName);
            break;
        case 'siliconflow':
        case 'deepseek':
        case 'volcano':
        case 'gemini':
        case 'caiyun':
        case 'custom_openai':
            await testAITranslateConnection(provider, apiKey, modelName, baseUrl);
            break;
        default:
            showToast('未知的服务商类型', 'warning');
    }
}

/**
 * 测试Ollama连接并获取模型列表
 */
async function testOllamaConnection() {
    showToast('正在测试Ollama连接...', 'info');
    try {
        const response = await fetch('/api/test_ollama_connection');
        const data = await response.json();
        if (data.success) {
            showToast(`Ollama连接成功! 版本: ${data.version || '未知'}`, 'success');
            // 直接从返回数据中获取模型列表
            if (data.models && data.models.length > 0) {
                renderModelListForSettings('settingsOllamaModelsList', data.models);
            } else {
                showToast('Ollama已连接，但未找到已安装的模型', 'warning');
            }
        } else {
            showToast('Ollama连接失败: ' + (data.message || '未知错误'), 'error');
        }
    } catch (error) {
        showToast('Ollama连接测试出错: ' + error.message, 'error');
    }
}

/**
 * 测试Sakura连接并获取模型列表
 */
async function testSakuraConnection() {
    showToast('正在测试Sakura连接...', 'info');
    try {
        const response = await fetch('/api/test_sakura_connection?force=true');
        const data = await response.json();
        if (data.success) {
            showToast('Sakura连接成功!', 'success');
            // 直接从返回数据中获取模型列表
            if (data.models && data.models.length > 0) {
                renderModelListForSettings('settingsSakuraModelsList', data.models);
            }
        } else {
            showToast('Sakura连接失败: ' + (data.message || '未知错误'), 'error');
        }
    } catch (error) {
        showToast('Sakura连接测试出错: ' + error.message, 'error');
    }
}

/**
 * 测试百度翻译连接
 */
async function testBaiduTranslateConnection(appId, appKey) {
    if (!appId || !appKey) {
        showToast('请填写App ID和App Key', 'warning');
        return;
    }
    
    showToast('正在测试百度翻译连接...', 'info');
    try {
        const response = await fetch('/api/test_baidu_translate_connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ app_id: appId, app_key: appKey })
        });
        const data = await response.json();
        if (data.success) {
            showToast(data.message || '百度翻译连接成功!', 'success');
        } else {
            showToast('百度翻译连接失败: ' + (data.message || '未知错误'), 'error');
        }
    } catch (error) {
        showToast('百度翻译连接测试出错: ' + error.message, 'error');
    }
}

/**
 * 测试有道翻译连接
 */
async function testYoudaoTranslateConnection(appKey, appSecret) {
    if (!appKey || !appSecret) {
        showToast('请填写App Key和App Secret', 'warning');
        return;
    }
    
    showToast('正在测试有道翻译连接...', 'info');
    try {
        const response = await fetch('/api/test_youdao_translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ appKey: appKey, appSecret: appSecret })
        });
        const data = await response.json();
        if (data.success) {
            showToast(data.message || '有道翻译连接成功!', 'success');
        } else {
            showToast('有道翻译连接失败: ' + (data.message || '未知错误'), 'error');
        }
    } catch (error) {
        showToast('有道翻译连接测试出错: ' + error.message, 'error');
    }
}

/**
 * 测试AI翻译服务连接 (SiliconFlow, DeepSeek, 火山引擎, Gemini, 彩云小译, 自定义OpenAI)
 */
async function testAITranslateConnection(provider, apiKey, modelName, baseUrl) {
    // 彩云小译不需要模型名称
    if (provider === 'caiyun') {
        if (!apiKey) {
            showToast('请填写彩云小译Token', 'warning');
            return;
        }
    } else {
        if (!apiKey) {
            showToast('请填写API Key', 'warning');
            return;
        }
        if (!modelName) {
            showToast('请填写模型名称', 'warning');
            return;
        }
    }
    
    // 自定义服务需要Base URL
    if (provider === 'custom_openai' && !baseUrl) {
        showToast('自定义服务需要填写Base URL', 'warning');
        return;
    }
    
    const providerNames = {
        'siliconflow': 'SiliconFlow',
        'deepseek': 'DeepSeek',
        'volcano': '火山引擎',
        'gemini': 'Google Gemini',
        'caiyun': '彩云小译',
        'custom_openai': '自定义服务'
    };
    
    showToast(`正在测试${providerNames[provider] || provider}连接...`, 'info');
    
    try {
        const response = await fetch('/api/test_ai_translate_connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: provider,
                api_key: apiKey,
                model_name: modelName,
                base_url: baseUrl
            })
        });
        const data = await response.json();
        if (data.success) {
            showToast(data.message || '连接成功!', 'success');
        } else {
            showToast(data.message || '连接失败', 'error');
        }
    } catch (error) {
        showToast('连接测试出错: ' + error.message, 'error');
    }
}

/**
 * 渲染模型列表到设置面板
 */
function renderModelListForSettings(containerId, models) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = '';
    models.forEach(model => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'model-button';
        btn.textContent = model;
        btn.addEventListener('click', () => {
            // 移除其他按钮的选中状态
            container.querySelectorAll('.model-button').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            // 更新隐藏的模型名称输入框
            const modelInput = document.getElementById('settingsModelName');
            if (modelInput) modelInput.value = model;
        });
        container.appendChild(btn);
    });
    
    // 更新提示文字
    const hint = document.getElementById('settingsLocalModelHint');
    if (hint) {
        hint.textContent = models.length > 0 ? '点击选择要使用的模型' : '未找到可用模型';
    }
}

/**
 * 精简侧边栏 - 隐藏已移到设置模态框的配置项
 * 为不支持CSS :has()选择器的浏览器提供JavaScript备选方案
 */
function simplifySidebar() {
    // 获取需要隐藏的元素
    const elementsToHide = [
        // 文本检测器选择
        document.getElementById('textDetector')?.closest('div'),
        // 检测框扩展选项
        document.getElementById('detectorExpandOptions'),
        // OCR引擎选择
        document.getElementById('ocrEngine')?.closest('div'),
        // PaddleOCR选项
        document.getElementById('paddleOcrOptions'),
        // 百度OCR选项
        document.getElementById('baiduOcrOptions'),
        // AI视觉OCR选项
        document.getElementById('aiVisionOcrOptions'),
        // 翻译服务相关元素
        document.getElementById('modelProvider')?.closest('div'),
        document.getElementById('apiKey')?.closest('div'),
        document.getElementById('modelName')?.closest('div'),
        document.getElementById('customBaseUrl')?.closest('div'),
        document.getElementById('rpmTranslation')?.closest('div'),
        document.getElementById('translationMaxRetries')?.closest('div'),
    ];
    
    // 添加隐藏类
    elementsToHide.forEach(el => {
        if (el) {
            el.classList.add('sidebar-hidden-config');
        }
    });
}

// ===== AI校对设置相关函数 =====

/**
 * 初始化AI校对设置UI
 */
function initProofreadingSettingsUI() {
    // 同步启用状态
    const enabledCheckbox = document.getElementById('settingsProofreadingEnabled');
    const originalCheckbox = document.getElementById('proofreadingEnabled');
    if (enabledCheckbox && originalCheckbox) {
        enabledCheckbox.checked = originalCheckbox.checked;
        
        // 双向同步
        enabledCheckbox.addEventListener('change', () => {
            originalCheckbox.checked = enabledCheckbox.checked;
            originalCheckbox.dispatchEvent(new Event('change', { bubbles: true }));
        });
    }
    
    // 绑定添加轮次按钮
    const addRoundBtn = document.getElementById('settingsAddRoundButton');
    if (addRoundBtn) {
        addRoundBtn.addEventListener('click', addNewProofreadingRound);
    }
    
    // 初始化轮次列表
    renderProofreadingRounds();
}

/**
 * 渲染校对轮次列表
 */
function renderProofreadingRounds() {
    const container = document.getElementById('settingsProofreadingRoundsContainer');
    if (!container) return;
    
    container.innerHTML = '';
    
    // 从state模块获取轮次配置
    const rounds = state.proofreadingRounds || [];
    
    if (rounds.length === 0) {
        // 添加一个默认轮次
        addNewProofreadingRound();
        return;
    }
    
    rounds.forEach((round, index) => {
        addRoundToSettingsUI(round, index);
    });
    
    // 渲染轮次内的提示词选择器
    setTimeout(() => renderProofreadingPromptPickers(), 100);
}

/**
 * 添加新的校对轮次
 */
function addNewProofreadingRound() {
    // 使用导入的state模块
    if (!state.proofreadingRounds || state.proofreadingRounds.length === undefined) {
        // 如果数组不存在，通过 setter 初始化
        if (state.setProofreadingRounds) {
            state.setProofreadingRounds([]);
        }
    }
    
    const newRound = {
        name: `轮次 ${state.proofreadingRounds.length + 1}`,
        provider: 'siliconflow',
        apiKey: '',
        modelName: '',
        customBaseUrl: '',
        batchSize: 3,
        sessionReset: 20,
        rpmLimit: 7,
        lowReasoning: false,
        forceJsonOutput: true,
        prompt: constants.DEFAULT_PROOFREADING_PROMPT
    };
    
    state.proofreadingRounds.push(newRound);
    addRoundToSettingsUI(newRound, state.proofreadingRounds.length - 1);
    
    // 同步到原有的容器
    syncProofreadingToOriginal();
    
    // 渲染轮次内的提示词选择器
    setTimeout(() => renderProofreadingPromptPickers(), 100);
}

/**
 * 将轮次添加到设置模态框UI
 */
function addRoundToSettingsUI(round, index) {
    const container = document.getElementById('settingsProofreadingRoundsContainer');
    if (!container) return;
    
    const roundDiv = document.createElement('div');
    roundDiv.className = 'settings-proofreading-round';
    roundDiv.dataset.index = index;
    
    roundDiv.innerHTML = `
        <div class="settings-group" style="margin-bottom: 15px; border: 1px solid var(--border-color);">
            <div class="settings-group-title" style="display: flex; justify-content: space-between; align-items: center;">
                <input type="text" class="round-name-input" value="${round.name}" placeholder="轮次名称" 
                       style="border: none; background: transparent; font-weight: 600; font-size: 1em; width: auto;">
                <button class="remove-round-btn" style="background: #dc3545; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer;">删除</button>
            </div>
            <div class="settings-row">
                <div class="settings-item">
                    <label>AI服务商:</label>
                    <select class="round-provider">
                        <option value="siliconflow" ${round.provider === 'siliconflow' ? 'selected' : ''}>SiliconFlow</option>
                        <option value="deepseek" ${round.provider === 'deepseek' ? 'selected' : ''}>DeepSeek</option>
                        <option value="volcano" ${round.provider === 'volcano' ? 'selected' : ''}>火山引擎</option>
                        <option value="gemini" ${round.provider === 'gemini' ? 'selected' : ''}>Google Gemini</option>
                        <option value="custom_openai" ${round.provider === 'custom_openai' ? 'selected' : ''}>自定义OpenAI兼容服务</option>
                    </select>
                </div>
                <div class="settings-item">
                    <label>API Key:</label>
                    <div class="password-input-wrapper">
                        <input type="text" class="round-api-key secure-input" value="${round.apiKey || ''}" placeholder="填写API Key" autocomplete="off">
                        <button type="button" class="password-toggle-btn" tabindex="-1">
                            <span class="eye-icon">👁</span>
                            <span class="eye-off-icon">👁‍🗨</span>
                        </button>
                    </div>
                </div>
            </div>
            <div class="settings-item">
                <label>模型名称:</label>
                <div class="model-input-with-fetch">
                    <input type="text" class="round-model-name" value="${round.modelName || ''}" placeholder="如 gemini-2.5-flash-preview-05-20">
                    <button type="button" class="fetch-models-btn round-fetch-models-btn" title="获取可用模型列表">
                        <span class="fetch-icon">🔍</span>
                        <span class="fetch-text">获取模型</span>
                    </button>
                </div>
                <div class="round-model-select-container model-select-container" style="display:none;">
                    <select class="round-model-select model-select">
                        <option value="">-- 选择模型 --</option>
                    </select>
                    <span class="round-model-count model-count"></span>
                </div>
            </div>
            <div class="settings-item custom-base-url-row" style="${round.provider === 'custom_openai' ? '' : 'display:none;'}">
                <label>Base URL:</label>
                <input type="text" class="round-custom-base-url" value="${round.customBaseUrl || ''}" placeholder="如 https://your-api-endpoint.com">
            </div>
            <div class="settings-row">
                <div class="settings-item">
                    <label>批次大小:</label>
                    <input type="number" class="round-batch-size" value="${round.batchSize || 3}" min="1" max="10" style="width: 80px;">
                </div>
                <div class="settings-item">
                    <label>会话重置频率:</label>
                    <input type="number" class="round-session-reset" value="${round.sessionReset || 20}" min="1" style="width: 80px;">
                </div>
                <div class="settings-item">
                    <label>RPM限制:</label>
                    <input type="number" class="round-rpm-limit" value="${round.rpmLimit || 7}" min="1" max="100" style="width: 80px;">
                </div>
            </div>
            <div class="settings-row">
                <div class="settings-item">
                    <label class="settings-checkbox">
                        <input type="checkbox" class="round-low-reasoning" ${round.lowReasoning ? 'checked' : ''}>
                        <span>关闭思考功能</span>
                    </label>
                </div>
                <div class="settings-item">
                    <label class="settings-checkbox">
                        <input type="checkbox" class="round-force-json" ${round.forceJsonOutput ? 'checked' : ''}>
                        <span>强制JSON输出</span>
                    </label>
                </div>
            </div>
            <div class="settings-item">
                <label>校对提示词:</label>
                <textarea class="round-prompt" rows="4" style="width: 100%;">${round.prompt || ''}</textarea>
            </div>
        </div>
    `;
    
    container.appendChild(roundDiv);
    
    // 绑定事件
    bindRoundEvents(roundDiv, index);
}

/**
 * 绑定轮次的事件
 */
function bindRoundEvents(roundDiv, index) {
    // 删除按钮
    roundDiv.querySelector('.remove-round-btn')?.addEventListener('click', () => {
        if (state.proofreadingRounds && state.proofreadingRounds.length > 1) {
            state.proofreadingRounds.splice(index, 1);
            renderProofreadingRounds();
            syncProofreadingToOriginal();
        } else {
            showToast('至少需要保留一个校对轮次');
        }
    });
    
    // 服务商变更 - 显示/隐藏Base URL
    roundDiv.querySelector('.round-provider')?.addEventListener('change', function() {
        const customUrlRow = roundDiv.querySelector('.custom-base-url-row');
        if (customUrlRow) {
            customUrlRow.style.display = this.value === 'custom_openai' ? '' : 'none';
        }
        updateRoundFromUI(roundDiv, index);
    });
    
    // 其他输入框变更
    const inputs = roundDiv.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('change', () => updateRoundFromUI(roundDiv, index));
        input.addEventListener('blur', () => updateRoundFromUI(roundDiv, index));
    });
}

/**
 * 从UI更新轮次配置
 */
function updateRoundFromUI(roundDiv, index) {
    // 使用导入的state模块
    if (!state.proofreadingRounds || !state.proofreadingRounds[index]) return;
    
    const round = state.proofreadingRounds[index];
    
    round.name = roundDiv.querySelector('.round-name-input')?.value || `轮次 ${index + 1}`;
    round.provider = roundDiv.querySelector('.round-provider')?.value || 'siliconflow';
    round.apiKey = roundDiv.querySelector('.round-api-key')?.value || '';
    round.modelName = roundDiv.querySelector('.round-model-name')?.value || '';
    round.customBaseUrl = roundDiv.querySelector('.round-custom-base-url')?.value || '';
    round.batchSize = parseInt(roundDiv.querySelector('.round-batch-size')?.value) || 3;
    round.sessionReset = parseInt(roundDiv.querySelector('.round-session-reset')?.value) || 20;
    round.rpmLimit = parseInt(roundDiv.querySelector('.round-rpm-limit')?.value) || 7;
    round.lowReasoning = roundDiv.querySelector('.round-low-reasoning')?.checked || false;
    round.forceJsonOutput = roundDiv.querySelector('.round-force-json')?.checked || false;
    round.prompt = roundDiv.querySelector('.round-prompt')?.value || '';
    
    syncProofreadingToOriginal();
}

/**
 * 同步校对设置到原有的容器（保持兼容性）
 */
function syncProofreadingToOriginal() {
    // 同步启用状态
    const settingsEnabled = document.getElementById('settingsProofreadingEnabled');
    const originalEnabled = document.getElementById('proofreadingEnabled');
    if (settingsEnabled && originalEnabled) {
        originalEnabled.checked = settingsEnabled.checked;
        if (state.setProofreadingEnabled) {
            state.setProofreadingEnabled(settingsEnabled.checked);
        }
    }
    
    // 触发原有ai_proofreading模块的更新（如果存在）
    if (typeof window.updateProofreadingState === 'function') {
        window.updateProofreadingState();
    }
}

// ===== 提示词库管理 =====

// 提示词库缓存
let promptLibrary = [];
// 当前正在编辑的提示词索引（-1表示新建）
let editingPromptIndex = -1;

/**
 * 初始化提示词管理功能
 */
function initPromptLibrary() {
    // 绑定新建提示词按钮
    const addBtn = document.getElementById('addNewPromptBtn');
    if (addBtn) {
        addBtn.addEventListener('click', () => showPromptEditArea(-1));
    }
    
    // 绑定取消按钮
    const cancelBtn = document.getElementById('cancelPromptEditBtn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', hidePromptEditArea);
    }
    
    // 绑定保存按钮
    const saveBtn = document.getElementById('savePromptToLibraryBtn');
    if (saveBtn) {
        saveBtn.addEventListener('click', savePromptToLibrary);
    }
    
    // 加载提示词库
    loadPromptLibrary();
}

/**
 * 从后端加载提示词库
 */
async function loadPromptLibrary() {
    try {
        const response = await fetch('/api/get_prompts');
        const data = await response.json();
        
        if (data.prompt_names && Array.isArray(data.prompt_names)) {
            // 将名称列表转换为完整对象列表
            promptLibrary = [];
            for (const name of data.prompt_names) {
                // 获取每个提示词的内容 (使用GET方法)
                const contentRes = await fetch(`/api/get_prompt_content?prompt_name=${encodeURIComponent(name)}`);
                const contentData = await contentRes.json();
                if (contentData.prompt_content) {
                    promptLibrary.push({
                        name: name,
                        content: contentData.prompt_content
                    });
                }
            }
        }
        
        // 渲染提示词库列表和所有选择器
        renderPromptLibraryList();
        renderAllPromptPickers();
    } catch (e) {
        console.error('加载提示词库失败:', e);
    }
}

/**
 * 渲染提示词库列表
 */
function renderPromptLibraryList() {
    const container = document.getElementById('promptLibraryList');
    if (!container) return;
    
    if (promptLibrary.length === 0) {
        container.innerHTML = '<div class="empty-prompt-hint">暂无保存的提示词，点击上方按钮创建</div>';
        return;
    }
    
    container.innerHTML = promptLibrary.map((prompt, index) => `
        <div class="prompt-library-item" data-index="${index}">
            <div class="prompt-info">
                <div class="prompt-name">
                    <span class="prompt-icon">📝</span>
                    ${escapeHtml(prompt.name)}
                </div>
                <div class="prompt-preview">${escapeHtml(prompt.content.substring(0, 80))}${prompt.content.length > 80 ? '...' : ''}</div>
            </div>
            <div class="prompt-actions">
                <button class="prompt-action-btn edit-btn" data-index="${index}">编辑</button>
                <button class="prompt-action-btn delete-btn" data-index="${index}">删除</button>
            </div>
        </div>
    `).join('');
    
    // 绑定编辑和删除按钮事件
    container.querySelectorAll('.edit-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.dataset.index);
            showPromptEditArea(idx);
        });
    });
    
    container.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.dataset.index);
            deletePromptFromLibrary(idx);
        });
    });
}

/**
 * 渲染所有提示词选择器
 */
function renderAllPromptPickers() {
    // 获取所有提示词选择器容器
    const pickerIds = [
        'translatePromptsChips',      // 翻译服务 - 漫画翻译
        'textboxPromptsChips',        // 翻译服务 - 文本框
        'aiVisionOcrPromptsChips',    // AI视觉OCR
        'hqPromptsChips',             // 高质量翻译
    ];
    
    pickerIds.forEach(id => {
        const container = document.getElementById(id);
        if (container) {
            renderPromptPicker(container);
        }
    });
    
    // 为AI校对轮次的提示词也添加选择器
    renderProofreadingPromptPickers();
}

/**
 * 渲染单个提示词选择器
 */
function renderPromptPicker(container) {
    if (promptLibrary.length === 0) {
        container.innerHTML = '<span class="empty-hint">暂无保存的提示词</span>';
        return;
    }
    
    container.innerHTML = promptLibrary.map((prompt, index) => `
        <button type="button" class="prompt-chip" data-index="${index}" title="${escapeHtml(prompt.content.substring(0, 100))}">
            <span class="chip-icon">📝</span>
            ${escapeHtml(prompt.name)}
        </button>
    `).join('');
    
    // 获取目标textarea的ID
    const picker = container.closest('.saved-prompts-picker');
    const targetId = picker?.dataset?.target;
    
    // 绑定点击事件
    container.querySelectorAll('.prompt-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const idx = parseInt(chip.dataset.index);
            if (promptLibrary[idx] && targetId) {
                const textarea = document.getElementById(targetId);
                if (textarea) {
                    textarea.value = promptLibrary[idx].content;
                    textarea.dispatchEvent(new Event('change', { bubbles: true }));
                    showToast(`已应用提示词: ${promptLibrary[idx].name}`, 'success', 2000);
                }
            }
        });
    });
}

/**
 * 为AI校对轮次渲染提示词选择器
 */
function renderProofreadingPromptPickers() {
    // 找到所有校对轮次的提示词textarea
    const roundDivs = document.querySelectorAll('.settings-proofreading-round');
    roundDivs.forEach((roundDiv, index) => {
        const promptTextarea = roundDiv.querySelector('.round-prompt');
        if (!promptTextarea) return;
        
        // 检查是否已存在选择器
        let picker = roundDiv.querySelector('.saved-prompts-picker');
        if (!picker) {
            // 创建选择器
            picker = document.createElement('div');
            picker.className = 'saved-prompts-picker';
            picker.dataset.target = `round-prompt-${index}`;
            picker.innerHTML = `
                <span class="picker-label">📑 快速选择:</span>
                <div class="prompts-chips-container"></div>
            `;
            promptTextarea.parentNode.appendChild(picker);
            promptTextarea.id = `round-prompt-${index}`;
        }
        
        // 渲染选择器内容
        const chipsContainer = picker.querySelector('.prompts-chips-container');
        if (chipsContainer) {
            renderPromptPicker(chipsContainer);
        }
    });
}

/**
 * 显示提示词编辑区域
 */
function showPromptEditArea(index) {
    editingPromptIndex = index;
    const editArea = document.getElementById('promptEditArea');
    const titleEl = document.getElementById('promptEditTitle');
    const nameInput = document.getElementById('promptLibraryName');
    const contentInput = document.getElementById('promptLibraryContent');
    
    if (!editArea || !nameInput || !contentInput) return;
    
    if (index === -1) {
        // 新建模式
        titleEl.textContent = '新建提示词';
        nameInput.value = '';
        contentInput.value = '';
    } else {
        // 编辑模式
        titleEl.textContent = '编辑提示词';
        const prompt = promptLibrary[index];
        if (prompt) {
            nameInput.value = prompt.name;
            contentInput.value = prompt.content;
        }
    }
    
    editArea.style.display = 'block';
    nameInput.focus();
}

/**
 * 隐藏提示词编辑区域
 */
function hidePromptEditArea() {
    const editArea = document.getElementById('promptEditArea');
    if (editArea) {
        editArea.style.display = 'none';
    }
    editingPromptIndex = -1;
}

/**
 * 保存提示词到库
 */
async function savePromptToLibrary() {
    const nameInput = document.getElementById('promptLibraryName');
    const contentInput = document.getElementById('promptLibraryContent');
    
    const name = nameInput?.value?.trim();
    const content = contentInput?.value?.trim();
    
    if (!name) {
        showToast('请输入提示词名称', 'warning');
        return;
    }
    
    if (!content) {
        showToast('请输入提示词内容', 'warning');
        return;
    }
    
    // 检查名称是否重复（排除正在编辑的那个）
    const duplicateIndex = promptLibrary.findIndex((p, i) => p.name === name && i !== editingPromptIndex);
    if (duplicateIndex !== -1) {
        showToast('已存在同名的提示词', 'warning');
        return;
    }
    
    try {
        // 保存到后端
        const response = await fetch('/api/save_prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt_name: name,
                prompt_content: content
            })
        });
        
        const data = await response.json();
        if (data.message) {
            // 如果是编辑模式且名称改变了，需要删除旧的
            if (editingPromptIndex !== -1) {
                const oldName = promptLibrary[editingPromptIndex]?.name;
                if (oldName && oldName !== name) {
                    await fetch('/api/delete_prompt', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt_name: oldName })
                    });
                }
            }
            
            showToast('提示词保存成功', 'success');
            hidePromptEditArea();
            
            // 重新加载提示词库
            await loadPromptLibrary();
        }
    } catch (e) {
        console.error('保存提示词失败:', e);
        showToast('保存提示词失败', 'error');
    }
}

/**
 * 从库中删除提示词
 */
async function deletePromptFromLibrary(index) {
    const prompt = promptLibrary[index];
    if (!prompt) return;
    
    if (!confirm(`确定要删除提示词 "${prompt.name}" 吗？`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/delete_prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt_name: prompt.name })
        });
        
        const data = await response.json();
        if (data.message) {
            showToast('提示词已删除', 'success');
            await loadPromptLibrary();
        }
    } catch (e) {
        console.error('删除提示词失败:', e);
        showToast('删除提示词失败', 'error');
    }
}

/**
 * HTML转义函数
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 初始化密码显示/隐藏切换按钮
 */
function initPasswordToggleButtons() {
    // 使用事件委托处理所有密码切换按钮
    document.addEventListener('click', (e) => {
        const toggleBtn = e.target.closest('.password-toggle-btn');
        if (!toggleBtn) return;
        
        const wrapper = toggleBtn.closest('.password-input-wrapper');
        if (!wrapper) return;
        
        const input = wrapper.querySelector('input');
        if (!input) return;
        
        // 切换 secure-input 类来显示/隐藏密码
        if (input.classList.contains('secure-input') && !input.classList.contains('showing')) {
            input.classList.add('showing');
            toggleBtn.classList.add('showing');
        } else {
            input.classList.remove('showing');
            toggleBtn.classList.remove('showing');
        }
    });
}

/**
 * 初始化模型获取按钮
 */
function initFetchModelsButtons() {
    // 翻译服务 - 获取模型按钮
    const fetchModelsBtn = document.getElementById('settingsFetchModelsBtn');
    if (fetchModelsBtn) {
        fetchModelsBtn.addEventListener('click', () => {
            fetchAndDisplayModels('translate');
        });
    }
    
    // 高质量翻译 - 获取模型按钮
    const hqFetchModelsBtn = document.getElementById('settingsHqFetchModelsBtn');
    if (hqFetchModelsBtn) {
        hqFetchModelsBtn.addEventListener('click', () => {
            fetchAndDisplayModels('hq');
        });
    }
    
    // AI视觉OCR - 获取模型按钮
    const aiVisionFetchModelsBtn = document.getElementById('settingsAiVisionFetchModelsBtn');
    if (aiVisionFetchModelsBtn) {
        aiVisionFetchModelsBtn.addEventListener('click', () => {
            fetchAndDisplayModels('aiVision');
        });
    }
    
    // 模型选择下拉框变化事件 - 翻译服务
    const modelSelect = document.getElementById('settingsModelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            const modelNameInput = document.getElementById('settingsModelName');
            if (modelNameInput && e.target.value) {
                modelNameInput.value = e.target.value;
            }
        });
    }
    
    // 模型选择下拉框变化事件 - 高质量翻译
    const hqModelSelect = document.getElementById('settingsHqModelSelect');
    if (hqModelSelect) {
        hqModelSelect.addEventListener('change', (e) => {
            const modelNameInput = document.getElementById('settingsHqModelName');
            if (modelNameInput && e.target.value) {
                modelNameInput.value = e.target.value;
            }
        });
    }
    
    // 模型选择下拉框变化事件 - AI视觉OCR
    const aiVisionModelSelect = document.getElementById('settingsAiVisionModelSelect');
    if (aiVisionModelSelect) {
        aiVisionModelSelect.addEventListener('change', (e) => {
            const modelNameInput = document.getElementById('settingsAiVisionModelName');
            if (modelNameInput && e.target.value) {
                modelNameInput.value = e.target.value;
            }
        });
    }
    
    // 使用事件委托处理校对轮次的模型获取按钮
    document.addEventListener('click', async (e) => {
        const fetchBtn = e.target.closest('.round-fetch-models-btn');
        if (!fetchBtn) return;
        
        // 支持多种容器类名
        const roundItem = fetchBtn.closest('.settings-proofreading-round') || fetchBtn.closest('.proofreading-round');
        if (!roundItem) return;
        
        await fetchRoundModels(roundItem, fetchBtn);
    });
    
    // 使用事件委托处理校对轮次的模型选择
    document.addEventListener('change', (e) => {
        if (e.target.classList.contains('round-model-select')) {
            const roundItem = e.target.closest('.settings-proofreading-round') || e.target.closest('.proofreading-round');
            if (!roundItem) return;
            
            const modelNameInput = roundItem.querySelector('.round-model-name');
            if (modelNameInput && e.target.value) {
                modelNameInput.value = e.target.value;
            }
        }
    });
}

/**
 * 获取校对轮次的模型列表
 */
async function fetchRoundModels(roundItem, fetchBtn) {
    await doFetchModels({
        providerSelect: roundItem.querySelector('.round-provider'),
        apiKeyInput: roundItem.querySelector('.round-api-key'),
        baseUrlInput: roundItem.querySelector('.round-custom-base-url'),
        modelSelectDiv: roundItem.querySelector('.round-model-select-container'),
        modelSelect: roundItem.querySelector('.round-model-select'),
        modelCount: roundItem.querySelector('.round-model-count'),
        modelNameInput: roundItem.querySelector('.round-model-name'),
        fetchBtn: fetchBtn
    });
}

/**
 * 获取并显示模型列表
 * @param {string} type - 'translate', 'hq', 或 'aiVision'
 */
async function fetchAndDisplayModels(type) {
    // 根据类型获取对应的元素 ID
    const elementConfig = {
        translate: {
            provider: 'settingsModelProvider',
            apiKey: 'settingsApiKey',
            baseUrl: 'settingsCustomBaseUrl',
            fetchBtn: 'settingsFetchModelsBtn',
            modelSelectDiv: 'settingsModelSelectDiv',
            modelSelect: 'settingsModelSelect',
            modelCount: 'settingsModelCount',
            modelNameInput: 'settingsModelName'
        },
        hq: {
            provider: 'settingsHqTranslateProvider',
            apiKey: 'settingsHqApiKey',
            baseUrl: 'settingsHqCustomBaseUrl',
            fetchBtn: 'settingsHqFetchModelsBtn',
            modelSelectDiv: 'settingsHqModelSelectDiv',
            modelSelect: 'settingsHqModelSelect',
            modelCount: 'settingsHqModelCount',
            modelNameInput: 'settingsHqModelName'
        },
        aiVision: {
            provider: 'settingsAiVisionProvider',
            apiKey: 'settingsAiVisionApiKey',
            baseUrl: 'settingsCustomAiVisionBaseUrl',
            fetchBtn: 'settingsAiVisionFetchModelsBtn',
            modelSelectDiv: 'settingsAiVisionModelSelectDiv',
            modelSelect: 'settingsAiVisionModelSelect',
            modelCount: 'settingsAiVisionModelCount',
            modelNameInput: 'settingsAiVisionModelName'
        }
    };
    
    const config = elementConfig[type];
    if (!config) {
        console.error('未知的类型:', type);
        return;
    }
    
    await doFetchModels({
        providerSelect: document.getElementById(config.provider),
        apiKeyInput: document.getElementById(config.apiKey),
        baseUrlInput: document.getElementById(config.baseUrl),
        fetchBtn: document.getElementById(config.fetchBtn),
        modelSelectDiv: document.getElementById(config.modelSelectDiv),
        modelSelect: document.getElementById(config.modelSelect),
        modelCount: document.getElementById(config.modelCount),
        modelNameInput: document.getElementById(config.modelNameInput)
    });
}

/**
 * 通用模型获取核心函数
 */
async function doFetchModels(elements) {
    const { providerSelect, apiKeyInput, baseUrlInput, fetchBtn, 
            modelSelectDiv, modelSelect, modelCount, modelNameInput } = elements;
    
    if (!providerSelect || !apiKeyInput) {
        console.error('获取模型: 找不到必要的元素');
        return;
    }
    
    let provider = providerSelect.value;
    const apiKey = apiKeyInput.value.trim();
    const baseUrl = baseUrlInput?.value.trim() || '';
    
    // 验证
    if (!apiKey) {
        alert('请先填写 API Key');
        apiKeyInput.focus();
        return;
    }
    
    // 检查是否支持模型获取
    const supportedProviders = ['siliconflow', 'deepseek', 'volcano', 'gemini', 'custom_openai', 'custom_openai_vision'];
    if (!supportedProviders.includes(provider)) {
        alert(`${getProviderName(provider)} 不支持自动获取模型列表`);
        return;
    }
    
    // 自定义服务需要 base_url
    if ((provider === 'custom_openai' || provider === 'custom_openai_vision') && !baseUrl) {
        alert('自定义服务需要先填写 Base URL');
        baseUrlInput?.focus();
        return;
    }
    
    // 将 custom_openai_vision 映射为 custom_openai 发送给后端
    const apiProvider = provider === 'custom_openai_vision' ? 'custom_openai' : provider;
    
    // 显示加载状态
    fetchBtn.classList.add('loading');
    fetchBtn.disabled = true;
    const originalText = fetchBtn.querySelector('.fetch-text').textContent;
    fetchBtn.querySelector('.fetch-text').textContent = '获取中...';
    
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
            const currentModel = modelNameInput?.value || '';
            if (currentModel) {
                modelSelect.value = currentModel;
            }
        } else {
            alert(data.message || '未获取到模型列表');
            modelSelectDiv.style.display = 'none';
        }
    } catch (error) {
        console.error('获取模型列表失败:', error);
        alert('获取模型列表失败: ' + error.message);
        modelSelectDiv.style.display = 'none';
    } finally {
        // 恢复按钮状态
        fetchBtn.classList.remove('loading');
        fetchBtn.disabled = false;
        fetchBtn.querySelector('.fetch-text').textContent = originalText;
    }
}

/**
 * 获取服务商显示名称
 */
function getProviderName(provider) {
    const names = {
        'siliconflow': 'SiliconFlow',
        'deepseek': 'DeepSeek',
        'volcano': '火山引擎',
        'gemini': 'Google Gemini',
        'custom_openai': '自定义OpenAI',
        'custom_openai_vision': '自定义OpenAI视觉',
        'ollama': 'Ollama',
        'sakura': 'Sakura',
        'caiyun': '彩云小译',
        'baidu_translate': '百度翻译',
        'youdao_translate': '有道翻译'
    };
    return names[provider] || provider;
}

/**
 * 刷新提示词选择器（供外部调用）
 */
export function refreshPromptPickers() {
    loadPromptLibrary();
}

// 导出初始化函数
export default {
    initSettingsModal,
    bindTestButtons,
    refreshPromptPickers
};
