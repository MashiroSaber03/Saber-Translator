// src/app/static/js/translation_validator.js
// 翻译配置验证模块

import * as ui from './ui.js';

// 本地存储键名
const DISMISS_SETUP_REMINDER_KEY = 'saber_translator_dismiss_setup_reminder';

// 需要 API Key 的服务商
const PROVIDERS_REQUIRING_API_KEY = [
    'siliconflow', 'deepseek', 'volcano', 'caiyun',
    'baidu_translate', 'youdao_translate', 'gemini', 'custom_openai'
];

// 本地服务商（需要模型名称，但不需要 API Key）
const LOCAL_PROVIDERS = ['ollama', 'sakura'];

// 需要自定义 Base URL 的服务商
const PROVIDERS_REQUIRING_BASE_URL = ['custom_openai'];

/**
 * 初始化翻译配置验证器
 * 页面加载时显示设置提醒弹窗（如果用户未选择不再显示）
 */
export function initTranslationValidator() {
    setTimeout(function() {
        checkAndShowSetupReminder();
    }, 500);
}

/**
 * 检查并显示设置提醒弹窗
 */
function checkAndShowSetupReminder() {
    var dismissed = localStorage.getItem(DISMISS_SETUP_REMINDER_KEY);
    if (dismissed === 'true') {
        console.log('用户已选择不再显示设置提醒');
        return;
    }
    showSetupReminderModal();
}

/**
 * 显示设置提醒弹窗
 */
function showSetupReminderModal() {
    if (document.getElementById('setupReminderModal')) {
        return;
    }

    var modalDiv = document.createElement('div');
    modalDiv.id = 'setupReminderModal';
    modalDiv.className = 'setup-reminder-overlay';
    modalDiv.innerHTML = [
        '<div class="setup-reminder-content">',
        '  <div class="setup-reminder-icon">&#9881;</div>',
        '  <h3 class="setup-reminder-title">欢迎使用 Saber-Translator</h3>',
        '  <p class="setup-reminder-text">',
        '    在开始翻译前，请先点击顶部的 <strong>&#9881; 设置</strong> 按钮，配置您的翻译服务商和 API Key。',
        '  </p>',
        '  <div class="setup-reminder-hint">',
        '    <p>&#128161; <strong>提示：</strong>支持 SiliconFlow、DeepSeek、火山引擎、Gemini 等多种 AI 翻译服务，也支持本地部署的 Ollama 和 Sakura。</p>',
        '  </div>',
        '  <div class="setup-reminder-buttons">',
        '    <button id="setupReminderGoSettings" class="setup-reminder-btn-primary">前往设置</button>',
        '    <button id="setupReminderClose" class="setup-reminder-btn-secondary">稍后设置</button>',
        '  </div>',
        '  <div class="setup-reminder-dismiss">',
        '    <label>',
        '      <input type="checkbox" id="setupReminderDismiss">',
        '      <span>不再显示此提醒</span>',
        '    </label>',
        '  </div>',
        '</div>'
    ].join('');

    document.body.appendChild(modalDiv);

    // 添加样式
    addSetupReminderStyles();

    // 绑定事件
    var goSettingsBtn = document.getElementById('setupReminderGoSettings');
    var closeBtn = document.getElementById('setupReminderClose');
    var dismissCheckbox = document.getElementById('setupReminderDismiss');

    goSettingsBtn.addEventListener('click', function() {
        closeSetupReminderModal(dismissCheckbox.checked);
        var settingsBtn = document.getElementById('openSettingsBtn');
        if (settingsBtn) {
            settingsBtn.click();
        }
    });

    closeBtn.addEventListener('click', function() {
        closeSetupReminderModal(dismissCheckbox.checked);
    });

    modalDiv.addEventListener('click', function(e) {
        if (e.target === modalDiv) {
            closeSetupReminderModal(dismissCheckbox.checked);
        }
    });
}

/**
 * 关闭设置提醒弹窗
 */
function closeSetupReminderModal(shouldDismiss) {
    if (shouldDismiss) {
        localStorage.setItem(DISMISS_SETUP_REMINDER_KEY, 'true');
        console.log('用户选择永久关闭设置提醒');
    }
    var modal = document.getElementById('setupReminderModal');
    if (modal) {
        modal.remove();
    }
}

/**
 * 添加设置提醒弹窗的 CSS 样式
 */
function addSetupReminderStyles() {
    if (document.getElementById('setupReminderStyles')) {
        return;
    }

    var style = document.createElement('style');
    style.id = 'setupReminderStyles';
    style.textContent = [
        '.setup-reminder-overlay {',
        '  position: fixed;',
        '  top: 0;',
        '  left: 0;',
        '  width: 100%;',
        '  height: 100%;',
        '  background: rgba(0, 0, 0, 0.5);',
        '  display: flex;',
        '  justify-content: center;',
        '  align-items: center;',
        '  z-index: 10000;',
        '}',
        '.setup-reminder-content {',
        '  background: var(--card-bg, #fff);',
        '  border-radius: 12px;',
        '  padding: 24px;',
        '  max-width: 450px;',
        '  width: 90%;',
        '  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);',
        '  animation: setupReminderFadeIn 0.3s ease;',
        '}',
        '@keyframes setupReminderFadeIn {',
        '  from { opacity: 0; transform: scale(0.9); }',
        '  to { opacity: 1; transform: scale(1); }',
        '}',
        '.setup-reminder-icon {',
        '  text-align: center;',
        '  font-size: 48px;',
        '  margin-bottom: 16px;',
        '}',
        '.setup-reminder-title {',
        '  margin: 0 0 12px 0;',
        '  text-align: center;',
        '  color: var(--text-primary, #333);',
        '  font-size: 18px;',
        '}',
        '.setup-reminder-text {',
        '  margin: 0 0 20px 0;',
        '  text-align: center;',
        '  color: var(--text-secondary, #666);',
        '  font-size: 14px;',
        '  line-height: 1.6;',
        '}',
        '.setup-reminder-hint {',
        '  background: var(--bg-secondary, #f5f5f5);',
        '  border-radius: 8px;',
        '  padding: 12px;',
        '  margin-bottom: 20px;',
        '}',
        '.setup-reminder-hint p {',
        '  margin: 0;',
        '  font-size: 13px;',
        '  color: var(--text-secondary, #666);',
        '}',
        '.setup-reminder-buttons {',
        '  display: flex;',
        '  flex-direction: column;',
        '  gap: 12px;',
        '}',
        '.setup-reminder-btn-primary {',
        '  padding: 12px 24px;',
        '  background: var(--primary-color, #4a90d9);',
        '  color: white;',
        '  border: none;',
        '  border-radius: 8px;',
        '  cursor: pointer;',
        '  font-size: 14px;',
        '  font-weight: 500;',
        '  transition: filter 0.2s;',
        '}',
        '.setup-reminder-btn-primary:hover {',
        '  filter: brightness(1.1);',
        '}',
        '.setup-reminder-btn-secondary {',
        '  padding: 10px 24px;',
        '  background: transparent;',
        '  color: var(--text-secondary, #666);',
        '  border: 1px solid var(--border-color, #ddd);',
        '  border-radius: 8px;',
        '  cursor: pointer;',
        '  font-size: 14px;',
        '  transition: background 0.2s;',
        '}',
        '.setup-reminder-btn-secondary:hover {',
        '  background: var(--bg-secondary, #f5f5f5);',
        '}',
        '.setup-reminder-dismiss {',
        '  margin-top: 16px;',
        '  text-align: center;',
        '}',
        '.setup-reminder-dismiss label {',
        '  display: inline-flex;',
        '  align-items: center;',
        '  gap: 6px;',
        '  cursor: pointer;',
        '  font-size: 13px;',
        '  color: var(--text-secondary, #888);',
        '}',
        '.setup-reminder-dismiss input {',
        '  width: 16px;',
        '  height: 16px;',
        '  cursor: pointer;',
        '}',
        '@keyframes settingsBtnPulse {',
        '  0%, 100% { transform: scale(1); }',
        '  50% { transform: scale(1.1); }',
        '}'
    ].join('\n');

    document.head.appendChild(style);
}

/**
 * 验证普通翻译配置
 * @returns {object} { valid: boolean, message: string }
 */
export function validateTranslationConfig() {
    var modelProvider = $('#modelProvider').val();
    var apiKey = ($('#apiKey').val() || '').trim();
    var modelName = ($('#modelName').val() || '').trim();
    var customBaseUrl = ($('#customBaseUrl').val() || '').trim();

    if (!modelProvider) {
        return {
            valid: false,
            message: '请先在顶部 ⚙️ 设置菜单中选择翻译服务商'
        };
    }

    if (PROVIDERS_REQUIRING_API_KEY.indexOf(modelProvider) !== -1) {
        if (!apiKey) {
            return {
                valid: false,
                message: '请先在顶部 ⚙️ 设置菜单中填写 ' + getProviderDisplayName(modelProvider) + ' 的 API Key'
            };
        }
    }

    if (!modelName) {
        if (LOCAL_PROVIDERS.indexOf(modelProvider) !== -1) {
            return {
                valid: false,
                message: '请先在顶部 ⚙️ 设置菜单中填写 ' + getProviderDisplayName(modelProvider) + ' 的模型名称'
            };
        }
        if (PROVIDERS_REQUIRING_API_KEY.indexOf(modelProvider) !== -1) {
            return {
                valid: false,
                message: '请先在顶部 ⚙️ 设置菜单中填写 ' + getProviderDisplayName(modelProvider) + ' 的模型名称'
            };
        }
    }

    if (PROVIDERS_REQUIRING_BASE_URL.indexOf(modelProvider) !== -1) {
        if (!customBaseUrl) {
            return {
                valid: false,
                message: '使用自定义 OpenAI 服务时，请先在顶部 ⚙️ 设置菜单中填写 Base URL'
            };
        }
    }

    return { valid: true, message: '' };
}

/**
 * 验证高质量翻译配置
 * @returns {object} { valid: boolean, message: string }
 */
export function validateHqTranslationConfig() {
    var provider = $('#hqTranslateProvider').val() || '';
    var apiKey = ($('#hqApiKey').val() || '').trim();
    var modelName = ($('#hqModelName').val() || '').trim();
    var customBaseUrl = ($('#hqCustomBaseUrl').val() || '').trim();

    if (!provider) {
        return {
            valid: false,
            message: '请先在顶部 ⚙️ 设置菜单中选择高质量翻译的服务商'
        };
    }

    if (!apiKey) {
        return {
            valid: false,
            message: '请先在顶部 ⚙️ 设置菜单中填写高质量翻译的 API Key'
        };
    }

    if (!modelName) {
        return {
            valid: false,
            message: '请先在顶部 ⚙️ 设置菜单中填写高质量翻译的模型名称'
        };
    }

    if (provider === 'custom_openai' && !customBaseUrl) {
        return {
            valid: false,
            message: '使用自定义服务时，请先在顶部 ⚙️ 设置菜单中填写高质量翻译的 Base URL'
        };
    }

    return { valid: true, message: '' };
}

/**
 * 验证 AI 校对配置
 * @param {Array} proofreadingRounds - 校对轮次配置
 * @returns {object} { valid: boolean, message: string }
 */
export function validateProofreadingConfig(proofreadingRounds) {
    if (!proofreadingRounds || proofreadingRounds.length === 0) {
        return {
            valid: false,
            message: '请先在顶部 ⚙️ 设置菜单中添加至少一个校对轮次'
        };
    }

    for (var i = 0; i < proofreadingRounds.length; i++) {
        var round = proofreadingRounds[i];
        var roundName = round.name || ('轮次' + (i + 1));

        if (!round.provider) {
            return {
                valid: false,
                message: '请先在顶部 ⚙️ 设置菜单中为校对 ' + roundName + ' 选择服务商'
            };
        }

        if (!round.apiKey) {
            return {
                valid: false,
                message: '请先在顶部 ⚙️ 设置菜单中为校对 ' + roundName + ' 填写 API Key'
            };
        }

        if (!round.modelName) {
            return {
                valid: false,
                message: '请先在顶部 ⚙️ 设置菜单中为校对 ' + roundName + ' 填写模型名称'
            };
        }
    }

    return { valid: true, message: '' };
}

/**
 * 翻译前验证配置，验证失败时显示错误消息
 * @param {string} type - 验证类型：'normal' | 'hq' | 'proofread'
 * @param {object} options - 额外选项
 * @returns {boolean} 验证是否通过
 */
export function validateBeforeTranslation(type, options) {
    type = type || 'normal';
    options = options || {};

    var result;

    switch (type) {
        case 'normal':
            result = validateTranslationConfig();
            break;
        case 'hq':
            result = validateHqTranslationConfig();
            break;
        case 'proofread':
            result = validateProofreadingConfig(options.proofreadingRounds);
            break;
        default:
            result = validateTranslationConfig();
    }

    if (!result.valid) {
        ui.showGeneralMessage(result.message, 'error');
        highlightSettingsButton();
        return false;
    }

    return true;
}

/**
 * 高亮设置按钮以引导用户
 */
function highlightSettingsButton() {
    var settingsBtn = document.getElementById('openSettingsBtn');
    if (!settingsBtn) return;

    settingsBtn.style.animation = 'settingsBtnPulse 0.5s ease-in-out 3';
    settingsBtn.style.boxShadow = '0 0 10px var(--primary-color, #4a90d9)';

    setTimeout(function() {
        settingsBtn.style.animation = '';
        settingsBtn.style.boxShadow = '';
    }, 3000);
}

/**
 * 获取服务商显示名称
 */
function getProviderDisplayName(provider) {
    var names = {
        'siliconflow': 'SiliconFlow',
        'deepseek': 'DeepSeek',
        'volcano': '火山引擎',
        'caiyun': '彩云小译',
        'baidu_translate': '百度翻译',
        'youdao_translate': '有道翻译',
        'gemini': 'Google Gemini',
        'custom_openai': '自定义 OpenAI',
        'ollama': 'Ollama',
        'sakura': 'Sakura'
    };
    return names[provider] || provider;
}

/**
 * 重置"不再显示"状态（用于测试或用户主动重置）
 */
export function resetSetupReminderDismiss() {
    localStorage.removeItem(DISMISS_SETUP_REMINDER_KEY);
    console.log('设置提醒状态已重置');
}
