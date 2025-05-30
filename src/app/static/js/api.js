// src/app/static/js/api.js

// 引入 jQuery (或者使用 fetch API)
// import $ from 'jquery'; // 如果使用 npm 安装了 jQuery
// 或者确保 jQuery 已经全局加载
import * as state from './state.js'; // 导入 state 以获取当前 rpm 设置
import * as constants from './constants.js'; // 确保导入前端常量

/**
 * 封装 AJAX 请求
 * @param {string} url - API 端点 URL
 * @param {string} method - HTTP 方法 (GET, POST, etc.)
 * @param {object} [data=null] - 发送的数据 (对于 POST 请求)
 * @param {string} [dataType='json'] - 预期的数据类型
 * @param {boolean} [contentTypeJson=true] - 是否设置 Content-Type 为 application/json
 * @returns {Promise<object>} - 返回一个 Promise 对象，包含响应数据或错误信息
 */
function makeApiRequest(url, method, data = null, dataType = 'json', contentTypeJson = true) {
    const options = {
        url: url, // URL 已经包含了 /api 前缀 (由蓝图定义)
        type: method,
        dataType: dataType,
    };

    if (data) {
        options.data = data instanceof FormData ? data : JSON.stringify(data);
        if (data instanceof FormData) {
            options.processData = false; // FormData 不需要 jQuery 处理
            options.contentType = false; // FormData 不需要设置 Content-Type
        } else if (contentTypeJson) {
            options.contentType = 'application/json';
        }
    }

    console.log(`发起 API 请求: ${method} ${url}`, data ? '携带数据' : '');

    return new Promise((resolve, reject) => {
        $.ajax(options)
            .done((response) => {
                console.log(`API 响应 (${url}):`, response);
                resolve(response);
            })
            .fail((jqXHR, textStatus, errorThrown) => {
                console.error(`API 请求失败 (${url}):`, textStatus, errorThrown, jqXHR.responseText);
                let errorMsg = `请求失败: ${textStatus}`;
                if (jqXHR.responseJSON && jqXHR.responseJSON.error) {
                    errorMsg = jqXHR.responseJSON.error;
                } else if (jqXHR.responseText) {
                    // 尝试提取文本错误信息
                    try {
                         const errData = JSON.parse(jqXHR.responseText);
                         if(errData.error) errorMsg = errData.error;
                    } catch(e) {
                         // 如果不是 JSON，截取部分文本
                         errorMsg = jqXHR.responseText.substring(0, 100);
                    }
                }
                reject({ message: errorMsg, status: jqXHR.status, errorThrown: errorThrown });
            });
    });
}

// --- 翻译与渲染 API ---

/**
 * 请求翻译或消除文字
 * @param {object} params - 包含所有翻译参数的对象
 * @returns {Promise<object>} - 包含翻译结果的 Promise
 */
export function translateImageApi(params) {
    // 从 state 中获取当前的 rpm 设置并添加到参数中
    const apiParams = {
        ...params,
        rpm_limit_translation: state.rpmLimitTranslation,
        rpm_limit_ai_vision_ocr: state.rpmLimitAiVisionOcr
    };

    // VVVVVV 新增逻辑：为自定义AI视觉OCR添加Base URL VVVVVV
    if (params.ocr_engine === 'ai_vision' && params.ai_vision_provider === constants.CUSTOM_AI_VISION_PROVIDER_ID_FRONTEND) {
        if (state.customAiVisionBaseUrl && state.customAiVisionBaseUrl.trim() !== '') {
            apiParams.custom_ai_vision_base_url = state.customAiVisionBaseUrl;
        } else {
            // 如果 Base URL 为空，翻译请求可能会失败，或者后端会处理此情况。
            // 前端最好在发起请求前就进行校验 (已在 events.js 中添加)。
            // 此处可以加一个警告，或者如果后端不处理空URL，则应在此处阻止请求。
            console.warn("translateImageApi: 尝试使用自定义AI视觉OCR但未提供Base URL。");
        }
    }
    // ^^^^^^ 结束新增逻辑 ^^^^^^

    console.log("translateImageApi: 发送的参数（含rpm和可能的自定义视觉Base URL）:", apiParams);
    return makeApiRequest('/api/translate_image', 'POST', apiParams);
}

/**
 * 请求重新渲染整个图像
 * @param {object} params - 包含渲染参数的对象 (image, clean_image, bubble_texts, bubble_coords, ...)
 * @returns {Promise<object>} - 包含渲染结果的 Promise
 */
export function reRenderImageApi(params) {
    return makeApiRequest('/api/re_render_image', 'POST', params);
}

/**
 * 请求重新渲染单个气泡
 * @param {object} params - 包含单气泡渲染参数的对象 (bubble_index, all_texts, ...)
 * @returns {Promise<object>} - 包含渲染结果的 Promise
 */
export function reRenderSingleBubbleApi(params) {
    return makeApiRequest('/api/re_render_single_bubble', 'POST', params);
}

/**
 * 请求将设置应用到所有图片
 * @param {object} params - 包含设置和图片数据的对象
 * @returns {Promise<object>} - 包含结果的 Promise
 */
export function applySettingsToAllApi(params) {
    return makeApiRequest('/api/apply_settings_to_all_images', 'POST', params);
}

/**
 * 请求翻译单段文本
 * @param {object} params - 包含 original_text, target_language, 等参数的对象
 * @returns {Promise<object>} - 包含翻译结果的 Promise
 */
export function translateSingleTextApi(params) {
    // 从 state 中获取当前的 rpm 设置并添加到参数中
    const apiParams = {
        ...params,
        rpm_limit_translation: state.rpmLimitTranslation
        // AI Vision OCR rpm 不适用于单文本翻译
    };
    console.log("translateSingleTextApi: 发送的参数（含rpm）:", apiParams);
    return makeApiRequest('/api/translate_single_text', 'POST', apiParams);
}


// --- 配置管理 API ---

/**
 * 获取模型使用历史
 * @returns {Promise<object>}
 */
export function getModelInfoApi() {
    return makeApiRequest('/api/get_model_info', 'GET');
}

/**
 * 获取指定服务商的使用过的模型
 * @param {string} provider - 服务商名称
 * @returns {Promise<object>}
 */
export function getUsedModelsApi(provider) {
    return makeApiRequest(`/api/get_used_models?model_provider=${provider}`, 'GET');
}

/**
 * 保存模型信息
 * @param {string} provider - 服务商名称
 * @param {string} modelName - 模型名称
 * @returns {Promise<object>}
 */
export function saveModelInfoApi(provider, modelName) {
    return makeApiRequest('/api/save_model_info', 'POST', { modelProvider: provider, modelName: modelName });
}

/**
 * 获取漫画翻译提示词信息
 * @returns {Promise<object>}
 */
export function getPromptsApi() {
    return makeApiRequest('/api/get_prompts', 'GET');
}

/**
 * 保存漫画翻译提示词
 * @param {string} name - 提示词名称
 * @param {string} content - 提示词内容
 * @returns {Promise<object>}
 */
export function savePromptApi(name, content) {
    return makeApiRequest('/api/save_prompt', 'POST', { prompt_name: name, prompt_content: content });
}

/**
 * 获取指定名称的漫画翻译提示词内容
 * @param {string} name - 提示词名称
 * @returns {Promise<object>}
 */
export function getPromptContentApi(name) {
    return makeApiRequest(`/api/get_prompt_content?prompt_name=${name}`, 'GET');
}

/**
 * 重置漫画翻译提示词为默认
 * @returns {Promise<object>}
 */
export function resetPromptApi() {
    return makeApiRequest('/api/reset_prompt_to_default', 'POST');
}

/**
 * 删除指定名称的漫画翻译提示词
 * @param {string} name - 提示词名称
 * @returns {Promise<object>}
 */
export function deletePromptApi(name) {
    return makeApiRequest('/api/delete_prompt', 'POST', { prompt_name: name });
}

/**
 * 获取文本框提示词信息
 * @returns {Promise<object>}
 */
export function getTextboxPromptsApi() {
    return makeApiRequest('/api/get_textbox_prompts', 'GET');
}

/**
 * 保存文本框提示词
 * @param {string} name - 提示词名称
 * @param {string} content - 提示词内容
 * @returns {Promise<object>}
 */
export function saveTextboxPromptApi(name, content) {
    return makeApiRequest('/api/save_textbox_prompt', 'POST', { prompt_name: name, prompt_content: content });
}

/**
 * 获取指定名称的文本框提示词内容
 * @param {string} name - 提示词名称
 * @returns {Promise<object>}
 */
export function getTextboxPromptContentApi(name) {
    return makeApiRequest(`/api/get_textbox_prompt_content?prompt_name=${name}`, 'GET');
}

/**
 * 重置文本框提示词为默认
 * @returns {Promise<object>}
 */
export function resetTextboxPromptApi() {
    return makeApiRequest('/api/reset_textbox_prompt_to_default', 'POST');
}

/**
 * 删除指定名称的文本框提示词
 * @param {string} name - 提示词名称
 * @returns {Promise<object>}
 */
export function deleteTextboxPromptApi(name) {
    return makeApiRequest('/api/delete_textbox_prompt', 'POST', { prompt_name: name });
}


// --- 系统管理 API ---

/**
 * 上传 PDF 文件进行处理
 * @param {FormData} formData - 包含 PDF 文件的 FormData 对象
 * @returns {Promise<object>} - 包含提取图像数据的 Promise
 */
export function uploadPdfApi(formData) {
    // FormData 请求不需要设置 contentType 为 JSON
    return makeApiRequest('/api/upload_pdf', 'POST', formData, 'json', false);
}

/**
 * 请求清理调试文件
 * @returns {Promise<object>}
 */
export function cleanDebugFilesApi() {
    return makeApiRequest('/api/clean_debug_files', 'POST');
}

/**
 * 测试 Ollama 连接
 * @returns {Promise<object>}
 */
export function testOllamaConnectionApi() {
    return makeApiRequest('/api/test_ollama_connection', 'GET');
}

/**
 * 测试 Sakura 连接
 * @param {boolean} [forceRefresh=false] - 是否强制刷新模型列表
 * @returns {Promise<object>}
 */
export function testSakuraConnectionApi(forceRefresh = false) {
    const url = forceRefresh ? '/api/test_sakura_connection?force=true' : '/api/test_sakura_connection';
    return makeApiRequest(url, 'GET');
}

/**
 * 测试 LAMA 修复 API 端点 (GET 请求，无参数)
 * @returns {Promise<object>}
 */
export function testLamaRepairApi() {
    return makeApiRequest('/api/test_lama_repair', 'GET');
}

/**
 * 测试参数解析 API 端点
 * @param {object} params - 要测试的参数对象
 * @returns {Promise<object>}
 */
export function testParamsApi(params) {
    return makeApiRequest('/api/test_params', 'POST', params);
}

/**
 * 测试百度OCR连接
 * @param {string} apiKey - 百度OCR API Key
 * @param {string} secretKey - 百度OCR Secret Key
 * @returns {Promise<object>}
 */
export function testBaiduOcrConnectionApi(apiKey, secretKey) {
    return makeApiRequest('/api/test_baidu_ocr_connection', 'POST', {
        api_key: apiKey,
        secret_key: secretKey
    });
}

// --- 插件管理 API ---

/**
 * 获取插件列表
 * @returns {Promise<object>}
 */
export function getPluginsApi() {
    return makeApiRequest('/api/plugins', 'GET');
}

/**
 * 启用指定插件
 * @param {string} pluginName - 插件名称
 * @returns {Promise<object>}
 */
export function enablePluginApi(pluginName) {
    return makeApiRequest(`/api/plugins/${pluginName}/enable`, 'POST');
}

/**
 * 禁用指定插件
 * @param {string} pluginName - 插件名称
 * @returns {Promise<object>}
 */
export function disablePluginApi(pluginName) {
    return makeApiRequest(`/api/plugins/${pluginName}/disable`, 'POST');
}

/**
 * 删除指定插件
 * @param {string} pluginName - 插件名称
 * @returns {Promise<object>}
 */
export function deletePluginApi(pluginName) {
    return makeApiRequest(`/api/plugins/${pluginName}`, 'DELETE');
}

/**
 * 获取插件配置规范
 * @param {string} pluginName - 插件名称
 * @returns {Promise<object>}
 */
export function getPluginConfigSchemaApi(pluginName) {
    return makeApiRequest(`/api/plugins/${pluginName}/config_schema`, 'GET');
}

/**
 * 获取插件当前配置
 * @param {string} pluginName - 插件名称
 * @returns {Promise<object>}
 */
export function getPluginConfigApi(pluginName) {
    return makeApiRequest(`/api/plugins/${pluginName}/config`, 'GET');
}

/**
 * 保存插件配置
 * @param {string} pluginName - 插件名称
 * @param {object} configData - 配置数据
 * @returns {Promise<object>}
 */
export function savePluginConfigApi(pluginName, configData) {
    return makeApiRequest(`/api/plugins/${pluginName}/config`, 'POST', configData);
}

/**
 * 请求后端仅检测图片中的气泡框
 * @param {string} imageData - Base64 编码的图像数据 (不含 'data:image/...' 前缀)
 * @param {number} [confThreshold=0.6] - YOLOv5 置信度阈值
 * @returns {Promise<object>} - 包含检测结果坐标的 Promise ({success: boolean, bubble_coords: Array})
 */
export function detectBoxesApi(imageData, confThreshold = 0.6) {
    const params = {
        image: imageData,
        conf_threshold: confThreshold
    };
    // 如果使用单独的 API 端点
    return makeApiRequest('/api/detect_boxes', 'POST', params);

    // --- 或者，如果选择修改现有 API ---
    // const translateParams = {
    //     image: imageData,
    //     detect_only: true, // 添加特殊标记
    //     conf_threshold: confThreshold
    //     // 可能需要传递一些虚拟的其他参数以满足后端验证
    // };
    // return makeApiRequest('/api/translate_image', 'POST', translateParams);
    // ---------------------------------
}

/**
 * 获取所有插件的默认启用状态
 * @returns {Promise<object>} - 包含状态字典的 Promise ({success: boolean, states: object})
 */
export function getPluginDefaultStatesApi() {
    return makeApiRequest('/api/plugins/default_states', 'GET');
}

/**
 * 设置指定插件的默认启用状态
 * @param {string} pluginName - 插件名称
 * @param {boolean} enabled - true 表示默认启用, false 表示默认禁用
 * @returns {Promise<object>} - 包含结果的 Promise
 */
export function setPluginDefaultStateApi(pluginName, enabled) {
    const safePluginName = encodeURIComponent(pluginName); // 对插件名进行编码以防特殊字符
    return makeApiRequest(`/api/plugins/${safePluginName}/set_default_state`, 'POST', { enabled: enabled });
}

// --- 会话管理 API ---

/**
 * 请求后端保存当前会话状态。
 * @param {string} sessionName - 要保存的会话名称。
 * @param {object} sessionData - 包含当前状态的完整对象 (ui_settings, images, currentImageIndex)。
 * @returns {Promise<object>} - 包含后端响应的 Promise ({success: boolean, message?: string, error?: string})。
 */
export function saveSessionApi(sessionName, sessionData) {
    const payload = {
        session_name: sessionName,
        session_data: sessionData
    };
    // 调用通用的请求函数，使用 POST 方法，并发送 JSON 数据
    return makeApiRequest('/api/sessions/save', 'POST', payload);
}

// --- 会话管理 API (续) ---

/**
 * 请求后端获取所有已保存会话的列表。
 * @returns {Promise<object>} - 包含后端响应的 Promise ({success: boolean, sessions?: Array<object>, error?: string})。
 *                            sessions 数组包含对象如 {name: string, saved_at: string, image_count: number, version: string}。
 */
export function listSessionsApi() {
    // 调用通用的请求函数，使用 GET 方法
    return makeApiRequest('/api/sessions/list', 'GET');
}

/**
 * 请求后端加载指定名称的会话数据。
 * @param {string} sessionName - 要加载的会话名称。
 * @returns {Promise<object>} - 包含后端响应的 Promise ({success: boolean, session_data?: object, error?: string})。
 *                            session_data 包含完整的会话状态。
 */
export function loadSessionApi(sessionName) {
    // 将会话名称作为 URL 查询参数传递
    // 使用 encodeURIComponent 确保名称中的特殊字符被正确编码
    const encodedName = encodeURIComponent(sessionName);
    return makeApiRequest(`/api/sessions/load?name=${encodedName}`, 'GET');
}

/**
 * 请求后端删除指定的会话。
 * @param {string} sessionName - 要删除的会话名称。
 * @returns {Promise<object>} - 包含后端响应的 Promise ({success: boolean, message?: string, error?: string})。
 */
export function deleteSessionApi(sessionName) {
    const payload = {
        session_name: sessionName
    };
    return makeApiRequest('/api/sessions/delete', 'POST', payload);
}

/**
 * 请求后端重命名指定的会话。
 * @param {string} oldName - 当前的会话名称。
 * @param {string} newName - 新的会话名称。
 * @returns {Promise<object>} - 包含后端响应的 Promise ({success: boolean, message?: string, error?: string})。
 */
export function renameSessionApi(oldName, newName) {
    const payload = {
        old_name: oldName,
        new_name: newName
    };
    return makeApiRequest('/api/sessions/rename', 'POST', payload);
}

/**
 * 测试AI视觉OCR连接
 * @param {string} provider - 服务提供商(如'siliconflow')
 * @param {string} apiKey - API密钥
 * @param {string} modelName - 模型名称
 * @param {string} prompt - OCR提示词(可选)
 * @returns {Promise<object>}
 */
export function testAiVisionOcrApi(provider, apiKey, modelName, prompt = null) {
    const payload = {
        provider: provider,
        api_key: apiKey,
        model_name: modelName,
        prompt: prompt
    };

    // VVVVVV 新增逻辑：为自定义AI视觉OCR添加Base URL VVVVVV
    if (provider === constants.CUSTOM_AI_VISION_PROVIDER_ID_FRONTEND) {
        if (state.customAiVisionBaseUrl && state.customAiVisionBaseUrl.trim() !== '') {
            payload.custom_ai_vision_base_url = state.customAiVisionBaseUrl;
        } else {
            // 如果 Base URL 为空，测试请求应该失败。
            // 在 events.js 中已经做了检查，这里可以作为最后防线。
            console.error("testAiVisionOcrApi: 自定义AI视觉服务需要填写Base URL！");
            return Promise.reject({ message: "自定义AI视觉服务需要填写Base URL！" });
        }
    }
    // ^^^^^^ 结束新增逻辑 ^^^^^^

    return makeApiRequest('/api/test_ai_vision_ocr', 'POST', payload);
}

/**
 * 获取字体列表
 * @param {Function} successCallback - 成功时的回调函数
 * @param {Function} errorCallback - 失败时的回调函数
 */
export function getFontListApi(successCallback, errorCallback) {
    $.ajax({
        url: '/api/get_font_list',
        type: 'GET',
        success: function(response) {
            if (successCallback) successCallback(response);
        },
        error: function(xhr, status, error) {
            console.error('获取字体列表失败:', error);
            if (errorCallback) errorCallback(error);
        }
    });
}

/**
 * 上传字体文件
 * @param {File} fontFile - 字体文件
 * @param {Function} successCallback - 成功时的回调函数
 * @param {Function} errorCallback - 失败时的回调函数
 */
export function uploadFontApi(fontFile, successCallback, errorCallback) {
    const formData = new FormData();
    formData.append('font', fontFile);
    
    $.ajax({
        url: '/api/upload_font',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (successCallback) successCallback(response);
        },
        error: function(xhr, status, error) {
            console.error('上传字体失败:', error);
            if (errorCallback) errorCallback(xhr.responseJSON?.error || error);
        }
    });
}

// --- TODO: 在后续步骤中添加重命名的 API 调用函数 ---
// export function renameSessionApi(oldName, newName) { ... }
