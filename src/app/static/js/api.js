// src/app/static/js/api.js

// 引入 jQuery (或者使用 fetch API)
// import $ from 'jquery'; // 如果使用 npm 安装了 jQuery
// 或者确保 jQuery 已经全局加载

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
 * @param {object} params - 包含所有翻译参数的对象 (image, target_language, ..., skip_translation, remove_only)
 * @returns {Promise<object>} - 包含翻译结果的 Promise
 */
export function translateImageApi(params) {
    return makeApiRequest('/api/translate_image', 'POST', params);
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
    return makeApiRequest('/api/translate_single_text', 'POST', params);
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
