import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const {
  getMock,
  postMock,
  putMock,
  deleteMock,
  uploadMock,
} = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  putMock: vi.fn(),
  deleteMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    put: putMock,
    delete: deleteMock,
    upload: uploadMock,
  },
}))

vi.mock('@/api/v2/content', () => ({
  newIdempotencyKey: () => 'test-idempotency-key',
}))

const pluginV2 = {
  pluginId: 'sample_plugin',
  displayName: 'Sample Plugin',
  author: 'Tests',
  description: 'v3 plugin',
  state: 'enabled',
  defaultEnabled: false,
  runtimeEnabled: true,
  config: { enabled: true },
  configRevision: 4,
  errorMessage: null,
  pluginVersionId: 'version-1',
  packageVersion: '1.0.0',
  currentRevision: 7,
  manifest: {
    schema_version: 3,
    plugin_id: 'sample_plugin',
    display_name: 'Sample Plugin',
    package_version: '1.0.0',
    entrypoint: 'plugin.py:Plugin',
    hooks: ['before_ocr'],
    supported_steps: ['ocr'],
    supported_modes: ['standard'],
    priority: 0,
    failure_policy: 'continue',
    author: 'Tests',
    description: 'v3 plugin',
    default_enabled: false,
    config_schema: {
      enabled: { type: 'boolean', default: true },
    },
  },
  configSchema: {
    enabled: { type: 'boolean', default: true },
  },
}

describe('plugin v3 api', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    putMock.mockReset()
    deleteMock.mockReset()
    uploadMock.mockReset()
    vi.restoreAllMocks()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('downloads the immutable current package from the v2 route', async () => {
    const blob = new Blob(['zip-bytes'], { type: 'application/zip' })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: {
        get: vi.fn().mockImplementation((name: string) => (
          name.toLowerCase() === 'content-disposition'
            ? 'attachment; filename="sample_plugin.zip"'
            : null
        )),
      },
      blob: vi.fn().mockResolvedValue(blob),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { exportPlugin } = await import('@/api/plugin')
    const result = await exportPlugin('sample_plugin')

    expect(fetchMock).toHaveBeenCalledWith('/api/v2/plugins/sample_plugin/export')
    expect(result.filename).toBe('sample_plugin.zip')
    expect(result.blob).toBe(blob)
  })

  it('lets the backend identify and import a v3 package', async () => {
    const file = new File(['backend-owned-zip'], 'sample_plugin.zip', {
      type: 'application/zip',
    })
    uploadMock.mockResolvedValue({
      pluginId: 'sample_plugin',
      pluginVersionId: 'version-1',
      packageVersion: '1.0.0',
      currentRevision: 7,
    })

    const { importPlugin } = await import('@/api/plugin')
    await importPlugin(file)

    expect(uploadMock).toHaveBeenCalledTimes(1)
    const [url, formData, config] = uploadMock.mock.calls[0] || []
    expect(url).toBe('/api/v2/plugins/import')
    expect(formData).toBeInstanceOf(FormData)
    expect((formData as FormData).get('file')).toBe(file)
    expect((formData as FormData).get('baseRevision')).toBe('0')
    expect(config).toEqual({
      headers: { 'Idempotency-Key': 'test-idempotency-key' },
    })
    expect(getMock).not.toHaveBeenCalled()
  })

  it('retries an existing package with backend-provided CAS metadata', async () => {
    const file = new File(['backend-owned-zip'], 'renamed-package.zip', {
      type: 'application/zip',
    })
    uploadMock
      .mockRejectedValueOnce({
        status: 409,
        details: {
          pluginId: 'sample_plugin',
          currentRevision: 7,
        },
      })
      .mockResolvedValueOnce({
        pluginId: 'sample_plugin',
        pluginVersionId: 'version-2',
        packageVersion: '2.0.0',
        currentRevision: 8,
      })
    const { importPlugin } = await import('@/api/plugin')
    await expect(importPlugin(file)).rejects.toMatchObject({ status: 409 })
    await importPlugin(file, true)

    const replacementForm = uploadMock.mock.calls[1]?.[1] as FormData
    expect(replacementForm.get('baseRevision')).toBe('7')
    expect(getMock).not.toHaveBeenCalled()
  })

  it('does not coerce malformed import conflict metadata', async () => {
    const file = new File(['backend-owned-zip'], 'malformed-conflict.zip', {
      type: 'application/zip',
    })
    uploadMock.mockRejectedValueOnce({
      status: 409,
      details: {
        pluginId: 'sample_plugin',
        currentRevision: '7',
      },
    })
    const { importPlugin } = await import('@/api/plugin')

    await expect(importPlugin(file)).rejects.toMatchObject({ status: 409 })
    await expect(importPlugin(file, true)).rejects.toThrow(
      '插件替换上下文已失效',
    )
    expect(uploadMock).toHaveBeenCalledTimes(1)
  })

  it('uses v2 management routes and cached CAS revisions', async () => {
    getMock
      .mockResolvedValueOnce({ items: [pluginV2] })
      .mockResolvedValueOnce({
        pluginId: 'sample_plugin',
        schema: pluginV2.configSchema,
        value: pluginV2.config,
        configRevision: 4,
      })
    putMock
      .mockResolvedValueOnce({ ...pluginV2, runtimeEnabled: true })
      .mockResolvedValueOnce({
        pluginId: 'sample_plugin',
        schema: pluginV2.configSchema,
        value: { enabled: false },
        configRevision: 5,
      })
      .mockResolvedValueOnce({ ...pluginV2, defaultEnabled: true })
    deleteMock.mockResolvedValue({ deleted: true })

    const {
      deletePlugin,
      enablePlugin,
      getPluginConfigDocument,
      getPlugins,
      savePluginConfig,
      setPluginDefaultState,
    } = await import('@/api/plugin')

    await getPlugins()
    await enablePlugin('sample_plugin')
    await expect(getPluginConfigDocument('sample_plugin')).resolves.toEqual({
      schema: pluginV2.configSchema,
      value: pluginV2.config,
    })
    await savePluginConfig('sample_plugin', { enabled: false })
    await setPluginDefaultState('sample_plugin', true)
    await deletePlugin('sample_plugin')

    expect(putMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/plugins/sample_plugin/runtime-enabled',
      { enabled: true },
    )
    expect(putMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/plugins/sample_plugin/config',
      { baseRevision: 4, config: { enabled: false } },
      { headers: { 'Idempotency-Key': 'test-idempotency-key' } },
    )
    expect(putMock).toHaveBeenNthCalledWith(
      3,
      '/api/v2/plugins/sample_plugin/default-enabled',
      { enabled: true },
      { headers: { 'Idempotency-Key': 'test-idempotency-key' } },
    )
    expect(deleteMock).toHaveBeenCalledWith(
      '/api/v2/plugins/sample_plugin',
      {
        headers: {
          'If-Match': '7',
          'Idempotency-Key': 'test-idempotency-key',
        },
      },
    )
  })

  it('drops CAS revisions for plugins absent from an authoritative list', async () => {
    getMock
      .mockResolvedValueOnce({ items: [pluginV2] })
      .mockResolvedValueOnce({ items: [] })

    const { deletePlugin, getPlugins } = await import('@/api/plugin')
    await getPlugins()
    await getPlugins()

    await expect(deletePlugin('sample_plugin')).rejects.toThrow(
      '插件版本已变化，请刷新后重试',
    )
    expect(deleteMock).not.toHaveBeenCalled()
  })
})
