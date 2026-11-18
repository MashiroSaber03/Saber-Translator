const assert = require('node:assert/strict');
const test = require('node:test');
const prepare = require('./prepare_release.cjs');
const { mkdtempSync, readFileSync, rmSync } = require('node:fs');
const { resolve, join } = require('node:path');
const { spawnSync } = require('node:child_process');

process.env.RELEASE_TAG = 'v9.0.0';

const root = resolve(__dirname, '../..');
const version = JSON.parse(readFileSync(join(root, 'version.json'), 'utf8')).version;
const workflow = readFileSync(join(root, '.github/workflows/release.yml'), 'utf8').replace(/\r\n/g, '\n');
const versionScript = workflow.match(/- name: Get version\n[\s\S]*?        run: \|\n([\s\S]*?)(?=\n      - name:)/)[1]
  .split('\n').map(line => line.slice(10)).join('\n');

function runVersion(env, verify) {
  const temporary = mkdtempSync(join(root, '.tmp-release-version-'));
  try {
    const output = join(temporary, 'output');
    const result = spawnSync(process.platform === 'win32' ? 'C:/Program Files/Git/bin/bash.exe' : 'bash',
      ['-eu', '-c', versionScript], { cwd: root, encoding: 'utf8', env: {
        ...process.env, EVENT_NAME: 'workflow_dispatch', INPUT_VERSION: '', INPUT_PUBLISH: '',
        REF_NAME: 'dev', REF_TYPE: 'branch', ...env, GITHUB_OUTPUT: output.replace(/\\/g, '/'),
      } });
    verify(result, () => Object.fromEntries(readFileSync(output, 'utf8').trim().split('\n').map(line => line.split('='))));
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
}

for (const [name, env, publish] of [
  ['dev push', { EVENT_NAME: 'push' }, false],
  ['pull request', { EVENT_NAME: 'pull_request', REF_NAME: '195/merge' }, false],
  ['manual branch validation', {}, false],
  ['manual tag validation', { REF_TYPE: 'tag', REF_NAME: `v${version}` }, false],
  ['manual publication', { INPUT_PUBLISH: 'true' }, true],
  ['manual tag publication', { INPUT_PUBLISH: 'true', REF_TYPE: 'tag', REF_NAME: `v${version}` }, true],
  ['tag push', { EVENT_NAME: 'push', REF_TYPE: 'tag', REF_NAME: `v${version}` }, true],
]) {
  test(`${name} uses the intended publishing mode`, () => runVersion(env, (result, outputs) => {
    assert.equal(result.status, 0, result.stderr || result.error?.message);
    assert.deepEqual(outputs(), { VERSION: version, TAG_NAME: `v${version}`, PUBLISH: String(publish) });
  }));
}

for (const input of ['v1.2.3', `${Number(version.split('.')[0]) + 1}.0.0`]) {
  test(`invalid or mismatched manual version ${input} is rejected`, () => runVersion(
    { INPUT_VERSION: input, INPUT_PUBLISH: 'true' }, result => {
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /Invalid release version|Release version differs/);
    }));
}

function fixture({ release, target, refError } = {}) {
  const created = [];
  const outputs = {};
  const github = {
    paginate: async () => release ? [release] : [],
    rest: {
      repos: { listReleases() {} },
      git: {
        getRef: async () => {
          if (refError) throw { status: refError };
          if (!target) throw { status: 404 };
          return { data: { object: target } };
        },
        getTag: async () => ({ data: { object: { type: 'commit', sha: 'current' } } }),
        createRef: async value => created.push(value),
      },
    },
  };
  return { created, outputs, run: () => prepare({ github,
    context: { repo: { owner: 'test', repo: 'test' }, sha: 'current' },
    core: { setOutput: (key, value) => { outputs[key] = value; } },
  }) };
}
test('new version creates a tag at the built commit', async () => {
  const f = fixture(); await f.run();
  assert.equal(f.created[0].sha, 'current');
  assert.equal(f.created[0].ref, 'refs/tags/v9.0.0');
  assert.equal(f.outputs.create_release, 'true');
});
test('lightweight and annotated tags both resolve to the built commit', async () => {
  for (const target of [{ type: 'commit', sha: 'current' }, { type: 'tag', sha: 'annotation' }]) {
    const f = fixture({ target }); await f.run(); assert.equal(f.created.length, 0);
  }
});
test('wrong commit is rejected before modifying remote state', async () => {
  const f = fixture({ target: { type: 'commit', sha: 'other' } });
  await assert.rejects(f.run(), /points to/); assert.equal(f.created.length, 0);
});
test('published releases are never modified', async () => {
  const f = fixture({ release: { tag_name: 'v9.0.0', draft: false } });
  await assert.rejects(f.run(), /already published/); assert.equal(f.created.length, 0);
});
test('retry reuses the draft without overwriting release notes', async () => {
  const f = fixture({ release: { tag_name: 'v9.0.0', draft: true }, target: { type: 'commit', sha: 'current' } });
  await f.run(); assert.equal(f.outputs.create_release, 'false'); assert.equal(f.created.length, 0);
});
test('untagged draft cannot be retargeted to a different commit', async () => {
  const f = fixture({ release: { tag_name: 'v9.0.0', draft: true, target_commitish: 'other' } });
  await assert.rejects(f.run(), /different commit/); assert.equal(f.created.length, 0);
});
test('permission failures are not mistaken for missing tags', async () => {
  const f = fixture({ refError: 403 });
  await assert.rejects(f.run(), error => error.status === 403); assert.equal(f.created.length, 0);
});
