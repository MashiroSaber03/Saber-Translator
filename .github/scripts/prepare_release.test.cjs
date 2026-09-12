const assert = require('node:assert/strict');
const test = require('node:test');
const prepare = require('./prepare_release.cjs');

process.env.RELEASE_TAG = 'v9.0.0';
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
