// Called by actions/github-script before creating or reusing a draft release.
module.exports = async ({ github, context, core }) => {
  const tag = process.env.RELEASE_TAG;
  const repo = context.repo;
  const releases = await github.paginate(github.rest.repos.listReleases, repo);
  const release = releases.find(item => item.tag_name === tag);
  if (release && !release.draft) {
    throw new Error(`${tag} is already published; use a new version.`);
  }

  let target;
  try {
    target = (await github.rest.git.getRef({ ...repo, ref: `tags/${tag}` })).data.object;
  } catch (error) {
    if (error.status !== 404) throw error;
  }
  while (target?.type === 'tag') {
    target = (await github.rest.git.getTag({ ...repo, tag_sha: target.sha })).data.object;
  }
  if (target && target.sha !== context.sha) {
    throw new Error(`${tag} points to ${target.sha}, but this run builds ${context.sha}.`);
  }
  if (!target) {
    if (release && release.target_commitish !== context.sha) {
      throw new Error(`${tag} draft belongs to a different commit; use a new version.`);
    }
    // GITHUB_TOKEN-created tags do not start a second workflow run.
    await github.rest.git.createRef({ ...repo, ref: `refs/tags/${tag}`, sha: context.sha });
  }
  core.setOutput('create_release', release ? 'false' : 'true');
};
