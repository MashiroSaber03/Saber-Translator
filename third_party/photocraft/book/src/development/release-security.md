# Release security

PhotoCraft's release workflow builds platform packages and a draft GitHub Release from the `release` branch. The authoritative process is [`docs/releasing.md`](https://github.com/storytold/photocraft/blob/main/docs/releasing.md).

## Implemented workflow elements

- platform-specific signing/notarization steps exist for macOS and Windows;
- signing secrets are scoped through the GitHub `release` environment;
- release artifacts include `SHA256SUMS.txt`;
- the final release job receives `contents: write`, while build jobs use narrower access;
- artifacts remain in a draft release until a maintainer publishes them.

Signing is conditional. The workflow can produce unsigned artifacts when signing material is unavailable and emits warnings. Consumers and maintainers must verify the job summary and artifact signature rather than assuming every release is signed.

## Future hardening

- generate a machine-readable SBOM for each release;
- publish provenance/attestations tied to the source commit and workflow;
- make signature state explicit in release metadata;
- verify checksums, signatures, notarization, and installer behavior before publication;
- review action pinning and release-tool provenance;
- document key rotation and compromise response.

Building, signing, drafting, publishing, and deploying are separate states. A successful build must not be reported as a published release.
