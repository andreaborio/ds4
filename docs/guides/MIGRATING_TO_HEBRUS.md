# Migrating From DS4 To Hebrus

Status: bridge-release guide. The bridge is implemented in source but is not a
published release yet. Use the version, tag, repository URL, and checksums from
the eventual release announcement rather than guessing them from this guide.

> [!IMPORTANT]
> Hebrus began as a fork of
> [`antirez/ds4`](https://github.com/antirez/ds4) and retains substantial
> implementation, architecture, utilities, and Git history from that project.
> [llama.cpp](https://github.com/ggml-org/llama.cpp),
> [GGML](https://github.com/ggml-org/ggml), and
> [MLX](https://github.com/ml-explore/mlx) are material engineering and
> validation references. The complete provenance boundary is in
> [`ACKNOWLEDGMENTS.md`](../../ACKNOWLEDGMENTS.md) and
> [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

This migration changes the public engine name without changing model bytes,
cache formats, runtime behavior, or existing automation all at once. The safe
path is to install the bridge first, prove both command identities, and only
then move human-facing scripts to the canonical Hebrus names.

## What changes in the bridge

| Surface | Canonical after installing the bridge | Compatibility behavior |
| --- | --- | --- |
| CLI | `hebrus` | `ds4` remains a symlink to the same CLI binary |
| HTTP server | `hebrus-server` | `ds4-server` remains a symlink to the same server binary |
| Agent | `hebrus-agent` | `ds4-agent` remains a symlink to the same agent binary |
| Benchmark | `hebrus-bench` | `ds4-bench` remains a symlink to the same benchmark binary |
| Evaluation | `hebrus-eval` | `ds4-eval` remains a symlink to the same evaluation binary |
| Structured identity | `engine_id: "hebrus"` | Legacy invocation reports `engine_id: "ds4"` |
| Companion app | Hebrus Studio is the selected public name | The bridge preserves its existing bundle ID, data root, and `DSBOX_*` configuration namespace |

The invoked command name may change help headings, prompts, build labels, and
`engine_id`. Options, defaults, exit codes, protocols, generated tokens, model
admission, and resolved runtime plans must remain equivalent for every command
pair. Compatibility commands remain supported through at least the complete
1.x release line.

## What does not change

Do not rename, rewrite, repack, or copy any of the following as part of this
migration:

- the embedded `ds4.expert_major.v2` tensor or its version, manifest, offsets,
  digests, and storage wire values;
- the retired `ds4.expert_major.v1` rejection marker;
- published GGUF filenames, immutable revisions, byte counts, or SHA-256
  values that contain `DS4`;
- disk-KV magic, version, or payload ABI;
- current `DS4_*` engine environment variables;
- current Hebrus Studio `DSBOX_*` variables, `$HOME/.dsbox` data root,
  `$HOME/Library/Application Support/DSBox` Electron `userData` directory, or
  `com.dsbox.desktop` macOS bundle identifier;
- existing checkouts, model files, caches, conversations, downloads, Git
  history, tags, authorship, or historical links.

ExpertMajor is a disk ABI, not a product label. A display may say “Hebrus
ExpertMajor v2”; serialized identifiers must keep their historical bytes.

## 1. Record the working installation

Before changing a checkout or package, stop new inference work and record the
state needed to diagnose or roll back the upgrade:

```sh
git rev-parse HEAD
git status --short
git remote -v

ds4 --build-info
ds4 --capabilities=json
```

Also record the exact model path, byte count, checksum, and any `DS4_*`
variables used by the service. If Hebrus Studio manages the runtime, quit it
cleanly before replacing an application bundle; do not delete `$HOME/.dsbox`.

## 2. Install the announced bridge revision

Use the immutable release tag or commit from the release announcement. Until a
release is announced, the public repository remains
[`andreaborio/ds4`](https://github.com/andreaborio/ds4).

For a user-local source installation:

```sh
git fetch --tags origin
git switch --detach <announced-bridge-tag-or-commit>

make -j
make model-free-test
make install PREFIX="$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
```

Replace the angle-bracket placeholder with the exact announced identity. Do
not paste the placeholder literally. Package maintainers should stage the same
layout without writing to the host:

```sh
make install DESTDIR="$PWD/package-root" PREFIX=/usr/local
```

The staged `hebrus*` files are real executables. Each `ds4*` name is a relative
symlink to the corresponding canonical file, so a relocated package does not
capture a build-machine path.

## 3. Verify both identities before changing automation

Run the model-free identity checks first:

```sh
hebrus --build-info
hebrus --capabilities=json
ds4 --build-info
ds4 --capabilities=json
```

Both capability documents must use schema version 1, the same non-empty build
SHA, backend, executable role, model-family list, and ExpertMajor contract.
Only `engine_id` changes with invocation. If building from source, the complete
repository checks are:

```sh
make capabilities-test
make command-alias-test
make install-test
```

Stop if the build SHA, backend, role, model list, or ExpertMajor fields
contradict each other. Do not fall back from malformed Hebrus capability JSON
to source-string or binary-string probes.

## 4. Move command consumers gradually

New examples, packages, launch agents, and shell scripts should call the
canonical commands. Existing automation can continue using `ds4*` during the
bridge:

```diff
--- before
+++ after
-exec ds4-server -m "$MODEL" --ctx 8192
+exec hebrus-server -m "$MODEL" --ctx 8192
```

Do not rename `DS4_*` variables to speculative `HEBRUS_*` variables. Those
aliases are deferred until one central resolver can detect conflicting old and
new values and fail closed.

Integrations should discover the runtime by executing
`--capabilities=json`. During the bridge they must accept `engine_id` values
`hebrus` and `ds4`, require the expected role and backend, reject unknown schema
versions, and compare the complete ExpertMajor object. Do not infer support
from executable filenames alone.

## 5. Migrate the companion application without moving its data

Hebrus Studio is the selected public name for the companion application. Its
engine bridge should:

1. resolve `hebrus-server` before `ds4-server`;
2. validate schema-1 capability JSON and accept both bridge `engine_id` values;
3. require the server role, expected backend, exact build identity, and exact
   ExpertMajor contract;
4. use source/string fallback only for a legacy capability-less `ds4-server`;
5. reject a capability-less `hebrus-server`, malformed JSON, an unknown schema,
   a contradictory contract, or the wrong build.

The visible name does not authorize a storage migration. The bridge release
continues to use `com.dsbox.desktop`, `$HOME/.dsbox`, and the existing `DSBOX_*`
configuration names. Reusing those identities keeps installed models,
configuration, downloads, and local conversations in place.

### Replace `DSBox.app` with `Hebrus Studio.app`

The macOS upgrade is not an in-place Finder replacement. Because
`DSBox.app` and `Hebrus Studio.app` have different bundle filenames, dragging
the new application into `/Applications` does not replace `DSBox.app`; both
applications remain present until the old bundle is removed.
They nevertheless share the `com.dsbox.desktop` app identifier, legacy state,
and configured local control port (`4242` by default). Never run them at the
same time, including for a side-by-side comparison.

Use this order after obtaining a verified Hebrus Studio package:

1. Quit DSBox from its application menu and wait for its managed engine to
   stop. Confirm neither DSBox nor its server remains active and that the
   configured control port is no longer listening.
2. Confirm the existing state is present at both
   `$HOME/Library/Application Support/DSBox` and `$HOME/.dsbox`. Back up any
   irreplaceable configuration or conversation data according to the normal
   host backup policy; do not rename either directory.
3. Copy `Hebrus Studio.app` into `/Applications`. Finder leaves
   `/Applications/DSBox.app` beside it because the names differ; this is
   expected and is not permission to launch both.
4. Launch only Hebrus Studio. The bridge must keep Electron `userData` at the
   legacy `$HOME/Library/Application Support/DSBox` path and engine/application
   state at `$HOME/.dsbox`; it must not create a parallel fresh profile merely
   because the visible name changed.
5. Verify that existing settings, installed models, downloads, and local
   conversations are visible, then start and stop one model to confirm the
   expected control port and engine capability contract.
6. After verification, remove `/Applications/DSBox.app` or retain one old copy
   only in an offline rollback archive outside `/Applications`. Do not leave a
   routinely launchable old copy alongside Hebrus Studio.

If either state directory changes, an empty profile appears, or the old process
still owns the control port, quit Hebrus Studio and stop the upgrade. Do not
merge two profile directories or launch DSBox to “repair” state while Hebrus
Studio is running.

## 7. Handle the repository rename only after an announcement

No repository rename is implied by this guide. Existing clones continue to use
their current remote. After maintainers announce and verify an administrative
rename, inspect the redirect and then update `origin` to the exact URL in that
announcement:

```sh
git remote get-url origin
git fetch origin
```

Do not create a history-less replacement repository, reuse the old repository
name for another project, rewrite historical links, or change remotes to an
unreserved guessed namespace.

## Rollback

The engine compatibility bridge is designed to make rollback mechanical:

1. stop the running engine or Hebrus Studio process and confirm its port has
   been released;
2. restore the previously recorded executable package or checkout;
3. continue invoking `ds4*` with the unchanged model and data paths;
4. verify the restored `--build-info` and capability document;
5. preserve logs from the failed upgrade before retrying.

Do not roll back by rewriting GGUF metadata, moving `$HOME/.dsbox`, or converting a
published model. If the new package installed into a private staging root, use
`make uninstall` with the same `DESTDIR` and `PREFIX`; it removes only the ten
explicit command paths.

For an application rollback, keep the old app offline until it is needed, then:

1. quit Hebrus Studio and its managed engine completely;
2. remove or archive `/Applications/Hebrus Studio.app` so only one application
   can be launched;
3. restore the archived `DSBox.app` to `/Applications` without moving or
   rewriting either legacy state directory;
4. launch only DSBox and verify its settings, model catalog, conversations,
   control port, and engine build identity;
5. keep Hebrus Studio offline until the rollback issue is resolved.

The bridge release does not require a state-schema migration, so both app
identities use the same legacy Electron `userData` and `$HOME/.dsbox` state.
If a later release explicitly introduces a schema migration, its own release
notes and backup procedure take precedence over this bridge rollback.

## Completion checklist

- [ ] The installed build matches the announced tag or commit.
- [ ] `hebrus*` and `ds4*` command pairs resolve to the same executable files.
- [ ] Canonical and compatibility capability documents differ only where the
      invoked identity or role requires it.
- [ ] The exact qualified model artifact still passes size and SHA-256 checks.
- [ ] Existing `DS4_*`, `DSBOX_*`, `$HOME/.dsbox`, and serialized identifiers are
      unchanged.
- [ ] Electron `userData` remains at
      `$HOME/Library/Application Support/DSBox`; no second empty profile was
      created for the visible name.
- [ ] DSBox and Hebrus Studio were never active simultaneously, and only one
      application owns the configured control port.
- [ ] Hebrus Studio can admit the canonical server and fall back to the legacy
      server without weakening validation.
- [ ] Rollback has been tested without copying or deleting user data.
- [ ] Remote URLs were changed only after an authoritative rename announcement.

The normative compatibility rules are in
[`BRAND_COMPATIBILITY.md`](../contracts/BRAND_COMPATIBILITY.md); the supported
models and hardware boundary are in
[`RUNTIME_SUPPORT.md`](../contracts/RUNTIME_SUPPORT.md).
