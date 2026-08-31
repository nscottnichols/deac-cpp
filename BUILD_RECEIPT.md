# DEAC build receipts

`deac.e --build-identity` answers which source state was embedded. The
separate `deac.e --build-receipt` endpoint answers how that executable target
was configured to be built. It emits one newline-terminated, canonical JSON
document. Schema 1 has this top-level shape:

```json
{"schema_version":1,"receipt_sha256":"<64 lowercase hex digits>","receipt":{}}
```

`receipt_sha256` is SHA-256 over the compact canonical JSON bytes of the
nested `receipt` object. Consumers must reject unknown schema versions,
noncanonical encodings, a digest mismatch, and disagreement between
`receipt.source_identity` and `--build-identity`.

## Recorded configuration

The schema-1 payload records:

- the selected backend and a sorted, explicit list of material CMake cache
  entries;
- generator, configuration, and the CMake executable version, resolved path,
  and SHA-256;
- effective File API compile command fragments, definitions, includes,
  frameworks, language standard, precompiled headers, sources, and sysroot for
  the executable target;
- effective final-link fragments, language, LTO state, and sysroot;
- every expected direct build-target dependency, including its compile groups
  and any archive or link information;
- the expanded `CMAKE_AR` and `CMAKE_RANLIB` executable paths, resolved paths,
  and SHA-256 digests; and
- each used compiler driver's configured and resolved path, ID, version,
  target, SHA-256, and implicit include/link inputs.

The generated receipt translation unit is itself present in the target's
recorded compile group. Its path and compile configuration are recorded, but
its contents are not hashed into its own payload, avoiding a recursive digest.
The separately compiled build-identity object is recorded as a direct target
dependency. The final-link fragments and dependency record together cover the
known final target inputs; linked imported libraries remain represented in the
link fragments. The aggregate target fingerprint also binds the exact
compiler-first compile and final-link rule templates and the direct archive
templates accepted at configure time, plus the expanded archive-tool
identities. It is propagated to every compile-capable direct dependency, so a
compiler-byte, archive-tool-byte, or accepted-rule change followed by
reconfiguration invalidates those objects as well as the executable's objects.

The current CPU and SYCL configurations use the same schema. CUDA and HIP
values are representable, but native CMake CUDA/HIP languages currently fail
closed during receipt configuration because their intermediate device-link
rules are not yet represented. DEAC's HIP mode remains a CXX-driver
configuration. This document does not claim either GPU backend has passed real
compiler and device gates.

## Generation and freshness

Build receipts require CMake 3.27 or newer. CMake 3.27 introduced
`cmake_file_api(QUERY)`, which lets the project request codemodel 2.6, cache 2,
and toolchains 1 replies for the current configure/generate invocation. Before
3.27, a query file has to exist before configuration begins; creating it from
the project is too late and would require an externally coordinated query or a
two-pass configure. Raising the minimum is therefore required to make a
single normal configure safe rather than an incidental syntax update.

The requested replies are written at generation time. At build time, DEAC
selects the current File API index, follows that index's reply references, and
checks its source root, build root, configuration, target, backend, expected
direct dependencies, CMake executable, compiler metadata, and configured
archive-tool bytes. Symbolic refresh and rebuild tokens make every ordinary
build regenerate the receipt source, compile that source, and relink the
executable. Makefile graphs attach the always-missing rebuild token directly
to the receipt object and link. Ninja graphs explicitly touch the generated
source; on a coarse-timestamp filesystem they first wait for the clock to
advance beyond a marker written after the previous receipt-object compile.
CMake's normal regeneration step refreshes the File API first when project
configuration changes.

Single-config generators use their configured build type. Multi-config
generators create a distinct generated source, refresh edge, and adjacent
receipt under a `$<CONFIG>` directory. Building one configuration schedules
only that configuration's receipt edge; it does not touch another
configuration's generated source or receipt.

## Tool replacement boundary

At configure time DEAC resolves and hashes each enabled compiler driver and
the expanded `CMAKE_AR` and `CMAKE_RANLIB` executables as well as the CMake
executable, and fingerprints the accepted CMake compile, final-link, and
archive rule templates. An aggregate fingerprint is added as a private compile
definition to every compile-capable recorded target. If compiler, archive
tool, or CMake bytes change, receipt generation fails until CMake is rerun;
after reconfiguration, the changed definition invalidates previously compiled
target and direct-dependency objects even if the tool pathname stayed the
same. The identity dependency is independently rebuilt on every ordinary
build.

Receipt generation checks the current tool bytes before its translation unit
is compiled. A pre-link check repeats the comparison after object compilation
and rejects persistent replacement before a successful final link. Ordinary
CMake cannot prove against a hostile process that swaps a tool between
individual process launches and restores it before verification. The receipt
identifies and hashes the invoked compiler driver and expanded archive tools,
not every transitive program those tools may execute. Compiler and linker
launchers, `RULE_LAUNCH` hooks, code-analysis hooks, link-what-you-use, generic
or configuration-specific interprocedural optimization, and
`CMAKE_<LANG>_COMPILER_ARG1` are rejected because those indirections would
exceed this attribution boundary.

## Supported rule boundary

Schema 1 accepts only the direct, single-command, compiler-first CXX rules and
direct archive templates validated by the receipt module. It fails closed for
legacy static-library rules, multi-command or launcher-prefixed templates, and
native CMake CUDA/HIP languages. Unquoted shell control and compound-command
syntax (including `;`, `&&`, `||`, `|`, backticks, and `$()` substitution) is
rejected; quoted or escaped literal operator characters remain valid. Custom
toolchain files, user/project include hooks, cached module paths, and cached
CXX/CUDA/HIP compile, link, archive, or CUDA device-link rule-template
overrides are rejected because CMake does not provide reliable provenance for
all mutations through those routes. This intentionally excludes
generators/platforms such as MSVC whose normal rules do not satisfy that
boundary; support requires a platform-specific configure and fixture gate
rather than silently weakening attribution.

## Paths and copies

Structured source- and build-tree paths are canonicalized to `<SOURCE_ROOT>`
and `<BUILD_ROOT>`. This makes receipts independent of relocatable working
directories and avoids exposing those private paths. Absolute paths outside
the two roots—including installed compilers, CMake, SDKs, and libraries—are
retained because they are material attribution and may reveal local install
topology. Command fragments are retained verbatim and can also contain
user-supplied paths. Review a receipt before publishing it from a private
system.

CMake writes the same canonical bytes to
`build/deac/receipt/<CONFIG>/deac-build-receipt.json`, and the regression test
requires byte-for-byte equality with `--build-receipt`. This adjacent copy is
only a build-tree diagnostic and can be replaced independently after the
build. Verification and benchmark adapters must query the installed
executable's embedded endpoint as the authority; they may compare, but must
never trust, the adjacent file on its own.
