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
JSON never contains a raw C0 control character. CMake truncates decoded strings
at a semantic NUL, so schema 1 rejects both a raw NUL byte and a JSON `\u0000`
escape before parsing. Backspace, tab, newline, form feed, and carriage return
use their short JSON escapes; every other representable value from U+0001
through U+001F uses a lowercase `\u00xx` escape.

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
dependency. CMake's File API omits interface-only targets from its target and
dependency lists, so DEAC emits each declared direct interface contract
explicitly, with its name and `INTERFACE_LIBRARY` type but no invented compile,
archive, or link record. A deferred seal at the end of top-level generation
rechecks the final direct interface graph and the supported rule, launcher,
analysis, IPO, and link-hook boundary. A mutation after receipt registration
therefore fails the configure/generate step instead of escaping the earlier
snapshot. A required imported provider is resolved without a target-file
generator expression and must then appear as exactly one resolved
`libraries`-role fragment in the build-time File API reply. Required provider
artifacts are unique in every selected configuration, not merely as their
unevaluated multi-config expressions. These checks avoid adding a new
consumer-to-provider build-graph edge while still failing closed if the final
contract or provider link changes.

The final-link fragments and dependency record together cover the known final
target inputs. For HIP with BLAS enabled, the material cache entries include
`CMAKE_DISABLE_FIND_PACKAGE_hipblas`, `HIP_RUNTIME_INCLUDE_DIR`,
`DEAC_HIPBLAS_INCLUDE_DIR`, `DEAC_HIPBLAS_LIBRARY`, `hipblas_DIR`, and
`hipblas_ROOT`, including absent values, so both supported package discovery
and compatibility-fallback routes remain attributable. The aggregate target
fingerprint binds the exact compiler-first compile and final-link rule
templates and the direct archive templates in the receipt registration
directory, plus the expanded compiler and archive-tool identities. Every
buildable target's CMake source-directory scope must expose identical supported
rule templates and resolve to the same configured compiler, `CMAKE_AR`, and
`CMAKE_RANLIB` path, real path, and digest. The fingerprint is propagated to
every compile-capable direct dependency, so a compiler-byte,
archive-tool-byte, or accepted-rule change followed by reconfiguration
invalidates those objects as well as the executable objects.
Receipt generation requires every compile group of the executable and every
compile-capable recorded dependency to contain exactly one matching reserved
fingerprint definition and no duplicate or conflicting definition with that
identifier. Compile command fragments may not define or undefine that reserved
identifier outside the structured definitions array. Removing or shadowing the
injected definition is therefore a hard error rather than an apparently
attributable stale object.

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

`deac_target_add_build_receipt()` exports `DEAC_BUILD_RECEIPT_REFRESH` to its
caller as the primary refresh output. A receipt-only convenience target may
depend on that path without compiling or linking the consumer. Callers must not
use the generated receipt source BYPRODUCT as the dependency: with Unix
Makefiles, its rule remains owned by the consuming target's `build.make` and is
not a portable cross-target refresh edge.

The native-fragment decoder and literal-shell gate are currently supported and
fixture-tested on POSIX hosts with Ninja, Ninja Multi-Config, and Unix
Makefiles. Other host/generator shell dialects are not covered by schema 1 and
must not be enabled by weakening the POSIX checks.

Single-config generators require exactly one nonempty configured build type.
Multi-config generators create a distinct generated source, refresh edge, and
adjacent receipt under a `$<CONFIG>` directory. Building one configuration
schedules only that configuration's receipt edge; it does not touch another
configuration's generated source or receipt. Configuration names must be
path-safe and unique ignoring ASCII case because CMake's `$<CONFIG:...>`
comparison is case-insensitive. The exact generator name, single- versus
multi-config mode, and ordered configuration list must agree in the top-level
source and receipt-registration directories and are sealed again at the end of
generation, so a late directory mutation cannot invalidate the File API/output
mapping or leave stale cache evidence. The effective generator and active build
type/configuration-list variables must also equal their cache entries when
present, preventing a normal variable from shadowing contradictory material
cache evidence. The adjacent receipt path accepts only one literal `$<CONFIG>`
as a complete path component and rejects every other generator expression,
preventing a hidden or conditional token from collapsing configurations onto
one output.

## Tool replacement boundary

At configure time DEAC resolves and hashes each enabled compiler driver and
the expanded `CMAKE_AR` and `CMAKE_RANLIB` executables as well as the CMake
executable, and fingerprints the accepted CMake compile, final-link, and
archive rule templates. The configured compiler-target triple's presence and
value must exactly match the File API toolchain reply before it is published in
the receipt. An aggregate fingerprint is added as a private compile definition
to every compile-capable recorded target. If compiler, archive tool, or CMake
bytes change, receipt generation fails until CMake is rerun; after
reconfiguration, the changed definition invalidates previously compiled target
and direct-dependency objects even if the tool pathname stayed the same. The
identity dependency is independently rebuilt on every ordinary build.

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
native CMake CUDA/HIP languages. Outside POSIX quotes or protection that
survives CMake's generator, globbing, comments, redirection, command or process
substitution, compound-command operators, and grouping syntax are rejected.
Dollar signs are rejected in every quoting state because Make and Ninja can
expand them before POSIX shell quoting applies. An unquoted semicolon is also
rejected even when the input text contains a backslash, because CMake consumes
that protection while materializing a rule. Backticks are rejected unless
protected by POSIX single quotes; a backslash-escaped backtick is outside this
schema. Only the explicitly allowed standalone CMake rule placeholders may
occur, and generator expressions in recipe text are unsupported. The same
literal-shell policy is applied to effective File API compile, archive, and
link fragments, including material `CMAKE_CXX_FLAGS` and linker flags.

Custom toolchain files, user/project include hooks, cached module paths, and
cached CXX/CUDA/HIP compile, link, archive, or CUDA device-link rule-template
overrides are rejected because CMake does not provide reliable provenance for
all mutations through those routes. The end-of-generation seal repeats these
checks and rejects late generator/configuration-state changes,
target/directory/global launcher hooks, IPO, `LINK_WHAT_YOU_USE`, and
direct-interface mutations. This intentionally excludes generators/platforms
such as MSVC whose normal rules do not satisfy that boundary; support requires
a platform-specific configure and fixture gate rather than silently weakening
attribution.

## Paths and copies

Structured source- and build-tree paths are canonicalized to `<SOURCE_ROOT>`
and `<BUILD_ROOT>`. When roots overlap, the most-specific containing root wins;
a build nested inside the source tree therefore still uses `<BUILD_ROOT>` for
its generated files. This makes receipts independent of relocatable working
directories and avoids exposing those private paths. Absolute paths outside
the two roots—including installed compilers, CMake, SDKs, and libraries—are
retained because they are material attribution and may reveal local install
topology. Command fragments are retained verbatim after the literal-shell
check and can also contain user-supplied paths. Review a receipt before
publishing it from a private system.

Generated sources, refresh markers, compile markers, and adjacent receipts must
be normalized descendants of `CMAKE_BINARY_DIR`. A symlinked build root itself
is supported, but every descendant component is checked for existing or
dangling symlinks at configuration, deferred sealing, receipt generation, and
pre-link validation so an output cannot be redirected outside the build tree.

CMake writes the same canonical bytes to
`build/deac/receipt/<CONFIG>/deac-build-receipt.json`, and the regression test
requires byte-for-byte equality with `--build-receipt`. This adjacent copy is
only a build-tree diagnostic and can be replaced independently after the
build. Verification and benchmark adapters must query the installed
executable's embedded endpoint as the authority; they may compare, but must
never trust, the adjacent file on its own.
