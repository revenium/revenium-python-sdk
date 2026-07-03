# Vendored: revenium_metering

This directory is a verbatim copy of the `revenium-metering` Python package,
absorbed into the SDK per BACK-2151 to make `revenium-python-sdk` self-contained.

- Source package: `revenium-metering`
- Version: 6.8.2
- License: MIT (Revenium — same org)
- Vendored: 2026-06-29

This is generated (Stainless) code. Do not hand-edit the generated logic.
The only local modifications are (1) import-path rewrites from the absolute
`revenium_metering.*` namespace to this in-package location
(`revenium_middleware._metering`), and (2) aligning `_version.py`'s
`__version__` to the vendored distribution version (6.8.2 — upstream's
in-package string lagged at 6.8.1). See git history for the exact sites.

Public symbols re-exported by the SDK from here: `meter_tool`,
`report_tool_call`, `configure` (via `revenium_middleware/__init__.py`).
Internal-only: `ReveniumMetering`.
