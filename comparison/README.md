# Ubuntu remote snapshot

- Captured: `2026-07-29T11:11:57.230555+09:00`
- Remote repository: `/home/example/Desktop/loess` (not a Git worktree)
- Repository files: `351`
- Installed plugin files: `38`
- Plugin comparison: `38 identical`, `0 different`, `0 repository-only`, `0 installed-only`
- Selected output inventory: `13 code candidates`, `138 supporting text files`
- All four checksum dry-runs completed with no differences.
- Repository and installed QGIS plugin: `38/38` files are checksum-identical.
- Python syntax: `158/158` passed; Shell syntax: `21/21` passed.
- Test discovery: `159` tests collected.
- macOS snapshot run: `146 passed, 12 failed, 1 skipped`; failures are retained in
  `pytest_snapshot_run.txt` and are not Ubuntu QGIS3/Qt5/CUDA acceptance.
- The streaming scale-acceptance and assembly queue-fix stage files are already
  checksum-identical to their production files; see `known_output_promotions.tsv`.
- No macOS source file was overwritten or merged.
