# Documentation Consolidation Mapping

Planned target structure:

```
docs/
├── guide.md (consolidated best practices)
├── implementation/
│   ├── streaming.md (STREAMING_IMPLEMENTATION.md + parts of MIGRATION_COMPLETE.md)
│   ├── migration.md (MIGRATION_COMPLETE.md + IMPLEMENTATION_SUMMARY.md)
│   ├── configuration.md (configuration best practices)
│   └── testing.md (testing strategies)
├── reference/
│   ├── api.md (API reference)
│   ├── error-handling.md (error taxonomy and patterns)
│   └── troubleshooting.md
└── archive/
    ├── TORCH_TEST_FIX_COMPLETED.md
    ├── LOGURU_IMPLEMENTATION_SUMMARY.md
    └── FIXES_COMPLETED.md
```

High-overlap pairs to merge (examples):
- STREAMING_IMPLEMENTATION.md ↔ MIGRATION_COMPLETE.md
- IMPLEMENTATION_SUMMARY.md ↔ FIXES_COMPLETED.md

Notes:
- Keep `guide.md` as the single source for best practices; reference from others.
- Avoid duplication by linking to canonical sections.



