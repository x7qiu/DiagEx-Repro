# Symbol and legend extraction improvement plan

Starting evidence: r-f69f completed in 677.48 seconds, with 480 detections and one failed crop. On the frozen 297-instance review set, it retained 268 instances and classified 262 correctly. The same native candidates cover all 297. The earlier working run retained 278 and classified 256 correctly.

1. Repair the response boundary. Use one discriminant for symbol, uncertain, or a specific rejection reason; retain strict validation and compatibility with saved responses. Normalize known valve class aliases without turning actuator glyphs into valve bodies. Isolate invalid unanchored proposals. Remove obsolete derived checkpoint files when changing stage versions.
2. Improve source legend interpretation and handoff. Keep complete source rows, their identities, and classification evidence. Separate actuator role from contextual valve bodies. Treat geometric family suggestions as retrieval hints for equipment bodies, rank relevant printed definitions ahead of unrelated entries, and preserve image/reference budgets and family diversity.
3. Resolve disagreements with bounded inference. Geometry/classification conflicts should receive the existing targeted follow-up, retaining valid first-pass decisions. Keep two logical calls per crop and the run-level time/failure limits. Improve source-ink instructions for attached components and compressor/vessel distinctions using the applicable legend.
4. Validate on actual saved misses and independent successes. Use offline replay for normalization and malformed-response handling, source-image checks for legend interpretations, and paid production-client canaries for the changed model contract. Include negative/ambiguous candidates as well as successful motor, instrument, valve, vessel, and connector cases. Record actual dispatch counts and failures.
5. Run a complete extraction after canaries pass. Compare the same 297 reviewed instances and per-family results; inspect outstanding errors and full-run status. Target better family correctness than 262/297 while recovering at least the earlier 278/297 retained instances, with no transport runaway or silent malformed-row acceptance. Evaluate legend coverage and sampled source-row semantics separately; symbol scores do not establish legend accuracy.

Model calls are authorized by the user. Keep each experiment bounded, persist diagnostics, and avoid repeating an uncertain dispatch. Do not claim full-document precision from the partial review set.

## Validation outcome

Implemented v2.2 symbol inference and v3.1 legend interpretation/recovery. The complete r-4344 measurement retained 295/297 reviewed instances and classified 292 correctly, exceeding both targets. All 225 crops completed with no transport retries, duplicate native detections, stale checkpoints, or changed native geometry. Runtime increased to 24m 10s from 11m 18s. Remaining review uncertainty keeps the saved quality status partial.

The subsequent legend negative-review guard has separate production-client and cache-recovery validation; it was added while that full symbol run was already in progress. The full symbol measurement has not been relabeled as using the later legend. Detailed evidence, remaining errors, and validation limitations are in `output/symbol-legend-v22/REPORT.md`.
