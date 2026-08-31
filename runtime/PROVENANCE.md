# Provenance

The runtime was reconstructed from the user-supplied `code/tabm4pv.py` on
2026-08-29. Refactoring separated configuration, data loading, feature
construction, training, evaluation, and inference. The fixed scalar TabM
semantics are preserved, while data loading now pools multiple station files
into one shared model as requested.

Intentional protocol corrections are documented rather than hidden:

- the final test is not evaluated during epochs;
- protected evaluation is invoked separately;
- curve mode repeats the same scalar model across 16 target indices;
- station identity is metadata, not a model feature;
- no province output, cross-station feature join, plant aggregation, or
  capacity-weighted weather logic is present.
