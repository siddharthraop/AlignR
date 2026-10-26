## AlignR draft: `tpu7x-specifications` (reference)

**2 open question(s)** · grammar valid · 9/9 values sourced · 1 model call(s) · $0.0062

### Built from
- Siblings: `tpu-v6e-specifications.dita`, `tpu-v5p-specifications.dita`
- Facts: `facts/tpu7x.yaml` · schema `8b76` · model `gemini-3.8-flash`

### Checks
- Grammar: DITA 1.3 reference (RELAX NG) valid
- Rules: 0 errors, 0 warnings
- Grounding: 9/9 values

### Questions
- For the writer: The sibling topics give HBM capacity in different units (v6e uses GB, v5p uses GiB). TPU7x facts specify GiB. Should we standardize the unit across all generations?
- For the engineer: Source S1 lists HBM bandwidth as 7380 GBps, whereas source S2 lists it approximately as 7.37 TB/s (which corresponds to 7370 GBps rather than 7380 GBps). Which value is correct?

### Compared with the published topic
Edit ratio 19% against `topics/tpu7x-specifications.dita` (adapted from https://docs.cloud.google.com/tpu/docs/tpu7x).

Run `20261026-041740-4593` · 4,725 tokens in · 697 out. AlignR can't merge. Close every draft comment before merging: the open-questions check blocks the merge.
