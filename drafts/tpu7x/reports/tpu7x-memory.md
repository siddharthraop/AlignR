## AlignR draft: `tpu7x-memory` (concept)

**4 open question(s)** · grammar valid · 0/0 values sourced · 1 model call(s) · $0.0035

### Built from
- Siblings: none (new in this generation: stub)
- Facts: `facts/tpu7x.yaml` · schema `1b5a` · model `gemini-3.8-flash`

### Checks
- Grammar: DITA 1.3 concept (RELAX NG) valid
- Rules: 0 errors, 0 warnings
- Grounding: 0/0 values

### Questions
- For the writer: This topic is new in tpu7x and has no sibling to draft from. AlignR wrote an outline with the facts each section should cover. Write the prose, then close this comment.
- For the engineer: S1 lists the HBM bandwidth as {tpu7x-hbm-bandwidth} GBps (which equals 7.38 TB/s), while S2 reports it as approximately 7.37 TB/s. Which value is correct?
- For the writer: write this section from F8 (S2): Each chip has two chiplets. Each chiplet has one TensorCore, two SparseCores and GiB of HBM.
- For the writer: write this section from F9 (S2): Host memory connects to the chip with much lower bandwidth than HBM; it can hold offloaded activations or optimizer states.

### Compared with the published topic
Edit ratio 87% against `topics/tpu7x-memory.dita` (adapted from https://docs.cloud.google.com/tpu/docs/tpu7x).

Run `20261026-041740-4593` · 3,615 tokens in · 202 out. AlignR can't merge. Close every draft comment before merging: the open-questions check blocks the merge.
