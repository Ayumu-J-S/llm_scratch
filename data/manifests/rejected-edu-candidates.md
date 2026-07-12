# Rejected DATA-004 educational candidates

DATA-004 rejected the initially considered educational derivatives before any
corpus shard was downloaded:

- `HuggingFaceFW/fineweb-edu` at
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` documents that its quality
  classifier was trained from Llama 3 70B annotations. The Meta Llama 3 license
  restricts using model outputs or results to improve another large language
  model, leaving an avoidable downstream provenance question.
- `hotchpotch/fineweb-2-edu-japanese` at
  `180ca004c6a89b590daaad86cb062a07a5353c69` documents DeepSeek API teacher
  annotations. Current DeepSeek API terms permit training other models from
  outputs, but those terms took effect after this dataset was created and do
  not establish the historical terms governing the annotations.

The selected direct FineWeb and FineWeb2 sources use documented language
identification, heuristic filtering, and MinHash deduplication instead of a
generative-model quality-annotation stage. This decision avoids the additional
model-output licensing uncertainty; it does not remove the ordinary ODC-By and
Common Crawl content-rights caveats recorded in `README.md`.

Primary evidence:

- FineWeb-Edu pinned card: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md>
- Japanese Edu pinned card: <https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/blob/180ca004c6a89b590daaad86cb062a07a5353c69/README.md>
- Meta Llama 3 license: <https://github.com/meta-llama/llama3/blob/dab0ae890fd0316369b89901b199355034bbe1b5/LICENSE>
- Current DeepSeek API terms: <https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html>
