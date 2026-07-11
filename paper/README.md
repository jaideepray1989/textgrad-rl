# RulePI REALM 2026 Short Paper

Anonymous LaTeX submission draft for the REALM workshop at EMNLP 2026.

## Official Format

Checked on 2026-07-10:

- REALM archival short papers allow up to 4 content pages, with unlimited references and appendix: <https://realm-workshop.github.io/call_for_papers/>
- ACL review formatting requires the official style, A4 paper, two columns, line numbers, an abstract of at most 200 words, and a `Limitations` section after the conclusion: <https://acl-org.github.io/ACLPUB/formatting.html>
- `acl.sty` and `acl_natbib.bst` are from the official ACL style repository at commit `d5adc823ff0f80f98c80405ca0ab66c68e684409`: <https://github.com/acl-org/acl-style-files>

The compiled review PDF is 4 A4 pages. The conclusion ends on page 4; the required `Limitations` section follows it. All fonts are embedded.

## Files

- `realm_emnlp_textgrad_rl_draft.tex`: anonymous review source
- `references.bib`: verified bibliography
- `acl.sty`, `acl_natbib.bst`: official ACL style files
- `../output/pdf/rulepi_realm_2026_anonymous.pdf`: compiled review PDF

## Build

```bash
cd paper
tectonic realm_emnlp_textgrad_rl_draft.tex \
  --outdir build --keep-logs --keep-intermediates
```

## Result Provenance

The paper's primary numbers are recorded in the checked-in evidence summary:

- `../RULEPI_PAPER_EVIDENCE.md`

That summary was generated from the following local run artifacts. The `runs/` directory is intentionally ignored because complete benchmark runs can be large:

- `../runs/rulepi_textworld_10seed/summary.md`
- `../runs/rulepi_textworld_10seed/all_test_records.csv`
- `../runs/rulepi_textarena_supported_10seed/summary.md`
- `../runs/qwen25_7b_textworld24_full80_t5/textworld_24/summary.md`

The main claims are deliberately narrow: persistent rules improve the controlled structured actors and amortize task-local diagnostic retry. The draft does not claim a validation-gate advantage or generic LLM self-improvement.
