# F1 2023 pilot baseline diagnostic

- Source run: `runs/20260817T134240.218846Z-e5205a24`
- Diagnostic date: 2026-08-18
- Scope: six persisted baseline predictions; no new model, retrieval, or provider calls
- Model: `ragcv-qwen3-4b-pilot:v1`
- Prompt version: `verification-v2-citations`

## Question

The corrected gate produced `NOT_ENOUGH_EVIDENCE` for all six baseline cases. This diagnostic asks
whether the observed collapse is already explained by the baseline instruction or demonstrates
missing parametric knowledge in the 4B model.

## Observations

All six cases completed successfully on the first response. There were no technical errors, parse
errors, repair calls, evidence objects, or citations. The six raw reasons followed one pattern:

| Gold label | Cases | Observed reason pattern |
| --- | ---: | --- |
| `SUPPORTED` | 2 | No external evidence is available to verify the claim. |
| `REFUTED` | 2 | No external evidence is available to verify the claim or detail. |
| `NOT_ENOUGH_EVIDENCE` | 2 | No external evidence is available to verify the detail. |

The system prompt says that baseline mode has no external evidence, should evaluate from model
knowledge, and should choose NEE when uncertain. The rendered user prompt simultaneously presents
`Evidence: NO_EXTERNAL_EVIDENCE`. The model treated that expected condition as sufficient for NEE
in every case, including direct winner and podium claims. None of the reasons referred to uncertain
memory, a competing remembered fact, or an attempted factual assessment.

## Interpretation

The strongest explanation supported by this run is a prompt-conditioned absence-of-evidence
shortcut: the model followed the visible `NO_EXTERNAL_EVIDENCE` marker instead of exercising the
separate model-knowledge instruction. This run therefore does **not** distinguish adequate from
inadequate parametric Formula One knowledge. It would be too strong to conclude that the 4B model
lacks the facts, because its reasons show no attempt to use them.

This is a diagnostic inference from six cases, not a general model-capability result. The two NEE
gold cases also cannot establish appropriate abstention while the same rationale is used for every
label category.

## Minimal proposed test

Change exactly one experimental variable: create prompt version
`verification-v3-baseline-knowledge` with an explicit baseline clarification:

> In baseline mode, the absence of retrieved evidence is expected and is not by itself a reason
> for NOT_ENOUGH_EVIDENCE. Decide from model knowledge; use NOT_ENOUGH_EVIDENCE only when that
> knowledge is insufficient or uncertain.

Keep the model, six claims, label contract, temperature, seed, JSON mode, retrieval settings,
corpora, and gold annotations unchanged. Run the six baseline cases once before any three-condition
gate.

The purpose is not to force non-NEE predictions or tune against gold labels. It is to remove the
identified shortcut and observe whether the model then attempts a factual decision. Required
technical properties remain six completed predictions, first-pass or explicitly repaired valid
JSON, and empty baseline citation lists.

## Stop rule and next decision

- If the revised reasons demonstrate model-knowledge assessment, freeze the prompt and rerun the
  six-claim three-condition gate so all conditions share the new prompt hash.
- If all cases remain NEE for expressed factual uncertainty, accept the 4B baseline as weak and do
  not continue prompt tuning or start a model sweep.
- Only after a valid gate should the 18-claim, three-condition pilot be run.

Changing the prompt affects experimental comparability and requires explicit user approval before
implementation or model execution.
