# F1 2023 pilot baseline diagnostic

- Source run: `runs/20260817T134240.218846Z-e5205a24`
- Executed v3 run: `runs/20260818T093758.404026Z-dad0f43b`
- Diagnostic date: 2026-08-18
- Scope: six persisted v2 predictions followed by one predeclared six-case local v3 run
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

## Predeclared minimal test

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

Because the prompt affects experimental comparability, this test required and received explicit
user approval before implementation and model execution.

## Executed v3 result

The approved test ran once from clean Git commit `b4cc289` and completed all six planned cases in
51.7 seconds. Run metadata records `git_dirty: false`, prompt version
`verification-v3-baseline-knowledge`, and system-prompt hash
`32ef0e89423b4e3e49c2361d79800111830ee4ec6fde7f5ea61b6c73e2c230fb`. Offline re-evaluation
reproduced the persisted metrics.

- Six of six predictions were valid on the first response, with no repair or technical error.
- All citation lists were correctly empty.
- All six predictions remained `NOT_ENOUGH_EVIDENCE`.
- Accuracy remained 0.3333 and Macro-F1 remained 0.1667.
- Every reason again used only the absence of evidence, for example: `No evidence is provided to
  support or refute the claim.`
- No response attempted a model-knowledge assessment, including the winner and podium claims.

The clarification therefore did not produce Option A. The first diagnostic correctly identified
the observable shortcut, but a one-line explicit prohibition was insufficient to change this 4B
model's behavior. This still does not prove that the underlying model lacks the facts; it shows
that this controlled baseline interface did not elicit them under either prompt version.

## Applied stop rule

Do not create another prompt variant or start a model sweep. The v3 wording is the clearer baseline
contract, but adopting it for the controlled pilot would change the shared prompt hash and requires
an explicit decision. The recommended path is to freeze v3, accept the all-NEE 4B baseline as an
observed limitation, rerun the six-claim three-condition gate under the shared v3 prompt, and then
proceed to the 54-case pilot if that gate remains valid.
