# AI-powered camera-trap wildlife monitoring pipeline

A two-stage computer-vision + LLM pipeline for the manual review bottleneck in camera-trap conservation monitoring: **MegaDetector** filters blank frames and localises animals, then **Claude (Sonnet 5)** identifies the species from the cropped detection, the full frame, and the sibling frames in the same camera burst.

Built and evaluated on a 200-image random sample from the [Wellington Camera Traps dataset](https://lila.science/datasets/wellingtoncameratraps) (LILA BC).

---

## The problem

Camera traps fire on any movement, so a large share of what comes back is wind in the grass. Sorting those frames by hand is one of the bigger time costs in ecological fieldwork. This project tests whether an off-the-shelf detector combined with a general-purpose vision-language model can cut that work down, and how far the claim survives being measured properly.

## Pipeline

1. **Stage 1 — detection and filtering (MegaDetector).** Every image goes through Microsoft's MegaDetector, an open-source model used across conservation programmes worldwide. It draws a bounding box and confidence score around any animal, person or vehicle. Frames with no detection above 0.2 confidence are filtered out as blank.
2. **Stage 2 — species identification (Claude).** For every image MegaDetector flags as containing an animal, the pipeline sends Claude three pieces of visual evidence: the full frame with the detection boxed, for scale and habitat; an upscaled crop of the detected region; and the other frames from the same 3-shot burst. Claude has to describe what it can actually see before committing to a species, picks from the dataset's own category list, and is explicitly allowed to answer `unclassifiable` instead of guessing.

## Results

### Stage 1 — blank-frame filtering

| Metric | Result |
|---|---|
| Overall accuracy | **94.5%** (189/200) |
| True animal frames correctly kept | 172/181 (95.0% recall) |
| True blank frames correctly filtered | 17/19 |
| Animal frames wrongly discarded | 9 |

### Stage 2 — species identification (progression)

| Configuration | N | Correct | Abstained | Wrong | Raw accuracy | Accuracy when committed |
|---|---|---|---|---|---|---|
| Haiku, naive prompt (with location hint) | 174 | 19 | 25 | 130 | 10.9% | 12.8% |
| Haiku, location hint removed | 174 | 17 | 42 | 115 | 9.8% | 12.9% |
| Sonnet, crop only | 80 (dev) | 26 | 40 | 14 | 32.5% | 65.0% |
| Sonnet, + full-frame context | 80 (dev) | 32 | 37 | 11 | 40.0% | 74.4% |
| Sonnet, + burst sibling frames | 80 (dev) | 38 | 33 | 9 | 47.5% | 80.9% |
| **Sonnet + burst frames — full run** | **174** | **95** | **62** | **17** | **54.6%** | **84.8%** |

On true birds, the dataset's dominant class at 119 of 174 images: 72 identified correctly, 40 abstentions, and 7 confident misclassifications. That is 91.1% accuracy on the cases where the model committed.

### Timing benchmark

I timed a 30-image spot-check with a stopwatch, comparing unassisted review (including species lookup on a reference sheet) against reviewing the pipeline's output.

| | Time |
|---|---|
| Manual review (all burst frames shown, cheat sheet available) | 3:27 |
| Reviewing pipeline output | 1:27 |
| **Time saved** | **58.0%** |

## Diagnosing and fixing a geographic reasoning bias

The first working version of the species-ID prompt told Claude the images came from "Wellington, New Zealand". Accuracy came back at about 9%, barely above chance. The raw model output showed why: Claude was reasoning from geography rather than pixels, citing brushtail possums as common in Wellington to justify its answer on images that were plainly birds. Frequency analysis confirmed it. Claude predicted "possum" 123 times across 174 images; the true count was 3.

Removing the location hint and forcing an explicit `OBSERVATION` step — describe what is visually present before naming a species — almost eliminated the possum bias, from 123 guesses down to 34. But the predictions didn't shift toward genuine uncertainty. They shifted to a new default: hedgehog, from 1 guess to 61, against a true count of 10. Removing one shortcut doesn't cure the underlying cause. On a single small low-light frame there often isn't enough visual signal for reliable fine-grained ID, and a model under pressure to answer will find some plausible-sounding default.

The fix combined three changes, tested independently and cumulatively against a held-out dev set: a stronger model (Sonnet over Haiku), scene-scale context (the full frame with the detection boxed, not just an isolated crop), and the largest single contributor, giving Claude the other frames from the same 3-shot burst. That mirrors how the dataset's original human labellers worked, and it tracked a limitation I had noted on day one: the cameras fire in triplicate, and the labelled animal isn't visible in every individual frame.

## Problems found by running it by hand

Manually running and reviewing the pipeline's output surfaced things the numbers alone wouldn't have shown.

- While doing the manual half of the timing benchmark, I noticed the review pages showed only one frame per image, while the pipeline itself was being given all three burst frames. That made the manual pass a harder task than the one the AI was doing, and would have produced a misleading time-saved figure. I rebuilt both benchmark pages to show the full burst and redid the timing run from scratch. The 58% above is from the corrected version.
- The first version of the species cheat sheet picked its example image for each category arbitrarily, whichever appeared first in the dataset. Several examples were unrecognisable, and one didn't show the labelled animal at all. I changed the selection logic to pick the example with the highest MegaDetector detection confidence per species, as a proxy for image clarity, before using the sheet for the actual benchmark.
- Several images in the 30-image benchmark sample looked, on manual inspection, like clear misses by the pipeline. Pulling the model's raw output for each showed that 10 of the 11 disagreements were `unclassifiable` abstentions rather than confident wrong guesses. That distinction is why the results here report accuracy-when-committed alongside raw accuracy.

## Limitations

- Evaluated on a 200-image random sample, not the full 270k-image dataset. Category-level statistics for rare classes (true possum count = 3) should be read with that in mind.
- The 9 real animal frames discarded as blank are a genuine cost of the confidence threshold chosen. A conservation deployment would need to tune it against the relative cost of missed detections versus reviewer time.
- Single-frame image quality (low resolution, IR grain, motion blur) is the main limit on species ID, not prompt engineering. The high abstention rate is appropriate caution given that, rather than something to be prompted away.

## Stack

Python, [MegaDetector](https://github.com/agentmorris/MegaDetector) (PyTorch), Claude API (Sonnet 5 vision), pandas, PIL. Run in Google Colab.

## Repo contents

- `camera_trap_report.html` — full results dashboard with a mixed sample gallery
- `species_cheat_sheet.html` — reference sheet used for the manual timing benchmark
- Notebook — full pipeline, all experiments (A/B/C), and raw results for every run referenced above
