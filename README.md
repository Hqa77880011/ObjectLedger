# ObjectLedger

Implementation of **ObjectLedger: Event-Driven and Reappearance-Verified Object Permanence in Long Egocentric Video**.

The ledger maintains identity, existence, discrete visibility/container state and bounded semantic reachable-location hypotheses separately. Reliable observations can update identity; hidden predictions cannot. Inspected-empty locations provide negative existence evidence, while an unseen region supplies none. Verified reappearances train association and preceding transition paths without inserting future features into the forward state.

## Installation

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11 or 3.12 and activate the environment in your shell.

## Prepare a fixed observation front end

Use object/interaction observations from EPIC-KITCHENS/VISOR or Ego4D, keeping the same detector, appearance encoder, category predictions and event cues across methods. This implementation starts at that fixed front end. Export one JSON object per video into `data/videos.jsonl`; `video` is unique and `split` is `train`, `val` or `test`. Split complete recording/participant environments before fitting.

Each video contains time-sorted `frames`:

| Field | Meaning |
|---|---|
| `time` | Timestamp in seconds |
| `observations` | List of current visible object detections |
| `cues` | Fixed-size causal hand/object/container/context vector |
| `inspected_empty` | Location indices explicitly viewed and found empty; empty when unseen |
| `truth` | Optional evaluation/training labels by ground-truth object ID; never model inputs |

An observation has normalized `feature` (D values), category probability vector `category`, semantic `location` index, optional `reliable` flag, and annotation-only `identity`/`verified_reappearance`. `location` may encode a place/container relation such as `kitchen|drawer_A`. The last cue indicates that the object's predicted region remains within view, enabling an unmatched visible entry to become occluded. Other cues must be measured from the current/past frames rather than annotated future events.

For known objects, `truth` can contain `exists`, `visible`, `state`, `location`, `event`. State is `visible`, `occluded`, `contained`, `out-of-view` or `removed`; event is `pick-up`, `put-down`, `enter-container`, `exit-container`, `leave-view`, `reappear`, `remove` or `none`. Identity annotations supervise losses/evaluation only. Mark a reappearance verified only when the same physical instance is established. Training preparation propagates a verified return's positive existence label backward over its hidden interval unless an annotated removal intervenes. Test labels are never propagated into inference.

Copy `ledger.example.json` and edit feature/cue dimensions, semantic location vocabulary, gap scale and admissible event transitions. Self transitions and an aggregate unknown-reachable state are added automatically. An explicit full `allowed_transitions` tensor can instead specify all 8×(P+1)×(P+1) entries, with at least one allowed destination in every row.

## Train and track

```bash
objectledger data/videos.jsonl ledger.example.json --output outputs/full
```

`--epochs 40`, `--seed`, `--slots 5`, `--match-threshold 0.7`, `--identity-threshold 0.8` and `--event-threshold 0.7` control training, hypothesis budget and evidence gates. Register thresholds using validation videos before test comparison. An event becomes hard only above its event threshold; otherwise its posterior updates continuous existence/location quantities while discrete state stays unchanged. Existence updates use event evidence, matched observations and inspected-empty location mass, without a time-to-live term.

Reappearance assignment is a joint Hungarian solution including unmatched/new-object alternatives. Scores combine appearance, category, marginal semantic reachability, event compatibility and clipped gap. Low-confidence matches stay unresolved. The stored identity prototype updates only from reliable visible observations above the identity gate. Unknown tail location mass occupies one hypothesis slot, preventing truncation from silently discarding uncertainty.

## Comparisons

```bash
objectledger data/videos.jsonl ledger.example.json --variant continuous --output outputs/continuous
objectledger data/videos.jsonl ledger.example.json --variant ttl --output outputs/ttl
objectledger data/videos.jsonl ledger.example.json --variant last_location --output outputs/last_location
objectledger data/videos.jsonl ledger.example.json --variant no_verified_learning --output outputs/no_verified_learning
```

Continuous-ledger writes a learned propagated appearance feature while hidden. The matched auxiliary predictor is trained on observed feature continuations in all variants. TTL adds elapsed-time existence decay. Last-location collapses relational alternatives. No-verified-learning removes delayed identity/path supervision. All methods retain the same front end, cue access and observation timestamps.

## Output and metrics

`model.pt` stores model parameters, registered location graph and arguments. `history.csv` records real optimization losses. `ledger.jsonl` logs existence, state and location distributions; `associations.csv` logs accepted identities. `existence.png` summarizes per-object beliefs.

`metrics.json` includes reappearance identity accuracy, fixed-front-end IDF1, false merges/splits per 100 object episodes, recovery delay and unrecovered fraction, hidden-existence AUROC/AUPR, Brier score, 15 equal-mass-bin ECE, five-state macro-F1, reachable-set recall, mean explicit set size and source/location accuracy. IDF1 uses optimal global predicted-to-reference identity matching within each video. Recovery delay is averaged over eventually recovered episodes and accompanied by unrecovered fraction. A precise ground-truth location is not counted as recalled by the aggregate unknown slot. Undefined class/population metrics are `null`.

Inference timing covers the ledger/controller and excludes the fixed visual front end. Entries grow with the number of encountered objects; each entry's explicit hypothesis support is bounded by `--slots`, independently of video length. Output/audit logs are written separately from the live ledger. The compact transition model uses a learned event MLP and sparse allowed categorical transition kernels. All category/appearance observations and inspected-empty cues must come from the prefix-available front end. Future identity labels contribute delayed training losses only.

## Fixed reproduction setting

The fixed setting generates 12 training and six held-out textured-object scenes. Each scene contains observed noisy appearance patches, measured locations, hand/door cues and explicit empty-location inspections across pick-up, containment, return, leave-view, reappearance and removal. The 128-dimensional appearance descriptor uses pixel block means and gradients from independently sampled material textures; it never encodes object IDs. Removal is terminal.

```bash
python reproduce.py --plan
python reproduce.py --output outputs/finite-setting
```

`finite_data.py --output outputs/finite-setting/objects.jsonl --seed 7` supplies the existing ObjectLedger full/continuous pipeline, with eight scene cues, five location hypotheses, 40 epochs and seed 0. Training annotations and evaluation truth are distinct fields from observations; hidden existence/state labels are not scene cues. Identity metrics are conditioned on the supplied finite detections. Optional `--data objects.jsonl` accepts other observed scenes.
