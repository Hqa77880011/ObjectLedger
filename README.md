# ObjectLedger

Implementation of **ObjectLedger: Event-Driven and Reappearance-Verified Object Permanence in Long Egocentric Video**.

## Installation

Requires Python `>=3.11,<3.13`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0, pandas 2.2.3. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Generate finite textured-object observations with causal scene cues and separate identity/state truth:

```bash
python finite_data.py --output data/objects.jsonl --seed 7
```

For an external fixed visual frontend, Use object/interaction observations from EPIC-KITCHENS/VISOR or Ego4D, keeping the same detector, appearance encoder, category predictions and event cues across methods. This implementation starts at that fixed front end. Export one JSON object per video into `data/videos.jsonl`; `video` is unique and `split` is `train`, `val` or `test`. Split complete recording/participant environments before fitting.

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

## Training and inference

```bash
objectledger data/objects.jsonl ledger.example.json --output outputs/full --epochs 40 --slots 5 --seed 0
```

## Evaluation

The command fits transitions and evaluates tracking on held-out complete scenes.

## Comparisons and ablations

```bash
objectledger data/objects.jsonl ledger.example.json --variant continuous --output outputs/continuous
objectledger data/objects.jsonl ledger.example.json --variant ttl --output outputs/ttl
objectledger data/objects.jsonl ledger.example.json --variant last_location --output outputs/last_location
objectledger data/objects.jsonl ledger.example.json --variant no_verified_learning --output outputs/no_verified_learning
```

## Outputs

`model.pt`, `history.csv`, `ledger.jsonl`, `associations.csv`, `existence.png` and `metrics.json` retain model/configuration, losses, beliefs, associations and tracking measurements. The finite full/continuous pipeline is also available via:

```bash
python reproduce.py --plan
python reproduce.py --output outputs/finite-setting
```
