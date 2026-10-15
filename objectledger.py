import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


EVENTS = ["pick-up", "put-down", "enter-container", "exit-container", "leave-view", "reappear", "remove", "none"]
STATES = ["visible", "occluded", "contained", "out-of-view", "removed"]


class LedgerModel(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        d, cues = spec["feature_dimensions"], spec["cue_dimensions"]
        self.event = torch.nn.Sequential(torch.nn.Linear(d * 2 + cues + 2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
        self.propagate = torch.nn.Linear(d + cues, d)
        self.transition_logits = torch.nn.Parameter(torch.zeros(8, len(spec["locations"]) + 1, len(spec["locations"]) + 1))
        self.register_buffer("allowed", torch.tensor(spec["allowed_transitions"], dtype=torch.bool))
        self.beta = torch.nn.Parameter(torch.tensor([1., .5, .5, .5, -.5]))
        self.existence_event = torch.nn.Parameter(torch.tensor([0., 0., 0., 0., 0., 1., -4., 0.]))
        self.positive = torch.nn.Parameter(torch.tensor(1.))
        self.negative = torch.nn.Parameter(torch.tensor(1.))

    def event_posterior(self, entry, observation, cues, timestamp):
        feature = torch.tensor(observation["feature"], dtype=torch.float32) if observation else torch.zeros_like(entry["prototype"])
        values = torch.cat([entry["prototype"], feature, torch.tensor(cues, dtype=torch.float32), torch.tensor([np.log1p(timestamp - entry["last_seen"])], dtype=torch.float32), entry["logodds"].sigmoid()[None]])
        return torch.softmax(self.event(values), 0)

    def association(self, entry, observation, timestamp, cues):
        a = torch.nn.functional.normalize(torch.tensor(observation["feature"], dtype=torch.float32), dim=0)
        category = torch.tensor(observation["category"], dtype=torch.float32)
        location = int(observation["location"])
        location_probability = entry["hypothesis"][location] + entry["hypothesis"][-1] / len(self.spec["locations"])
        posterior = self.event_posterior(entry, observation, cues, timestamp)
        terms = torch.stack([torch.dot(entry["prototype"], a), torch.dot(entry["category"], category), torch.log(location_probability.clamp_min(1e-6)), posterior[5] - posterior[6], -torch.tensor(min(1., (timestamp - entry["last_seen"]) / self.spec.get("gap_scale", 60.)))])
        return (torch.nn.functional.softplus(self.beta) * terms).sum()

    def update(self, entry, observation, cues, timestamp, empty, args):
        posterior = self.event_posterior(entry, observation, cues, timestamp)
        hard = int(posterior.argmax()) if posterior.max() >= args.event_threshold else None
        transition_posterior = torch.nn.functional.one_hot(torch.tensor(hard), 8).float() if hard is not None else posterior
        transitions = torch.softmax(self.transition_logits.masked_fill(~self.allowed, -1e9), dim=-1)
        hypothesis = sum(transition_posterior[e] * (entry["hypothesis"] @ transitions[e]) for e in range(8))
        propagated = torch.nn.functional.normalize(self.propagate(torch.cat([entry["prototype"], torch.tensor(cues, dtype=torch.float32)])), dim=0)
        if observation:
            likelihood = torch.ones_like(hypothesis) * .05
            likelihood[int(observation["location"])] = .95
            hypothesis = hypothesis * likelihood
        negative = sum(entry["hypothesis"][i] for i in empty) if empty else entry["logodds"] * 0
        logodds = entry["logodds"] + torch.dot(posterior, self.existence_event) + torch.nn.functional.softplus(self.positive) * int(observation is not None) - torch.nn.functional.softplus(self.negative) * negative
        if args.variant == "ttl":
            logodds -= .02 * max(0., timestamp - entry["updated"])
        hypothesis = hypothesis / hypothesis.sum().clamp_min(1e-8)
        if args.variant == "last_location":
            index = int(hypothesis.argmax())
            hypothesis = torch.nn.functional.one_hot(torch.tensor(index), len(hypothesis)).float()
        elif args.slots < len(hypothesis):
            explicit = torch.topk(hypothesis[:-1], min(args.slots - 1, len(hypothesis) - 1)).indices
            keep = torch.zeros_like(hypothesis)
            keep[explicit] = hypothesis[explicit]
            keep[-1] = 1 - keep.sum()
            hypothesis = keep
        old = entry["state"]
        destination = [0, 0, 2, 0, 3, 0, 4, old]
        state_probabilities = torch.zeros(5)
        for event, target in enumerate(destination):
            state_probabilities[target] += posterior[event]
        if hard is not None:
            entry["state"] = destination[hard]
        elif observation is None and entry["state"] == 0 and cues[-1] > .5:
            entry["state"] = 1
        if observation is not None:
            confidence = entry.get("association", 1.)
            visible = observation.get("reliable", True)
            if visible and confidence >= args.identity_threshold:
                feature = torch.nn.functional.normalize(torch.tensor(observation["feature"], dtype=torch.float32), dim=0)
                entry["prototype"] = torch.nn.functional.normalize(.9 * entry["prototype"] + .1 * feature, dim=0)
                entry["category"] = .9 * entry["category"] + .1 * torch.tensor(observation["category"], dtype=torch.float32)
            entry["last_seen"] = timestamp
        elif args.variant == "continuous":
            entry["prototype"] = torch.nn.functional.normalize(.98 * entry["prototype"] + .02 * propagated, dim=0)
        entry.update(hypothesis=hypothesis, logodds=logodds, updated=timestamp)
        return posterior, state_probabilities, propagated


def new_entry(observation, timestamp, model, identity):
    location = torch.nn.functional.one_hot(torch.tensor(int(observation["location"])), len(model.spec["locations"]) + 1).float() * .9
    location[-1] = .1
    return dict(identity=identity, prototype=torch.nn.functional.normalize(torch.tensor(observation["feature"], dtype=torch.float32), dim=0), category=torch.tensor(observation["category"], dtype=torch.float32), logodds=torch.tensor(2.), hypothesis=location, state=0, last_seen=timestamp, updated=timestamp)


def sequence(model, record, args, training):
    ledger = []
    labels_by_identity = {}
    rows, matches = [], []
    loss = sum(p.sum() * 0 for p in model.parameters())
    for frame in record["frames"]:
        timestamp = frame["time"]
        observations = frame["observations"]
        cues = frame["cues"]
        previous = {i: e["hypothesis"] for i, e in enumerate(ledger)}
        n, m = len(ledger), len(observations)
        associations = torch.stack([torch.stack([model.association(e, o, timestamp, cues) for o in observations]) for e in ledger]) if n and m else torch.empty(n, m)
        accepted = {}
        if n and m:
            threshold = float(np.log(args.match_threshold / (1 - args.match_threshold)))
            scores = np.full((n + m, n + m), -1e6)
            scores[:n, :m] = associations.detach().numpy()
            scores[np.arange(n), m + np.arange(n)] = threshold / 2
            scores[n + np.arange(m), np.arange(m)] = threshold / 2
            scores[n:, m:] = 0
            rid, cid = linear_sum_assignment(-scores)
            for i, j in zip(rid, cid):
                if i < n and j < m and scores[i, j] >= threshold:
                    accepted[i] = j
            if training:
                for j, observation in enumerate(observations):
                    truth = observation.get("identity")
                    positive = [i for i in range(n) if labels_by_identity.get(i) == truth]
                    if positive and (not observation.get("verified_reappearance", False) or args.variant != "no_verified_learning"):
                        i = positive[0]
                        loss += -torch.log_softmax(associations[:, j] / .1, 0)[i]
                        if observation.get("verified_reappearance", False) and args.variant != "no_verified_learning":
                            loss += -torch.log(previous[i][int(observation["location"])].clamp_min(1e-6))
        used = set(accepted.values())
        for i, entry in enumerate(ledger):
            observation = observations[accepted[i]] if i in accepted else None
            entry["association"] = float(torch.sigmoid(associations[i, accepted[i]]).detach()) if observation else 0.
            posterior, state_probability, propagated = model.update(entry, observation, cues, timestamp, frame.get("inspected_empty", []), args)
            truth_id = labels_by_identity.get(i)
            truth = frame.get("truth", {}).get(str(truth_id), {})
            if training:
                if observation:
                    target_feature = torch.nn.functional.normalize(torch.tensor(observation["feature"], dtype=torch.float32), dim=0)
                    loss += .05 * torch.nn.functional.mse_loss(propagated, target_feature)
                if "exists" in truth:
                    p = entry["logodds"].sigmoid()
                    loss += (p - truth["exists"]) ** 2
                    loss += .1 * torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), torch.tensor(float(truth["exists"])))
                if "event" in truth:
                    loss += -torch.log(posterior[EVENTS.index(truth["event"])].clamp_min(1e-6))
                if "state" in truth:
                    loss += -torch.log(state_probability[STATES.index(truth["state"])].clamp_min(1e-6))
            if truth:
                rows.append(dict(video=record["video"], time=timestamp, prediction=i, truth_id=str(truth_id), visible=bool(truth.get("visible", False)), exists=int(truth.get("exists", 1)), existence=float(entry["logodds"].sigmoid().detach()), state=STATES[entry["state"]], truth_state=truth.get("state"), location=int(entry["hypothesis"].argmax()), truth_location=truth.get("location"), reachable=[k for k, p in enumerate(entry["hypothesis"][:-1].detach().numpy()) if p > 0], weights=entry["hypothesis"].detach().tolist()))
            if observation:
                matches.append(dict(video=record["video"], time=timestamp, prediction=i, truth_id=str(observation.get("identity", "")), reappearance=bool(observation.get("verified_reappearance", False))))
        for j, observation in enumerate(observations):
            if j in used:
                continue
            identity = len(ledger)
            ledger.append(new_entry(observation, timestamp, model, identity))
            labels_by_identity[identity] = observation.get("identity")
            matches.append(dict(video=record["video"], time=timestamp, prediction=identity, truth_id=str(observation.get("identity", "")), reappearance=bool(observation.get("verified_reappearance", False))))
    tensor_bytes = max((sum(v.numel() * v.element_size() for v in entry.values() if isinstance(v, torch.Tensor)) for entry in ledger), default=0)
    return loss / max(1, len(record["frames"])), rows, matches, len(ledger), tensor_bytes


def prepare_supervision(record):
    record = json.loads(json.dumps(record))
    seen = {}
    for index, frame in enumerate(record["frames"]):
        for observation in frame["observations"]:
            identity = str(observation["identity"])
            if observation.get("verified_reappearance", False) and identity in seen:
                begin = seen[identity] + 1
                removed = any(r.get("truth", {}).get(identity, {}).get("event") == "remove" for r in record["frames"][begin:index])
                if not removed:
                    for hidden in record["frames"][begin:index]:
                        hidden.setdefault("truth", {}).setdefault(identity, {})["exists"] = 1
            seen[identity] = index
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("spec")
    p.add_argument("--output", default="outputs/full")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--slots", type=int, default=5)
    p.add_argument("--match-threshold", type=float, default=.7)
    p.add_argument("--identity-threshold", type=float, default=.8)
    p.add_argument("--event-threshold", type=float, default=.7)
    p.add_argument("--variant", choices=["full", "continuous", "ttl", "last_location", "no_verified_learning"], default="full")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    records = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if len(set(r["video"] for r in records)) != len(records) or args.slots < 2:
        raise ValueError("Use one record per video and at least two hypothesis slots")
    spec = json.loads(Path(args.spec).read_text())
    if "allowed_transitions" not in spec:
        size = len(spec["locations"]) + 1
        masks = np.broadcast_to(np.eye(size, dtype=bool), (8, size, size)).copy()
        masks[:, -1, :] = True
        masks[:, :, -1] = True
        for name, pairs in spec.get("transitions", {}).items():
            event = EVENTS.index(name)
            for source, destination in pairs:
                masks[event, spec["locations"].index(source), spec["locations"].index(destination)] = True
        masks[EVENTS.index("none")] = np.eye(size, dtype=bool)
        spec["allowed_transitions"] = masks.tolist()
    allowed = np.asarray(spec["allowed_transitions"])
    if allowed.shape != (8, len(spec["locations"]) + 1, len(spec["locations"]) + 1) or not allowed.any(-1).all():
        raise ValueError("Every event/source-location transition needs an allowed destination, including unknown")
    torch.manual_seed(args.seed)
    model = LedgerModel(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    history = []
    for epoch in range(args.epochs):
        losses = []
        for raw in records:
            if raw["split"] != "train":
                continue
            record = prepare_supervision(raw) if args.variant != "no_verified_learning" else raw
            loss, _, _, _, _ = sequence(model, record, args, True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses))))
    model.eval()
    rows, matches, entries, entry_bytes = [], [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for record in records:
            if record["split"] == "test":
                _, observations, associations, count, memory = sequence(model, record, args, False)
                rows.extend(observations)
                matches.extend(associations)
                entries.append(count)
                entry_bytes.append(memory)
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(rows)
    associations = pd.DataFrame(matches)
    canonical = associations.sort_values("time").groupby(["video", "truth_id"]).prediction.first()
    correct = np.array([row.prediction == canonical.loc[(row.video, row.truth_id)] for row in associations.itertuples()])
    returned = associations.reappearance.to_numpy(bool)
    idtp = 0
    for video in associations.video.unique():
        group = associations[associations.video == video]
        matrix = pd.crosstab(group.prediction, group.truth_id).to_numpy()
        r, c = linear_sum_assignment(-matrix)
        idtp += int(matrix[r, c].sum())
    episodes = associations.groupby(["video", "truth_id"]).ngroups
    false_merges = int((associations.groupby(["video", "prediction"]).truth_id.nunique() - 1).clip(lower=0).sum())
    false_splits = int((associations.groupby(["video", "truth_id"]).prediction.nunique() - 1).clip(lower=0).sum())
    primary = np.array([row.prediction == canonical.loc[(row.video, row.truth_id)] for row in frame.itertuples()])
    metric_frame = frame[primary]
    hidden = metric_frame[~metric_frame.visible].copy()
    valid = frame.truth_state.notna()
    located = frame.truth_location.notna()
    hidden_probability = hidden.existence.to_numpy()
    hidden_label = hidden.exists.to_numpy()
    ece = 0.
    if len(hidden):
        for ids in np.array_split(np.argsort(hidden_probability), min(15, len(hidden))):
            ece += len(ids) / len(hidden) * abs(hidden_probability[ids].mean() - hidden_label[ids].mean())
    recovery = []
    for row in associations[associations.reappearance].itertuples():
        expected = canonical.loc[(row.video, row.truth_id)]
        later = associations[(associations.video == row.video) & (associations.truth_id == row.truth_id) & (associations.time >= row.time) & (associations.prediction == expected)]
        recovery.append(float(later.time.iloc[0] - row.time) if len(later) else None)
    metrics = dict(reappearance_identity_accuracy=float(correct[returned].mean()) if returned.any() else None, idf1=float(idtp / len(associations)), false_merges_per_100_episodes=100 * false_merges / episodes, false_splits_per_100_episodes=100 * false_splits / episodes, time_to_recover=float(np.mean([t for t in recovery if t is not None])) if any(t is not None for t in recovery) else None, unrecovered_fraction=float(np.mean([t is None for t in recovery])) if recovery else None, hidden_existence_auroc=float(roc_auc_score(hidden_label, hidden_probability)) if len(np.unique(hidden_label)) == 2 else None, hidden_existence_aupr=float(average_precision_score(hidden_label, hidden_probability)) if np.any(hidden_label) else None, brier=float(np.mean((hidden_probability - hidden_label) ** 2)) if len(hidden) else None, ece_equal_mass_15=float(ece) if len(hidden) else None, state_macro_f1=float(f1_score(frame.truth_state[valid], frame.state[valid], labels=STATES, average="macro", zero_division=0)) if valid.any() else None, reachable_set_recall=float(np.mean([int(r.truth_location) in r.reachable for r in frame[located].itertuples()])) if located.any() else None, mean_set_size=float(np.mean([len(v) for v in frame.reachable])), source_accuracy=float((frame.location[located] == frame.truth_location[located]).mean()) if located.any() else None, inference_seconds=elapsed, maximum_entries=max(entries), tensor_bytes_per_entry=max(entry_bytes), parameter_count=sum(p.numel() for p in model.parameters()))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), spec=spec, arguments=vars(args)), out / "model.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    frame.to_json(out / "ledger.jsonl", orient="records", lines=True)
    associations.to_csv(out / "associations.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    frame.groupby("truth_id").existence.mean().plot.bar(color="#4d8796")
    plt.ylabel("Mean existence belief")
    plt.tight_layout()
    plt.savefig(out / "existence.png", dpi=180)


if __name__ == "__main__":
    main()
