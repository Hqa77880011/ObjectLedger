"""Rendered object appearance and finite pick/hide/reappear/remove scene observations."""

import argparse
import json
from pathlib import Path

import numpy as np


def appearance(rng):
    """Random material texture is sampled independently of object identity labels."""
    yy, xx = np.mgrid[:16, :16]
    phase, frequency = rng.uniform(0, 2 * np.pi, 2), rng.uniform(0.1, 0.7, 2)
    patch = 0.5 + 0.2 * np.sin(xx * frequency[0] + phase[0]) * np.cos(yy * frequency[1] + phase[1])
    return np.clip(patch + rng.normal(0, 0.04, patch.shape), 0, 1)


def feature(patch):
    """128 observed appearance components: block means and horizontal gradients."""
    reduced = patch.reshape(8, 2, 8, 2).mean((1, 3))
    gradient = np.diff(reduced, axis=1, prepend=reduced[:, :1])
    vector = np.r_[reduced.ravel(), gradient.ravel()]
    return (vector / max(np.linalg.norm(vector), 1e-8)).astype(float).tolist()


def object_state(time, offset=0):
    tick = min(time + offset, 31)
    if tick < 5:return 0, "visible", "none", True, True
    if tick < 8:return 1, "visible", "pick-up" if tick == 5 else "none", True, True
    if tick < 15:return 2, "contained", "enter-container" if tick == 8 else "none", False, True
    if tick < 19:return 1, "visible", "exit-container" if tick == 15 else "none", True, True
    if tick < 24:return 4, "out-of-view", "leave-view" if tick == 19 else "none", False, True
    if tick < 28:return 0, "visible", "reappear" if tick == 24 else "none", True, True
    return 0, "removed", "remove" if tick == 28 else "none", False, False


def scene(seed, video, split):
    rng = np.random.default_rng(seed)
    textures = [appearance(rng) for _ in range(2)]
    frames, last_visible = [], [-1, -1]
    for time in range(32):
        observations, truth = [], {}
        states = [object_state(time, offset) for offset in (0, 3)]
        for identity, (location, state, event, visible, exists) in enumerate(states):
            truth[str(identity)] = dict(location=location, state=state, event=event,
                                        visible=visible, exists=int(exists))
            if visible:
                observed = np.clip(textures[identity] + rng.normal(0, 0.035, (16, 16)), 0, 1)
                observations.append(dict(feature=feature(observed), category=[0.7, 0.3],
                    location=location, identity=str(identity), reliable=True,
                    verified_reappearance=last_visible[identity] >= 0 and time - last_visible[identity] > 1))
                last_visible[identity] = time
        # Scene cues are noisy physical camera/door/hand measurements, not event IDs.
        hand_occupancy = sum(state[0] == 1 and state[4] for state in states) / 2
        door_open = any(time + offset in (8, 15) for offset in (0, 3))
        cues = np.array([hand_occupancy, float(door_open), np.sin(time / 3), np.cos(time / 3),
                         0.5, len(observations) / 2, 0.1, 0.5 + 0.4 * np.sin(time / 5)])
        cues = np.clip(cues + rng.normal(0, 0.025, 8), 0, 1).tolist()
        inspected = [0] if time >= 28 else []
        frames.append(dict(time=float(time), observations=observations, cues=cues,
                           inspected_empty=inspected, truth=truth))
    return dict(video=video, split=split, frames=frames, input_source="finite_object_scene")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for split, count in [("train", 12), ("test", 6)]:
            for index in range(count):
                record = scene(args.seed + index + (1000 if split == "test" else 0), f"{split}-{index}", split)
                stream.write(json.dumps(record, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
