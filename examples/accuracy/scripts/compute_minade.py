# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compute minADE from action_inference input.json + output.json.

Self-contained: does not import the alpamayo_r1 package.

Reference: chefu PyTorch baseline minADE = 0.82113 m on 644 clips.
"""
import argparse
import csv
import json
import os
from collections import Counter

import einops
import numpy as np
import torch

# Action-space constants for Alpamayo-R1-10B. Sourced from the HuggingFace
# config (nvidia/Alpamayo-R1-10B/config.json -> action_space) and mirrored in
# tensorrt_edgellm/action_models/alpamayo_r1/action_space/. Keep in sync with
# the checkpoint: changing any of these silently breaks waypoint accuracy.
DT = 0.1  # seconds per waypoint (10 Hz)
N_WAYPOINTS = 64  # 6.4 s prediction horizon
ACCEL_MEAN = 0.029052734375  # accel channel normalisation
ACCEL_STD = 0.6796875
CURVATURE_MEAN = 0.0002689361572265625  # curvature channel normalisation
CURVATURE_STD = 0.026123046875
V_LAMBDA = 1e-06  # 3rd-order Tikhonov weight for v0
V_RIDGE = 0.0001  # ridge weight for v0 least-squares
STEPS_3S = 30  # waypoint count for the 3.0 s horizon
STEPS_6S = 64  # waypoint count for the 6.4 s horizon


def so3_to_yaw(rot_mat):
    return torch.atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])


def unwrap_angle(phi):
    d = torch.diff(phi, dim=-1)
    d = torch.atan2(torch.sin(d), torch.cos(d))
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)],
                     dim=-1)


def third_order_D(N, lead, device, dtype):
    D = torch.zeros(*lead, max(N - 3, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 3, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 3.0
    D[..., rows, rows + 2] = -3.0
    D[..., rows, rows + 3] = 1.0
    return D


def construct_DTD(N, lead, device, dtype, lam, dt):
    lam_3 = lam / dt**6
    w = torch.full((*lead, max(N - 3, 0)), 1.0, dtype=dtype, device=device)
    D3 = third_order_D(N, lead, device=device, dtype=dtype)
    return lam_3 * einops.einsum(D3 * w.unsqueeze(-1), D3,
                                 "... i j, ... i k -> ... j k")


@torch.no_grad()
def estimate_v0(traj_history_xyz, traj_history_rot):
    full_xy = traj_history_xyz[..., :2]
    dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
    theta = so3_to_yaw(traj_history_rot)
    theta = unwrap_angle(theta)

    lead = list(dxy.shape[:-2])
    N = dxy.shape[-2]
    device, dtype = dxy.device, dxy.dtype

    g = 2 / DT * dxy
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    A = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b = g.flatten(start_dim=-2)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A[..., cos_rows, cols] = cos_theta[..., :-1]
    A[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A[..., sin_rows, cols] = sin_theta[..., :-1]
    A[..., sin_rows, cols + 1] = sin_theta[..., 1:]

    ATA = einops.einsum(A, A, "... i j, ... i k -> ... j k")
    rhs = einops.einsum(A, b, "... i j, ... i -> ... j")

    DTD = construct_DTD(N + 1,
                        tuple(lead),
                        device=device,
                        dtype=dtype,
                        lam=V_LAMBDA,
                        dt=DT)
    ridge = V_RIDGE * torch.eye(N + 1, dtype=dtype, device=device).expand(
        *lead, N + 1, N + 1)

    lhs = ATA + DTD + ridge
    L = torch.linalg.cholesky(lhs)
    v = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    return v[..., -1]


def validate_samples(requests, responses, num_samples):
    """Require complete finite trajectories for every requested clip."""
    if num_samples < 1 or not requests or len(requests) != len(responses):
        raise ValueError(
            "Expected a positive sample count and one response per request")
    counts = Counter(request["id"] for request in requests)
    if any(count < num_samples for count in counts.values()):
        raise ValueError(
            "Every clip must supply the requested number of candidates")
    for index, response in enumerate(responses):
        trajectory = np.asarray(response.get("output_trajectory", []),
                                dtype=np.float32)
        if trajectory.shape != (N_WAYPOINTS,
                                2) or not np.isfinite(trajectory).all():
            raise ValueError(
                f"Response {index} requires a finite {N_WAYPOINTS}x2 trajectory"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        required=True,
                        help="action_inference input JSON")
    parser.add_argument("--output",
                        required=True,
                        help="action_inference output JSON")
    parser.add_argument(
        "--gt",
        required=True,
        help="GT JSON (clip_id -> gt_xy/ego_history_xyz/ego_history_rot)")
    parser.add_argument("--num_traj_samples", type=int, default=6)
    parser.add_argument("--csv_out", default=None)
    args = parser.parse_args()

    N = args.num_traj_samples

    with open(args.input) as f:
        inp = json.load(f)
    with open(args.output) as f:
        out = json.load(f)
    with open(args.gt) as f:
        gt_data = json.load(f)

    requests = inp["requests"]
    responses = out["responses"]
    validate_samples(requests, responses, N)
    print(
        f"Requests: {len(requests)}, Responses: {len(responses)}, GT clips: {len(gt_data)}"
    )

    clip_data = {}
    for req, resp in zip(requests, responses):
        cid = req["id"]
        if cid not in clip_data:
            gt = gt_data[cid]
            clip_data[cid] = {
                "gt_xy":
                np.array(gt["gt_xy"], dtype=np.float32),
                "ego_history_xyz":
                torch.tensor(gt["ego_history_xyz"], dtype=torch.float32),
                "ego_history_rot":
                torch.tensor(gt["ego_history_rot"], dtype=torch.float32),
                "responses": [],
            }
        clip_data[cid]["responses"].append(resp)

    print(f"Clips: {len(clip_data)}")

    csv_out = args.csv_out or os.path.join(os.path.dirname(args.output),
                                           "minade_results.csv")
    fieldnames = ["clip_id", "min_ade_3s", "min_ade_6.4s"]
    min_ades_3s = []
    min_ades_6s = []
    skipped = 0

    with open(csv_out, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (cid, cd) in enumerate(sorted(clip_data.items())):
            resps = cd["responses"]
            gt_xy = cd["gt_xy"]
            actions = np.stack([
                np.array(r["output_trajectory"], dtype=np.float32)
                for r in resps[:N]
            ])
            sampled_action = torch.from_numpy(actions).reshape(N, 64, 2)

            hist_xyz = cd["ego_history_xyz"]
            hist_rot = cd["ego_history_rot"]

            hist_xyz_full = hist_xyz.unsqueeze(0).expand(N, -1, -1)
            hist_rot_full = hist_rot.unsqueeze(0).expand(N, -1, -1, -1)

            v0 = estimate_v0(hist_xyz_full, hist_rot_full)

            accel = sampled_action[..., 0] * ACCEL_STD + ACCEL_MEAN
            kappa = sampled_action[..., 1] * CURVATURE_STD + CURVATURE_MEAN

            dt = DT
            dt_2 = 0.5 * dt**2
            half_dt = 0.5 * dt

            velocity = torch.cat([
                v0.unsqueeze(-1),
                v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1)
            ],
                                 dim=-1)
            initial_yaw = torch.zeros_like(v0)
            theta = torch.cat([
                initial_yaw.unsqueeze(-1),
                initial_yaw.unsqueeze(-1) +
                torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1) +
                torch.cumsum(kappa * accel * dt_2, dim=-1),
            ],
                              dim=-1)

            x = torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt, dim=-1) \
                + torch.cumsum(velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt, dim=-1)
            y = torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt, dim=-1) \
                + torch.cumsum(velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt, dim=-1)

            pred_xy = torch.stack([x, y], dim=-1).numpy()
            d = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)

            ade_3s = d[:, :STEPS_3S].mean(axis=-1)
            min_ade_3s = float(ade_3s.min())
            ade_6s = d[:, :STEPS_6S].mean(axis=-1)
            min_ade_6s = float(ade_6s.min())

            writer.writerow({
                "clip_id": cid,
                "min_ade_3s": round(min_ade_3s, 5),
                "min_ade_6.4s": round(min_ade_6s, 5)
            })
            min_ades_3s.append(min_ade_3s)
            min_ades_6s.append(min_ade_6s)

            if (idx + 1) % 50 == 0:
                print(
                    f"  [{idx+1}/{len(clip_data)}] avg minADE@3s={np.mean(min_ades_3s):.4f}  @6.4s={np.mean(min_ades_6s):.4f}"
                )

    print(f"\n{'='*60}")
    print(f"Clips evaluated: {len(min_ades_6s)}, skipped: {skipped}")
    print(f"")
    print(f"  minADE6@3s:   {np.mean(min_ades_3s):.5f} m")
    print(f"  minADE6@6.4s: {np.mean(min_ades_6s):.5f} m")
    print(f"")
    print(f"Reference (chefu PyTorch, 644 clips):")
    print(f"  minADE6@6.4s: 0.82113 m")
    print(f"Reference (paper Table 6, AR1 0.5B + Route):")
    print(f"  minADE6@3s:   0.254 m")
    print(f"  minADE6@6.4s: 0.794 m")
    print(f"")
    print(f"CSV: {csv_out}")
    # Stable keys: accuracy.py greps MINADE_3S / MINADE_6S / MINADE_CLIPS.
    print(f"MINADE_3S={np.mean(min_ades_3s):.5f}")
    print(f"MINADE_6S={np.mean(min_ades_6s):.5f}")
    print(f"MINADE_CLIPS={len(min_ades_6s)}")


if __name__ == "__main__":
    main()
