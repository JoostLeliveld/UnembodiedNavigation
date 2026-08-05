# Framing — infrastructure localization as an assistive service

[Back to framings](README.md) · supersedes the storyline in
[`ICRA_FRAMING_ADDENDUM_2026-08-04`](ICRA_FRAMING_ADDENDUM_2026-08-04.md) (the *claims*
in that addendum stand; what changes is what the system is FOR)

## 1. The premise that was wrong, and the one that replaces it

Everything in this workstream had been framed as **replacing** onboard localization. That
framing loses to a simple objection: modern AMRs run onboard SLAM, which deploys in days
and needs no infrastructure at all. Argued as a replacement, we are pushing against the
industry's own direction of travel.

The correct framing is **assistive**. The infrastructure does not need to beat a LiDAR.
It needs to provide what a LiDAR structurally cannot:

| onboard SLAM gives | infrastructure gives |
|---|---|
| where **I** am | where **everyone** is |
| a map that drifts per robot, needing merging | one shared global frame by construction |
| what I can see from floor level | what is around the corner I cannot see |
| a robot that cannot know it is lost | an external observer that can |
| self-assessment | supervision by something that is **not the robot** |

Plus deployment: cameras mount anywhere. No cutting floors for guide wire, no surveying
laser reflectors, no floor fiducials to keep clean and re-mark.

## 2. This is a real, named research area — and we land on its stated gap

**[Infrastructure-based Autonomous Mobile Robots for Internal Logistics — Challenges and
Future Perspectives](https://arxiv.org/html/2512.15215v1)** (Dec 2025) is the reference
point. It defines exactly this architecture — the RAIL reference architecture: ceiling
cameras, an on-premise GPU cluster running perception and planning, and robots reduced to
"microcontrollers, safety sensors" executing trajectories over Wi-Fi. Their industrial
deployment is **Volvo Group Truck Operations' heavy-vehicle final assembly plant**: 15
ceiling cameras, mixed traffic with pedestrians and manually driven vehicles, ~150 m
routes, up to 130 transport operations per day.

Their stated motivation is ours: *"removing expensive onboard sensors and computational
units"* while gaining *"enhanced perception... increased computational budget... and
global coordination across a fleet."*

**Three things in that paper position this work precisely.**

**(a) Their open challenge is our contribution.** On localization they write: *"issues such
as occlusions, sensor faults, or limited coverage can disrupt such localization methods"*
and *"a promising direction is to combine infrastructure-based and onboard localization."*
That combination is only safe if the infrastructure reports **honest uncertainty** — which
is what we measure, and what we found it does not do.

**(b) Their pipeline contains the exact unexamined assumption we falsified.** They
calibrate with *"the pinhole camera model"* and a *"custom-made 2×1 metre calibration
target"*, then use a *"pre-computed homography"*. Nothing in that pipeline estimates a
**residual per-camera systematic** after calibration. We measured one: up to **78 mm**,
correlated across frames, surviving commissioning, and invisible to innovation gating.

**(c) They explicitly dismiss the timing term we model.** *"Since the cameras are not
hardware-synchronized, some errors arise due to time shifts between the different cameras.
However, the impact is small given the high processing rate (10 Hz)."* Our innovation
decomposition carries that term explicitly (`R_time = (H ẋ)σ_τ²(H ẋ)ᵀ`), and it is the
reason the same camera at the same place is less accurate on a fast pass than a slow one.
Whether it is negligible is an empirical question they assert rather than measure.

Their own architecture also fuses *"low-rate position estimates... with motor odometry"* —
the identical estimator setup our filter study operates on.

**Consequence for novelty:** the *system concept* is not ours to claim, and we should not
try. What is ours is the **measurement-model and trust contract underneath it**, which
that survey names as open and its deployment assumes away.

## 3. The assist contract

An assistive service is only useful if the client knows when to rely on it. State it as a
contract with three terms:

1. **The infrastructure reports a pose and an honest covariance.** Honest means calibrated:
   the true pose falls inside the stated 95 % region about 95 % of the time. Our
   measurement: the naive service manages **58 %**, claiming 2.8× more precision than it
   has. An assistive service that overstates its accuracy is **worse than no service**,
   because the client stands its own sensing down on a false promise.
2. **The service publishes where it is good, before being asked.** Achievable precision is
   a spatial field, known at commissioning from camera geometry, availability, and each
   camera's residual systematic. The robot should be able to look up "how well can I be
   known there" the same way it looks up a map.
3. **The service degrades legibly.** When a camera drifts, is occluded, or fails, the
   published field must change *before* the pose quality does — not after.

Terms 1 and 3 are measured and satisfied by this work. Term 2 is the
achievable-precision map.

## 4. What the contract unlocks: duty-cycled onboard sensing

This is the concrete benefit and it uses the map we already built.

The robot does not have to choose once between onboard and infrastructure localization. It
can choose **per region**, because achievable precision varies over the floor by a factor
of four in our own network:

- Where infrastructure precision meets the task tolerance → run onboard perception at low
  duty, or off. Save battery and compute.
- Where it does not — deep in an aisle, in a blind zone, near a leaning camera → wake the
  onboard stack.

The energy argument has support: *"larger onboard GPUs drain robot batteries several hours
faster"*
([Offload or Overload](https://arxiv.org/html/2603.18284v1)). Offloading is not free
either — that paper notes network latency degrades accuracy and naive cloud offloading is
impractical — which is precisely why the decision should be **spatial and precomputed**
rather than per-frame and reactive.

**Honesty note:** we have *not* measured a battery saving. State it as the motivation for
the policy, and claim only what we measured — that the policy is computable and that the
field it depends on is non-uniform. Quantifying the saving is future work, and it needs a
robot power model we do not have.

## 5. What this changes in the paper

Nothing in the evidence changes. The claims, the nulls and the numbers all stand. What
changes is what they are *for*:

| result | old role | role under the assist framing |
|---|---|---|
| 78 mm correlated per-camera bias | a flaw in our pipeline | the unexamined assumption in the reference architecture |
| belief overconfident, 42 % outside 95 % | a filtering problem | **violation of the assist contract** — the safety gate |
| correlation floor restores honesty | a filter trick | the mechanism that makes the contract satisfiable |
| gated calibration, correct-the-outliers | a calibration result | what makes deployment *easy* — you need not calibrate a network perfectly, only know which parts to trust |
| achievable-precision map | a planning curiosity | **the published service map**, and the basis for duty-cycling |
| single-`R` understates posterior ~90 % | planner correctness | why the *client* cannot be told a single number per position |

The headline becomes: **an assistive localization service is only useful if it is honest
about its own uncertainty; here is what makes it dishonest, and what fixes it.**

## 6. The facility

The reference deployment is a **mixed-traffic industrial internal-logistics environment**
— a vehicle assembly plant, not a dark warehouse. That matters: pedestrians and manually
driven vehicles are present, routes are long (~150 m), and the layout changes as product
mix changes. It is also brownfield, which is what makes "cameras mount anywhere, no floor
work" the deciding property.

Our facility should match that class: a working distribution/logistics floor with mixed
human and robot traffic, an inherited camera network, long haul routes between zones, and
storage structure that genuinely occludes. See the redesigned layout in
`experiments/warehouse_layout_sketches/`.

## Sources

- [Infrastructure-based AMRs for Internal Logistics — Challenges and Future Perspectives](https://arxiv.org/html/2512.15215v1) (RAIL architecture, Volvo deployment, open challenges)
- [Offload or Overload: A Platform Measurement Study of Mobile Robotic Manipulation Workloads](https://arxiv.org/html/2603.18284v1) (onboard GPU battery cost, offload latency trade-off)
- [Set-theoretic Localization for Mobile Robots with Infrastructure-based Sensing](https://arxiv.org/pdf/2110.01749) (infrastructure-based localization, related method)
- [Review of Autonomous Mobile Robots for the Warehouse Environment](https://arxiv.org/pdf/2406.08333)
- [MAPF: Definitions, Variants, and Benchmarks](https://arxiv.org/abs/1906.08291) (warehouse grid structure)
