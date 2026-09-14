# Commissioning v1 pilot and freeze report

The excluded pilot `fit_west_spine_fwd` passed on 2026-09-10 in
`logs/commissioning/reference_controlled_v1_pilot_20260910_v12`. This pilot is an
implementation check and is not part of the fit, development or audit evidence.

## Frozen acquisition decision

- Nominal straight-line cruise is 1.0 m/s. A 0.22 m/s nominal campaign is forbidden by
  the protocol validator.
- Five cameras run at 5 Hz with the frozen 960 px YOLO checkpoint declared in
  `campaign.yaml`.
- The reference controller alone commands the route. The fused estimator runs in shadow
  to close the correction ledger and cannot affect the path.
- Learning uses the pre-NIS detector and manager journals. NIS is diagnostic and is not a
  label for bias, availability or covariance learning.
- The 24 drives remain split by whole drive into eight fit, eight development and eight
  sealed audit drives. No scientific drive had been started at this freeze point.

## Passing pilot evidence

| Gate | v8 result |
|---|---:|
| Controller route and stationary anchors complete | yes |
| Declared nominal cruise | 1.0 m/s |
| Repository `valid_run` | true |
| Terminal stop verified | true |
| Correction ledger valid | true |
| Raw frame identity rows | 955 |
| Unique physical camera opportunities | 775 |
| Complete five-camera network rounds | 155 |
| Frame identity match fraction | 1.0 |
| Native-reference support fraction | 1.0 |
| Duplicate frame identities | 0 |
| Synchronized measurement rows | 289 |
| Selected RGB crops present | 289/289 |
| Selected RGB crop identity match fraction | 1.0 |

The pre-freeze campaign SHA-256 recorded by the passing table builder was
`93919fecfec73580494ce38cd378e37bdca20e78c9fdf6ac71a77c0f9b81f8c7`.
The final metadata-only freeze binds the acquisition scripts, launch helper and detector
source executed by v12. The frozen campaign SHA-256 is
`9df86f5fed9a8b0c61c258ac644b6a27e57bcdd14ae565f2bacf824208422c06`.
Every scientific drive manifest must record this hash.

## Rejected implementation pilots

Earlier pilot roots are retained and excluded. V9 was denied local UDP sockets by the
execution sandbox. V10 exposed two orphaned Gazebo servers from the failed scientific
attempt and stopped on CUDA memory exhaustion. V11 passed the route, ledger and
synchronization gates but failed the new crop gate with 0/283 crops. V12 is the first pilot
that passed the runtime gates and the selected-crop gate. Failed pilot directories must not
be reused or counted as campaign evidence.
