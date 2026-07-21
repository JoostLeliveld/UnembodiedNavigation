# Paper framings — multi-camera handover & fusion

[Back to module 07](../README.md)

Candidate framings of our own paper centered on this contribution. Each framing
is a way of telling the same evidence; we keep more than one until the pilot
data say which lands best.

## Framing A — *Spatial and Instantaneous Reliability-Aware Multi-Camera Fusion and Planning for Warehouse Robots*
The current working framing: static per-camera calibration is insufficient
because camera usefulness varies with position, individual detections,
availability and calibration health; we combine a spatial GP prior, calibrated
frame-level evidence and online health monitoring into camera-specific
covariance for fusion **and** closed-loop planning.

Plan of record (research questions, hypotheses, contribution claim, experiment
campaign E0–E8, statistics, beats-Toro criteria) lives in the owning study:
- [ROADMAP](../../../experiments/multicamera_fusion_extension/plans/ROADMAP.md)
- [Per-module plans 01–12](../../../experiments/multicamera_fusion_extension/plans/)

> Add a sibling section here when a second framing is drafted (e.g. a
> localization-only framing that drops the planning claim, or a
> calibration-health-centred framing). Keep each framing's evidence pointers
> honest about what is demonstrated vs. blocked on data.
