"""Append-only raw camera evidence, including misses, before manager selection.

This log records detector outputs, not scheduled frames the detector never processed.
Duplicate or malformed deliveries remain visible and are never silently counted as new.
"""
import json
import math


class CameraOpportunityLog:
    def __init__(self, handle):
        self.handle = handle
        self.seen = set()
        self.rows = 0

    def append(self, camera, payload, receive_stamp):
        record = dict(schema="camera_opportunity_log.v1", topic_camera=camera,
                      receive_stamp_s=float(receive_stamp), delivery_index=self.rows)
        self.rows += 1
        try:
            observation = json.loads(payload)
            if not isinstance(observation, dict):
                raise ValueError("observation is not an object")
            stamp = float(observation["timestamp_s"])
            batch = observation["source_batch_id"]
            if not math.isfinite(stamp) or not isinstance(batch, str) or not batch:
                raise ValueError("missing finite capture stamp or batch identity")
            if observation["camera_id"] != camera:
                raise ValueError("topic camera and observation camera disagree")
            if not isinstance(observation["detection_valid"], bool):
                raise ValueError("detection_valid is not boolean")
            identity = (camera, batch, stamp)
            record.update(observation=observation, duplicate=identity in self.seen,
                          valid_contract=True)
            self.seen.add(identity)
        except (ValueError, KeyError, TypeError) as exc:
            record.update(valid_contract=False, reason=str(exc), raw_payload=payload)
        self.handle.write(json.dumps(record, allow_nan=False, separators=(",", ":"))+"\n")
        self.handle.flush()
        return record
