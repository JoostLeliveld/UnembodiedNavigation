"""Per-camera Bayesian filtering of the ground-anchor scale and shift.

The single-frame robust fit remains the measurement generator.  This module
adds a small state-space layer over its two affine parameters, so successive
frames share information without ever sharing depth pixels or oracle data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .contracts import DepthConvention, FrameStatus, GroundFit


@dataclass(frozen=True)
class TemporalAnchorConfig:
    """Conservative defaults for a roughly ten-second map refresh."""

    information_half_life_s: float = 60.0
    default_dt_s: float = 10.0
    max_stale_s: float = 30.0
    student_t_df: float = 5.0
    innovation_gate_mahalanobis2: float = 16.0
    min_scale_relative_sigma: float = 0.005
    min_shift_sigma_depth: float = 0.01
    min_shift_sigma_inverse: float = 0.002
    reuse_prior_on_refusal: bool = True


@dataclass
class _TemporalState:
    mean: np.ndarray
    covariance: np.ndarray
    timestamp_s: float
    last_good_timestamp_s: float
    residual_variance: float
    accepted_updates: int
    last_fit: GroundFit


def _nearest_psd(matrix: np.ndarray, floor: float = 1e-16) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


class TemporalGroundAnchorFilter:
    """Independent Bayesian affine state for every camera/model/convention."""

    def __init__(self, config: TemporalAnchorConfig | None = None) -> None:
        self.config = config or TemporalAnchorConfig()
        self._states: dict[tuple[str, str, str], _TemporalState] = {}

    @staticmethod
    def _key(
        camera_id: str, model_name: str, convention: DepthConvention
    ) -> tuple[str, str, str]:
        return str(camera_id), str(model_name), convention.value

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._states.clear()
            return
        self._states = {
            key: value for key, value in self._states.items() if key[0] != camera_id
        }

    def _timestamp(self, timestamp_s: float, state: _TemporalState | None) -> float:
        if np.isfinite(timestamp_s):
            value = float(timestamp_s)
            if state is not None and value < state.timestamp_s - 1e-9:
                raise ValueError(
                    "temporal anchoring received non-monotonic time "
                    f"{value} after {state.timestamp_s}"
                )
            return value
        return 0.0 if state is None else state.timestamp_s + self.config.default_dt_s

    def _floors(self, mean: np.ndarray, convention: DepthConvention) -> np.ndarray:
        scale_sigma = self.config.min_scale_relative_sigma * max(abs(float(mean[0])), 1e-6)
        shift_sigma = (
            self.config.min_shift_sigma_inverse
            if convention.is_inverse
            else self.config.min_shift_sigma_depth
        )
        return np.asarray([scale_sigma**2, shift_sigma**2], dtype=float)

    def _measurement_covariance(self, fit: GroundFit) -> np.ndarray:
        mean = np.asarray([fit.scale, fit.shift], dtype=float)
        covariance = np.asarray(fit.parameter_covariance, dtype=float)
        if not np.isfinite(covariance).all():
            covariance = np.zeros((2, 2), dtype=float)
        covariance = _nearest_psd(covariance)
        covariance += np.diag(self._floors(mean, fit.convention))
        return _nearest_psd(covariance)

    def _predict(
        self, state: _TemporalState, timestamp_s: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        dt = max(0.0, timestamp_s - state.timestamp_s)
        half_life = max(self.config.information_half_life_s, 1e-6)
        retained_information = math.exp(-math.log(2.0) * dt / half_life)
        covariance = state.covariance / max(retained_information, 1e-6)
        covariance += np.diag(
            self._floors(state.mean, state.last_fit.convention)
        ) * (1.0 - retained_information)
        return state.mean.copy(), _nearest_psd(covariance), dt

    @staticmethod
    def _fit_with_posterior(
        fit: GroundFit,
        mean: np.ndarray,
        covariance: np.ndarray,
        residual_variance: float,
        note: str,
    ) -> GroundFit:
        notes = "; ".join(value for value in (fit.notes, note) if value)
        return replace(
            fit,
            scale=float(mean[0]),
            shift=float(mean[1]),
            sigma_fit=float(math.sqrt(max(residual_variance, 0.0))),
            parameter_cov=_nearest_psd(covariance),
            notes=notes,
        )

    def update(
        self,
        fit: GroundFit,
        *,
        camera_id: str,
        model_name: str,
        timestamp_s: float = float("nan"),
    ) -> tuple[GroundFit, dict]:
        key = self._key(camera_id, model_name, fit.convention)
        state = self._states.get(key)
        now = self._timestamp(timestamp_s, state)

        if state is None:
            if not fit.status.is_ok:
                return fit, {
                    "mode": "no_prior_refused",
                    "accepted": False,
                    "camera_id": camera_id,
                    "model_name": model_name,
                }
            mean = np.asarray([fit.scale, fit.shift], dtype=float)
            covariance = self._measurement_covariance(fit)
            residual_variance = max(float(fit.sigma_fit)**2, 1e-12)
            filtered = self._fit_with_posterior(
                fit, mean, covariance, residual_variance, "temporal Bayesian initialisation"
            )
            self._states[key] = _TemporalState(
                mean=mean,
                covariance=covariance,
                timestamp_s=now,
                last_good_timestamp_s=now,
                residual_variance=residual_variance,
                accepted_updates=1,
                last_fit=filtered,
            )
            return filtered, {
                "mode": "initialised",
                "accepted": True,
                "accepted_updates": 1,
                "posterior_mean": mean.tolist(),
                "posterior_covariance": covariance.tolist(),
            }

        predicted_mean, predicted_covariance, dt = self._predict(state, now)
        stale_age = now - state.last_good_timestamp_s

        if not fit.status.is_ok:
            state.mean = predicted_mean
            state.covariance = predicted_covariance
            state.timestamp_s = now
            if self.config.reuse_prior_on_refusal and stale_age <= self.config.max_stale_s:
                filtered = self._fit_with_posterior(
                    state.last_fit,
                    predicted_mean,
                    predicted_covariance,
                    state.residual_variance,
                    f"temporal prior reused after {fit.status.value}; age={stale_age:.1f}s",
                )
                return filtered, {
                    "mode": "stale_prior",
                    "accepted": False,
                    "refused_status": fit.status.value,
                    "stale_age_s": stale_age,
                    "posterior_mean": predicted_mean.tolist(),
                    "posterior_covariance": predicted_covariance.tolist(),
                }
            return fit, {
                "mode": "stale_limit_exceeded",
                "accepted": False,
                "refused_status": fit.status.value,
                "stale_age_s": stale_age,
            }

        measurement = np.asarray([fit.scale, fit.shift], dtype=float)
        measurement_covariance = self._measurement_covariance(fit)
        innovation = measurement - predicted_mean
        innovation_covariance = _nearest_psd(predicted_covariance + measurement_covariance)
        mahalanobis2 = float(innovation @ np.linalg.inv(innovation_covariance) @ innovation)
        if mahalanobis2 > self.config.innovation_gate_mahalanobis2:
            state.mean = predicted_mean
            state.covariance = predicted_covariance
            state.timestamp_s = now
            if self.config.reuse_prior_on_refusal and stale_age <= self.config.max_stale_s:
                filtered = self._fit_with_posterior(
                    state.last_fit,
                    predicted_mean,
                    predicted_covariance,
                    state.residual_variance,
                    f"temporal innovation rejected d2={mahalanobis2:.2f}; age={stale_age:.1f}s",
                )
                return filtered, {
                    "mode": "innovation_rejected_prior",
                    "accepted": False,
                    "innovation_mahalanobis2": mahalanobis2,
                    "stale_age_s": stale_age,
                    "posterior_mean": predicted_mean.tolist(),
                    "posterior_covariance": predicted_covariance.tolist(),
                }
            rejected = replace(
                fit,
                status=FrameStatus.HIGH_RESIDUAL,
                notes="; ".join(value for value in (
                    fit.notes,
                    f"temporal innovation d2={mahalanobis2:.2f} exceeded gate",
                ) if value),
            )
            return rejected, {
                "mode": "innovation_rejected_stale",
                "accepted": False,
                "innovation_mahalanobis2": mahalanobis2,
                "stale_age_s": stale_age,
            }

        degrees = max(self.config.student_t_df, 1.0)
        robust_weight = min(1.0, (degrees + 2.0) / (degrees + mahalanobis2))
        effective_measurement_covariance = measurement_covariance / max(robust_weight, 0.05)
        posterior_information = (
            np.linalg.inv(predicted_covariance) + np.linalg.inv(effective_measurement_covariance)
        )
        posterior_covariance = _nearest_psd(np.linalg.inv(posterior_information))
        posterior_mean = posterior_covariance @ (
            np.linalg.inv(predicted_covariance) @ predicted_mean
            + np.linalg.inv(effective_measurement_covariance) @ measurement
        )
        retained = math.exp(-math.log(2.0) * dt / max(self.config.information_half_life_s, 1e-6))
        residual_variance = (
            retained * state.residual_variance
            + (1.0 - retained) * max(float(fit.sigma_fit)**2, 1e-12)
        )
        filtered = self._fit_with_posterior(
            fit,
            posterior_mean,
            posterior_covariance,
            residual_variance,
            f"temporal Bayesian update weight={robust_weight:.3f} d2={mahalanobis2:.2f}",
        )
        state.mean = posterior_mean
        state.covariance = posterior_covariance
        state.timestamp_s = now
        state.last_good_timestamp_s = now
        state.residual_variance = residual_variance
        state.accepted_updates += 1
        state.last_fit = filtered
        return filtered, {
            "mode": "updated",
            "accepted": True,
            "accepted_updates": state.accepted_updates,
            "innovation_mahalanobis2": mahalanobis2,
            "student_weight": robust_weight,
            "posterior_mean": posterior_mean.tolist(),
            "posterior_covariance": posterior_covariance.tolist(),
        }

    def snapshot(self) -> dict:
        return {
            "/".join(key): {
                "mean": state.mean.tolist(),
                "covariance": state.covariance.tolist(),
                "timestamp_s": state.timestamp_s,
                "last_good_timestamp_s": state.last_good_timestamp_s,
                "accepted_updates": state.accepted_updates,
            }
            for key, state in sorted(self._states.items())
        }


__all__ = ["TemporalAnchorConfig", "TemporalGroundAnchorFilter"]
