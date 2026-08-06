# Modular Validation Workflow

This project should move module by module. A module is not "done" when it
produces an output once; it is done when its contract, failure modes, and
downstream consequences have been validated and documented.

## What Was Missed Before

These are the concrete things that caused confusion or rework:

| Missed issue | Why it mattered | New rule |
| --- | --- | --- |
| Moving to the next module before the previous one was fully validated. | Later failures became hard to diagnose because perception, projection, filtering, GP, and planning errors were entangled. | Every module needs an exit checklist before downstream work depends on it. |
| YOLO looked good by detection metrics, but localization error was still high. | A high box/mask metric does not guarantee that the selected bottom-centre pixel projects to the correct warehouse position. | Validate detector output with downstream localization error, not only YOLO precision/recall/mAP. |
| `R` was treated too casually, almost like a scalar. | `R_plan` is a 2x2 measurement covariance matrix in image-pixel units. The current model keeps the matrix diagonal and scales the variance. | Always show the shape and units of `R_plan`; say that the GP learns trust, not `R` itself. |
| The GP-to-`R_plan` story was not intuitive enough. | The collection pipeline and stochastic model looked disconnected. | Explain the chain as `sample pose -> detector score -> GP trust -> sigma_plan^2 -> R_plan`. |
| Odometry and ground truth were easy to mix up. | The system uses odometry for prediction/heading; ground truth is for evaluation and diagnostics, not a runtime state source. | Every plot/table must state whether it uses odometry, belief, state estimate, or ground truth. |
| Camera heading observability was overstated. | The camera measurement contributes image-space `x,y`; heading is odometry-driven and only indirectly affected through covariance. | Document the observed state components before using a module output downstream. |
| Noise terms were introduced without a clear stochastic-model path. | Process noise, command/encoder noise, and measurement noise got blurred together. | Keep a table for each module: what noise is assumed, injected, measured, or only used for evaluation. |
| Ambiguity, obstacle avoidance, no-go costs, and reliability were not cleanly separated. | The audience could read the GP as a traversability map or direct visibility reward. | In planning docs/figures, separate GP trust, `R_plan`, ambiguity, and no-go geometry. |
| Historical and current result surfaces were easy to mix. | Old 12/20 vs 16/20 and current 15/20 vs 20/20 claims can both exist, but not in the same active claim. | Label every result as current, historical, or diagnostic. |
| Calibration details were under-explained. | Homography plus affine correction directly affects the localization point used by later modules. | Show before/after projection residuals and keep calibration parameters traceable. |

## Module Exit Checklist

Before moving from one module to the next, write down and check:

| Gate | Required evidence |
| --- | --- |
| Contract | Inputs, outputs, units, coordinate frames, timestamps, and state components are documented. |
| Quantitative validation | At least one metric tests the module's actual downstream job, not only its internal proxy metric. |
| Qualitative validation | A plot or image shows the module working and shows at least one failure or weak region. |
| Source separation | Runtime inputs are separated from evaluation-only signals such as ground truth. |
| Noise/uncertainty | The relevant stochastic terms are named and mapped to where they enter the system. |
| Failure modes | Known bad cases and limits are documented before downstream claims are made. |
| Reproducibility | The command or artifact path needed to regenerate the evidence is listed. |
| README update | The module README explains the contribution, the validation evidence, and what not to overclaim. |

If a gate is missing, the next module can still be prototyped, but the result
must be labelled as exploratory and should not become a headline claim.

## Validation Ladder

Use this ladder inside each module:

1. **Static contract check**: confirm topic names, file paths, coordinate
   frames, matrix shapes, units, and configuration values.
2. **Single-example check**: inspect one representative frame/run by eye.
3. **Batch metric check**: compute the metric that matters for the next module.
4. **Stress/failure check**: include weak camera regions, stale measurements,
   high projection residuals, or collision-prone routes.
5. **Downstream smoke check**: run the smallest downstream consumer and verify
   that the interface behaves as expected.
6. **Documentation check**: update the README and evidence registry before
   moving on.

## Per-Module Gates

### YOLO Perception

Do not stop at detector mAP. Validate:

- selected bottom-centre pixel quality,
- score distribution over the warehouse,
- missed/weak detections in camera-poor regions,
- projected localization residual after image-to-BEV conversion,
- whether raw confidence is being used only as an empirical reliability proxy.

Exit evidence:

- detector validation image,
- bottom-centre diagnostic image,
- localization residual or projection-error summary,
- detector artifact manifest.

### Image-To-BEV And State Estimation

Validate the measurement before trusting the belief update:

- homography projection against ground truth samples,
- affine correction before/after residuals,
- timestamp freshness and stale-measurement handling,
- odometry-driven heading convention,
- camera `(x, y)` measurement only; no direct camera heading measurement.

Exit evidence:

- image-to-BEV figure,
- affine residual plot,
- topic/dataflow note showing runtime state source,
- explicit distinction between odometry, belief, state estimate, and ground truth.

### GP Reliability And `R_plan`

Validate the data-to-covariance chain:

```text
sample pose -> detector score -> GP trust -> sigma_plan^2 -> R_plan
```

Check:

- training samples cover the relevant warehouse regions,
- trust field looks plausible against known camera-poor regions,
- uncertainty discount is documented,
- `R_plan` matrix shape is shown as 2x2, symmetric, diagonal in the locked setup,
- GP changes measurement covariance only; it is not a map layer or direct route reward.

Exit evidence:

- sample/trust/covariance visual,
- explicit `R_plan` matrix equation with units,
- GP artifact path and manifest,
- command to regenerate the GP or README visual.

### Planning

Validate that planning changes for the intended reason:

- C1 and C2 differ only in planner-facing camera covariance,
- no-go/obstacle geometry is held fixed and shown separately,
- ambiguity is explained as expected observation uncertainty,
- routes are compared on matched task/seed pairs,
- optimizer failures or route-seed sensitivity are not hidden.

Exit evidence:

- matched-pair route visual,
- covariance along route,
- planner config/runtime contract,
- one small smoke run before campaign-level claims.

### Experiments

Validate the claim surface, not only the run logs:

- matched seeds and route definitions are pinned,
- metrics use ground truth only for evaluation,
- current and historical result surfaces are labelled separately,
- invalid, near-success, geometry breach, and physics-contact cases are not collapsed silently,
- every headline table links to config, logs, metrics, and figure provenance.

Exit evidence:

- campaign log,
- current result table,
- robustness/trajectory visual,
- evidence registry entry.

## Working Rule

The default way of working is:

```text
choose one module
-> define its contract
-> implement or adjust it
-> validate its true downstream job
-> document evidence and failure modes
-> only then depend on it from the next module
```

This should feel slower at first, but it prevents the expensive failure mode:
debugging a planner result when the real issue is hidden two modules upstream.
