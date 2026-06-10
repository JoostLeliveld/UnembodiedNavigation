# Appendix: Perception and Gaussian Process Reliability Details

This document compiles the minimal, straight-to-the-point LaTeX formulations and parameters required for reproducing the perception pipeline and Gaussian Process (GP) reliability field.

---

## 1. Main-Text Additions

### Perception Model Summary (to be placed in the main methodology text)
```latex
The external-camera image is processed by a trained YOLOv11 segmentation detector.
For each sampled pose, the pipeline records an image-space robot point and a raw
detector score. Missed detections are retained as zero-score samples when fitting
the reliability field.
```

### GP Method Pointer (to be added at the end of the GP Reliability section)
```latex
The exact GP artifact, fixed kernel parameters, score clipping, output
normalisation, sigma-point spread, and geometry-alignment checks used in the
reported runs are listed in Appendix~\ref{app:gp_reliability_details}.
```

---

## 2. LaTeX Appendix Content

```latex
\section{Perception Model and Detector Outputs}
\label{app:perception_model}

This appendix defines the perception model parameters and training dataset details. The perception dataset and the Gaussian Process (GP) reliability dataset are distinct: the former trains the detector, while the latter aggregates inference outcomes across static poses.

\subsection{Detector Architecture and Training}
The perception pipeline uses a YOLOv11 instance segmentation network fine-tuned on a simulator-specific dataset. The training configuration and parameters are detailed below:
\begin{itemize}
    \item \textbf{Model Checkpoint:} \texttt{logs/perception_models/aws_yolo_simseg_v2/model.pt} (fine-tuned from a pretrained baseline).
    \item \textbf{Input Image Size:} Trained at $640 \times 640$ pixels; evaluated at inference scale $imgsz = 480$ ($480 \times 270$ pixels).
    \item \textbf{Detected Class:} \texttt{robot} (Class ID 0, mapped from Gazebo semantic segment ID 23).
    \item \textbf{Training Split:} Deterministic group-level spatial-yaw bucket split to prevent spatial data leakage. The dataset contains 852 images divided into 683 training and 169 validation samples.
    \item \textbf{Hyperparameters:} Trained for 30 epochs with a batch size of 8, random seed 0, and optimizer auto-selection. Initial learning rate $\eta_0 = 0.01$, final decay fraction $\eta_f = 0.01$, momentum $0.937$, weight decay $0.0005$, and $3.0$ warmup epochs.
    \item \textbf{Augmentation:} Random horizontal flip (probability 0.5), translation (fraction 0.1), scaling (fraction 0.5), mosaic augmentation (probability 1.0), and random erasing (probability 0.4).
\end{itemize}

\subsection{Inference and Output Processing}
At runtime, the detector processes raw RGB frames at a confidence threshold of $\tau_{\mathrm{conf}} = 0.10$ and an intersection-over-union (IoU) threshold of 0.45.
\begin{itemize}
    \item \textbf{Point Selection:} The robot position is tracked using the bottom-centre of the predicted bounding box in image space. If a mask is detected, its centroid or bottom-most boundary is processed.
    \item \textbf{Failed Detections:} If no object is detected above the confidence threshold, the camera correction step is skipped, and the estimator relies on dead reckoning.
    \item \textbf{Score Logging:} The raw confidence score $c_i \in [0.10, 1.00]$ is recorded. Missed detections are logged with a score of $c_i = 0$. Raw confidence values serve as empirical visibility signals and are not treated as calibrated probabilities.
    \item \textbf{Runtime Signals:} Figures depicting camera availability use the raw YOLO detection flag and score. The Normalized Innovation Squared (NIS) gate correction status (\texttt{pixel_corr_accepted}) is disabled at runtime and does not represent visibility.
\end{itemize}


\section{GP Reliability Artifact and Fitting Details}
\label{app:gp_reliability_details}

This appendix records the GP artifact used for the reported route-choice
benchmark. Detector outcome scores are grouped by planar position, with missed
detections retained as zero-score samples. The GP is fit in logit space with
fixed RBF kernel hyperparameters and fixed observation-noise variance. The
reported artifact uses the same world geometry, camera pose, detector checkpoint,
and driveable-region definition as the planner runs.

\subsection{GP Dataset and Fitting}
\begin{itemize}
    \item \textbf{Artifact Path:} \texttt{logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz}
    \item \textbf{Pose Sampling:} Raw scores were captured by teleporting the robot through a 2D spatial grid of size $24 \times 20$ with 4 uniform headings per position ($960$ poses total). Aggregating detections across headings yields $228$ unique training locations $(x, y)$.
    \item \textbf{Score Clipping:} Planar average detector scores $p_i$ are clipped to $[\epsilon, 1-\epsilon]$ with $\epsilon = 0.001$ to prevent singularities in the logit transformation.
    \item \textbf{Logit Transformation:} The target values for GP regression are computed as:
    \begin{equation}
        f_i = \ln \left( \frac{p_i}{1 - p_i} \right)
    \end{equation}
    \item \textbf{Kernel and Regressor:} A scikit-learn \texttt{GaussianProcessRegressor} is fit using a fixed Radial Basis Function (RBF) kernel:
    \begin{equation}
        K(x, x') = \sigma_f^2 \exp \left( - \frac{\|x - x'\|^2}{2 l^2} \right)
    \end{equation}
    with length scale $l = 1.0$ and observation noise variance $\sigma_n^2 = 0.1$. Target normalisation is enabled (\texttt{normalize\_y=True}) and the hyperparameter optimizer is disabled (\texttt{optimizer=None}).
    \item \textbf{Conservative Field:} The planner queries the conservative probability field:
    \begin{equation}
        \rho_{\mathrm{plan}}(x) = \operatorname{sigmoid}(\mu_f(x) - \beta \sigma_f(x))
    \end{equation}
    where $\mu_f(x)$ and $\sigma_f(x)$ are the GP predictive mean and standard deviation in logit space, and $\beta = 1.0$ (or $\beta=0.75$ in legacy environments).
\end{itemize}

\subsection{Planner Integration and Constraints}
\begin{itemize}
    \item \textbf{Sigma-Point Query:} The expected visibility over the robot's belief distribution is computed via an Unscented Transform (UT) query of $\rho_{\mathrm{plan}}$ with spread parameter $\kappa_\sigma = 1.0$.
    \item \textbf{Precision Blending:} Planner-facing measurement covariance transitions between $R_{\mathrm{vis}} = 2.5$ pixels (when $\rho_{\mathrm{plan}} \approx 1$) and $R_{\mathrm{miss}} = 40.0$ pixels (when $\rho_{\mathrm{plan}} \approx 0$).
    \item \textbf{Geometry and Alignment Checklist:} The GP artifact embeds a JSON representation of the 18 collision prisms from \texttt{warehouse_aws.world.sdf} alongside its SHA-256 hash (\texttt{geometry_sha256}). The launch file validates this hash against the active simulator geometry to ensure alignment of obstacles, camera pose, and planner costs.
\end{itemize}
```
