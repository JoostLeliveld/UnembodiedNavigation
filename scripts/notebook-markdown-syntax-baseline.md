# Notebook Markdown Syntax Baseline

This file is a direct extraction of markdown cells from the selected notebooks.
It serves as the writing and syntax baseline for future notebook markdown in this repository.

## Source: `botnav-cart2polar-actionspace-compare.ipynb`

<!-- markdown-cell 01 -->
# Expected Free Energy Navigation: Point-Mass vs Differential-Drive

This notebook compares the two action spaces directly using the **original Julia cart2polar observation**:

\[
 g(x) = [\sqrt{x^2+y^2},\ \operatorname{atan2}(y,x)]
\]

So the comparison isolates dynamics/action-space effects under the same observation geometry and similar benchmark settings.

<!-- markdown-cell 02 -->
### Joost Leliveld, update: 02-03-2026

<!-- markdown-cell 03 -->
### Use case

- Same observation function and preference model for both planners.
- Same time settings and planning horizon.
- Compare behavior for:
  - point-mass control space `u=[u_x,u_y]`
  - differential-drive control space `u=[v,\omega]`

<!-- markdown-cell 04 -->
### Dynamics

Point-mass: linear constant-velocity model (`FieldBot`, `EFEAgent`).

Differential-drive: unicycle kinematics (`UnicycleBot`, `UnicycleEFEAgent`).

<!-- markdown-cell 05 -->
### Observations (Julia cart2polar)

Observation for both models depends only on world position `(x,y)`:

\[
 y = [r,\phi] = [\sqrt{x^2+y^2},\ \operatorname{atan2}(y,x)]
\]

<!-- markdown-cell 06 -->
## Model: EFE(ET2) single-run comparison

<!-- markdown-cell 07 -->
## Monte Carlo comparison (same observation model, different action spaces)

---

## Source: `camera-projection-investigation.ipynb`

<!-- markdown-cell 01 -->
# Camera Projection Investigation

**Goal**  
Derive the planar homography projection to the image plane, compute the Jacobian and Hessian, and connect these to the ambiguity term in EFE.

**Outline**
- Projection model (world → camera → pixels) using a homography
- Analytic Jacobian of the pixel mapping
- Hessian terms and where they blow up
- Ambiguity intuition + numerical example

<!-- markdown-cell 02 -->
## From ground-plane point to pixels


### Ground-plane point in the world frame


A point on the ground plane can be parameterized by its planar coordinates $(x,y)$ as

$$
\mathbf{p}_w(x,y)
=
\begin{bmatrix}
x \\ y \\ 0
\end{bmatrix}.
$$

To apply rigid transformations using matrix multiplication, we embed this Euclidean point
into **homogeneous coordinates** by appending a constant scale coordinate equal to 1:

$$
\tilde{\mathbf{P}}_w(x,y)
=
\begin{bmatrix}
X \\ Y \\ Z \\ 1
\end{bmatrix}
=
\begin{bmatrix}
x \\ y \\ 0 \\ 1
\end{bmatrix}.
$$

The tilde indicates that this is a homogeneous representation.
Any nonzero scalar multiple of this vector represents the same Euclidean point.

---

### 1) World $\rightarrow$ camera coordinates (extrinsic parameters)

The camera pose with respect to the world frame is described by:

- a rotation matrix $R_{cw} \in SO(3)$, mapping world axes to camera axes,
- a translation vector $t_{cw} \in \mathbb{R}^3$, giving the camera origin in world coordinates.

For a Euclidean 3D point $\mathbf{p}_w = (X,Y,Z)^\top$, the corresponding point in the
camera frame is

$$
\mathbf{p}_c
=
\begin{bmatrix}
X_c \\ Y_c \\ Z_c
\end{bmatrix}
=
R_{cw}
\begin{bmatrix}
X \\ Y \\ Z
\end{bmatrix}
+
t_{cw}.
$$

Substituting the ground-plane point $(X,Y,Z) = (x,y,0)$ yields

$$
\begin{bmatrix}
X_c \\ Y_c \\ Z_c
\end{bmatrix}
=
R_{cw}
\begin{bmatrix}
x \\ y \\ 0
\end{bmatrix}
+
t_{cw}.
$$

Writing the rotation matrix in terms of its column vectors,

$$
R_{cw} =
\begin{bmatrix}
r_1 & r_2 & r_3
\end{bmatrix},
$$

we obtain

$$
\begin{bmatrix}
X_c \\ Y_c \\ Z_c
\end{bmatrix}
=
\begin{bmatrix}
r_1 & r_2
\end{bmatrix}
\begin{bmatrix}
x \\ y
\end{bmatrix}
+
t_{cw}.
$$

Thus, when restricted to the ground plane, the camera-frame coordinates are an
**affine function** of the planar position $(x,y)$.

For later convenience, this affine mapping can be written in homogeneous form as

$$
\mathbf{p}_c(x,y)
=
\begin{bmatrix}
X_c \\ Y_c \\ Z_c
\end{bmatrix}
=
\begin{bmatrix}
r_1 & r_2 & t_{cw}
\end{bmatrix}
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}.
$$

---

### 2) Camera coordinates $\rightarrow$ normalized image coordinates (perspective projection)

The pinhole camera model maps 3D camera-frame points to **normalized image coordinates**
by perspective division:

$$
x_n = \frac{X_c}{Z_c},
\qquad
y_n = \frac{Y_c}{Z_c}.
$$

This division by depth $Z_c$ is the fundamental nonlinearity of the camera model.
It is also the source of projective effects such as vanishing points and
state-dependent sensitivity.

---

### 3) Normalized image coordinates $\rightarrow$ pixels (intrinsic parameters)

The mapping from normalized image coordinates to pixel coordinates is described by
the camera intrinsic matrix

$$
K =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix},
$$

where $f_x,f_y$ are focal lengths in pixel units and $(c_x,c_y)$ is the principal point.

In homogeneous coordinates, the projection to pixels can be written as

$$
\lambda
\begin{bmatrix}
u \\ v \\ 1
\end{bmatrix}
=
K
\begin{bmatrix}
X_c \\ Y_c \\ Z_c
\end{bmatrix},
\qquad \lambda \neq 0.
$$

The scalar $\lambda$ reflects the fact that homogeneous image coordinates are defined
only up to a nonzero scale factor.

Substituting the plane-restricted camera-frame point from step (1) gives

$$
\lambda
\begin{bmatrix}
u \\ v \\ 1
\end{bmatrix}
=
K
\begin{bmatrix}
r_1 & r_2 & t_{cw}
\end{bmatrix}
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}.
$$

This equation shows that, for points lying on a plane, the complete mapping from
ground-plane coordinates $(x,y)$ to image pixels $(u,v)$ is linear in homogeneous
coordinates.


#### 4) Definition of the planar homography (why a 3×3 matrix is enough)

This means the entire mapping from plane coordinates to image coordinates can be represented by a single 3×3 matrix:

$$
H \triangleq K
\begin{bmatrix}
r_1 & r_2 & t_{cw}
\end{bmatrix}
\in \mathbb{R}^{3\times 3}.
$$

The matrix $H$ is called a **planar homography** (or projective transformation).
It is the *composition* of:
1) restricting the 3D world to the plane $Z=0$ (so only $(x,y)$ matter),
2) transforming that plane into the camera frame via $(R_{cw}, t_{cw})$,
3) mapping camera rays to pixels via $K$.

With this definition, the plane-to-image mapping becomes

$$
\begin{bmatrix}
u \\ v \\ 1
\end{bmatrix}
\sim
H
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}.
$$

The symbol $\sim$ means **equality up to a nonzero scale factor**:
there exists some $\lambda \neq 0$ such that
$\lambda \begin{bmatrix} u & v & 1 \end{bmatrix}^\top = H \begin{bmatrix} x & y & 1 \end{bmatrix}^\top$.

---

#### 5) Where the denominator comes from (homogeneous $\rightarrow$ Euclidean)

The homography does not directly output Euclidean pixel coordinates $(u,v)$.
Instead, it outputs a **homogeneous image vector**:

$$
\begin{bmatrix}
\tilde{u} \\ \tilde{v} \\ \tilde{w}
\end{bmatrix}
=
H
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}.
$$



$$
\begin{bmatrix}
\tilde{u} \\ \tilde{v} \\ \tilde{w}
\end{bmatrix}
=
\begin{bmatrix}
h_{11}x + h_{12}y + h_{13} \\
h_{21}x + h_{22}y + h_{23} \\
h_{31}x + h_{32}y + h_{33}
\end{bmatrix}.
$$

To recover Euclidean pixel coordinates, we must **dehomogenize**, i.e., divide by the last component:

$$
u = \frac{\tilde{u}}{\tilde{w}},
\qquad
v = \frac{\tilde{v}}{\tilde{w}}.
$$

This is exactly analogous to the perspective projection step
$(x_n,y_n) = (X_c/Z_c,\, Y_c/Z_c)$:
the division is what turns a linear homogeneous mapping into a nonlinear Euclidean mapping.

We define the projective denominator

$$
d(x,y) \triangleq \tilde{w} = h_{31} x + h_{32} y + h_{33}.
$$

Substituting into the dehomogenization formula yields the explicit nonlinear mapping

$$
u(x,y) =
\frac{h_{11} x + h_{12} y + h_{13}}{d(x,y)},
\qquad
v(x,y) =
\frac{h_{21} x + h_{22} y + h_{23}}{d(x,y)}.
$$

So the denominator is not an extra assumption: it is simply the last homogeneous coordinate
that must be divided out to obtain Euclidean pixels.

<!-- markdown-cell 03 -->
## How $K$, $(R_{cw},t_{cw})$, and $H$ are computed in this code

In this notebook the camera is **not calibrated from images**.  
Instead, the intrinsics and pose are **assumed** from the input parameters:

- `cam_pos`, `look_at`, `up_hint`
- `img_width`, `img_height`
- `fov_h_rad`

### A) Intrinsics $K$

The code assumes square pixels and no distortion, and computes

$$
f = \frac{W/2}{\tan(\mathrm{fov}_h/2)},\qquad
c_x = W/2,\qquad
c_y = H/2
$$

so

$$
K =
\begin{bmatrix}
f & 0 & c_x \\
0 & f & c_y \\
0 & 0 & 1
\end{bmatrix}.
$$

### B) Extrinsics $(R_{cw},t_{cw})$ from a “look‑at” construction

The rotation is built from the camera position and look‑at target:

$$
\mathbf{z}_{cam} = \frac{\text{look\_at} - \text{cam\_pos}}{\|\text{look\_at} - \text{cam\_pos}\|},
\qquad
\mathbf{x}_{cam} = \frac{\mathbf{z}_{cam} \times \text{up\_hint}}{\|\mathbf{z}_{cam} \times \text{up\_hint}\|},
\qquad
\mathbf{y}_{cam} = \mathbf{z}_{cam} \times \mathbf{x}_{cam}.
$$

Then

$$
R_{cw} =
\begin{bmatrix}
\mathbf{x}_{cam} \\
\mathbf{y}_{cam} \\
\mathbf{z}_{cam}
\end{bmatrix},
\qquad
t_{cw} = -R_{cw}\,\text{cam\_pos}.
$$

This matches the code in `PlanarCamera`.

### C) Homography $H$

For a ground plane with $Z=0$, only the first two columns of $R_{cw}$ matter:

$$
H = K \begin{bmatrix} r_1 & r_2 & t_{cw} \end{bmatrix}.
$$

This $H$ is used for the forward projection (world → pixels), and $H^{-1}$ for pixel → world.

---

<!-- markdown-cell 04 -->
## Why the Jacobian/Hessian depend on $Z_c$ (and what $Z_c$ is)

In the pinhole model, a 3D point in the **camera frame** is
$$\mathbf{p}_c = \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}.$$

The image projection is
$$u = f_x \frac{X_c}{Z_c} + c_x,\qquad v = f_y \frac{Y_c}{Z_c} + c_y.$$

### What is $Z_c$?
$Z_c$ is the **depth** of the point measured along the camera’s optical axis.
- Large $Z_c$ means the point is far from the camera.
- Small $Z_c$ means the point is close to the camera.

### Why the Jacobian depends on $Z_c$
Taking derivatives of the projection gives
$$
\frac{\partial u}{\partial X_c} = \frac{f_x}{Z_c},\qquad
\frac{\partial u}{\partial Z_c} = -\frac{f_x X_c}{Z_c^2},
$$
and similarly for $v$.
So the Jacobian contains terms proportional to $1/Z_c$ and $1/Z_c^2$.
This means **pixel sensitivity increases rapidly as $Z_c$ decreases**.

### Why the Hessian depends on $Z_c$
Second derivatives introduce one more power of $Z_c$ in the denominator,
so Hessian terms scale like
$$\frac{1}{Z_c^3}.$$
This is why curvature (nonlinearity) explodes near small depth.

### How $Z_c$ is determined here
In this notebook, we compute
$$
\mathbf{p}_c = R_{cw}\,\mathbf{p}_w + t_{cw},
$$
with $\mathbf{p}_w = [x,y,0]^T$ on the ground plane.  
Then $Z_c$ is simply the **third component** of $\mathbf{p}_c$.

So in our setup, $Z_c$ is fully determined by:
- the camera pose $(R_{cw},t_{cw})$,
- and the world point $(x,y,0)$ on the ground plane.

<!-- markdown-cell 05 -->
## 15° setup at 2m: rough-footprint $1/Z_c^3$ vs ambiguity

For shallow cameras the exact ground footprint can become numerically fragile.
This cell uses a **rough rectangular ROI** (not the exact projected polygon) around the region of interest in front of the camera,
then compares:
- $1/Z_c^3$ (curvature scaling proxy),
- ET2 ambiguity,
- and the $Z_c^3$-scaled ambiguity ($\mathrm{Amb}\cdot Z_c^3$).

---

## Source: `diffdrivenav-bev2image-efe .ipynb`

<!-- markdown-cell 01 -->
# Expected Free energy minimization for mobile robot navigation

<!-- markdown-cell 02 -->
### Joost Leliveld, last update: 02-03-2026  
Heavily inspired from Wouter Kouw

<!-- markdown-cell 03 -->
### Usecase:

A robot that moves according to the constant-velocity model is observed by an external static camera. The camera provides image measurements (pixel location) to help the robot navigate from point A to point B. The ground is assumed to be planar and the camera intrinsics as well as the homography known. This makes the observation model nonlinear and the effects of this will be studied for different control types.

<!-- markdown-cell 04 -->
### Dynamics (differential-drive / unicycle model)

Consider a robot that moves according to the unicycle model with linear and angular velocity inputs:

$$z_k = \begin{bmatrix} x_k \\ y_k \\ \theta_k \end{bmatrix},\qquad
 u_k = \begin{bmatrix} v_k \\ \omega_k \end{bmatrix}.$$

The discrete-time dynamics are

$$
\begin{aligned}
 x_k &= x_{k-1} + v_k\,\Delta t\,\cos(\theta_{k-1}) \\
 y_k &= y_{k-1} + v_k\,\Delta t\,\sin(\theta_{k-1}) \\
 \theta_k &= \theta_{k-1} + \omega_k\,\Delta t
\end{aligned}
\qquad +\; q_k\,.
$$

<!-- markdown-cell 05 -->
Process noise is white, $q_k \sim \mathcal{N}(0,Q)$.

For the notebook we use a simple diagonal model in $[x,y,\theta]$:

$$
Q =
\begin{bmatrix}
\sigma_x^2 & 0 & 0 \\
0 & \sigma_y^2 & 0 \\
0 & 0 & \sigma_\theta^2
\end{bmatrix}\, .
$$

These values are *placeholders* and should be replaced with estimates from simulation or real data.

<!-- markdown-cell 06 -->
### Observations: external oblique camera (pixels)

A static external camera observes the robot in the image plane. The measurement is the pixel location:

$$o_k \triangleq \begin{bmatrix} u_k^{\mathrm{pix}} \\ v_k^{\mathrm{pix}} \end{bmatrix} \in \mathbb{R}^2.$$

The camera measurement is generated by a nonlinear projective mapping:

$$o_k = g(z_k) + r_k,\qquad r_k \sim \mathcal{N}(0,R),$$

where $g(z_k)$ is defined as the composition of (i) ground-plane embedding into 3D, (ii) rigid transform into the camera frame, and (iii) perspective projection to pixels. Since the camera observes position on the ground plane, the observation model depends only on the position components. 

#### Projective mapping (state $\rightarrow$ pixels)
https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html 
1) **Embed the ground-plane point into 3D (world frame).**  
Assuming a planar ground ($Z=0$):

$$P_w(p_k) \triangleq \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
=
\begin{bmatrix} x_k \\ y_k \\ 0 \\ 1 \end{bmatrix}.$$

2) **Transform world coordinates to camera coordinates.**  
Let $(R_{cw},t_{cw})$ be the camera extrinsics (world-to-camera):

$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
=
R_{cw}\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} + t_{cw}.$$

3) **Perspective division (nonlinearity).**  
Normalized image coordinates are obtained by:

$$x_n = \frac{X_c}{Z_c},\qquad y_n = \frac{Y_c}{Z_c}.$$

4) **Intrinsics map normalized coordinates to pixels.**  
With camera intrinsics $(f_x,f_y,c_x,c_y)$:

$$u_k^{\mathrm{pix}} = f_x x_n + c_x,\qquad
v_k^{\mathrm{pix}} = f_y y_n + c_y.$$

Putting these steps together defines $g(z_k)$:
$$g(z_k) \triangleq 
\begin{bmatrix}
f_x \frac{X_c(p_k)}{Z_c(p_k)} + c_x \\
f_y \frac{Y_c(p_k)}{Z_c(p_k)} + c_y
\end{bmatrix},
\qquad p_k = (x_k, y_k).$$

The mapping is nonlinear because of the perspective division by $Z_c(p_k)$, which induces state-dependent sensitivity of pixels to changes in ground-plane position.

<!-- markdown-cell 07 -->
### Probabilistic model

We use a probabilistic state-space model with nonlinear unicycle dynamics and a nonlinear camera observation model:

$$\begin{align}
p(z_0) &= \mathcal{N}(z_0 \mid m_0, S_0) \\
p(z_k \mid z_{k-1}, u_k) &= \mathcal{N}(z_k \mid f(z_{k-1}, u_k), Q) \\
p(o_k \mid z_k) &= \mathcal{N}(o_k \mid g(z_k), R)\, .
\end{align}$$

Here, $z_k \in \mathbb{R}^3$ is the ground-plane unicycle state and $o_k$ is the external camera measurement in the image plane.

<!-- markdown-cell 08 -->
### Gaussian approximation to likelihood

Because $g(\cdot)$ is nonlinear, the likelihood $p(o_k \mid z_k)$ is generally non-Gaussian as a function of $z_k$:

Using the projective observation model,

$$
g(z_k)
=
\begin{bmatrix}
f_x \dfrac{X_c(p_k)}{Z_c(p_k)} + c_x \\
f_y \dfrac{Y_c(p_k)}{Z_c(p_k)} + c_y
\end{bmatrix},
$$

the likelihood can be written (up to a normalization constant) as

$$
p(o_k \mid z_k)
=
\frac{1}{\sqrt{(2\pi)^2 \det R}}
\exp\!\Big(-\tfrac12 (o_k-g(z_k))^\top R^{-1}(o_k-g(z_k))\Big).
$$

$$
\log p(o_k \mid z_k)
=
-\tfrac12 \big(o_k - g(z_k)\big)^\top R^{-1} \big(o_k - g(z_k)\big).
$$

Since the term is not just a power of $o$ or $z$, the likelihood is non-Gaussian. We approximate the joint $(z_k, o_k)$ as a Gaussian:

$$p(o_k, z_k) \approx \mathcal{N}\!\left(
\begin{bmatrix} z_k \\ o_k \end{bmatrix}
\Bigg|\,
\begin{bmatrix} m_k \\ \mu_k \end{bmatrix},
\begin{bmatrix} S_k & \Gamma_k \\ \Gamma_k^{\top} & \Sigma_k \end{bmatrix}
\right)\, .$$

Here:
- $m_k, S_k$ are the predicted state mean and covariance,
- $\mu_k, \Sigma_k$ are the predicted observation mean and covariance,
- $\Gamma_k$ is the state--observation cross-covariance.

Because this joint is Gaussian, conditioning yields an approximate Gaussian likelihood:

$$\begin{align}
p(o_k \mid z_k)
&\approx \mathcal{N}\!\left(
o_k \,\Big|\,
\mu_k + \Gamma_k^{\top} S_k^{-1}(z_k - m_k),
\ \Sigma_k - \Gamma_k^{\top} S_k^{-1}\Gamma_k
\right)\, .
\end{align}$$

The parameters $(\mu_k, \Sigma_k, \Gamma_k)$ are obtained by a Gaussian approximation method:
first-order Taylor (ET1/EFE1), second-order Taylor (ET2/EFE2), or sigma-point (UT) approximation.

<!-- markdown-cell 09 -->
## Posterior predictive

The posterior predictive distribution of the observation is obtained by marginalizing
over the latent state:

$$
\begin{align}
p(o_k \mid o_{1:k-1})
&= \int p(o_k \mid z_k)\, p(z_k \mid o_{1:k-1}) \, dz_k \tag{1}\\
&= \int \mathcal N\!\left(
\begin{bmatrix} z_k \\ o_k \end{bmatrix}
\Bigg|\,
\begin{bmatrix} m_k \\ \mu_k \end{bmatrix},
\begin{bmatrix}
S_k & \Gamma_k \\
\Gamma_k^\top & \Sigma_k
\end{bmatrix}
\right) dz_k \tag{2}\\
&= \mathcal N(o_k \mid \mu_k, \Sigma_k). \tag{3}
\end{align}
$$

That is, marginalizing a joint Gaussian yields a Gaussian predictive distribution
with mean $\mu_k$ and covariance $\Sigma_k$.

<!-- markdown-cell 10 -->
## Planning and control

### Expected Free Energy (EFE) objective

The Expected Free Energy objective for a single time step is given by

$$
\mathcal J_k(u_t)
=
\mathcal C
+
\frac{1}{2}\,\mathrm{tr}\!\left(S_*^{-1}(\Sigma_t + \Sigma_\star)\right)
+
\frac{1}{2}\,\ln
\frac{\left| \Sigma_t - \Gamma_t^\top S_t^{-1} \Gamma_t \right|}
{\left| \Sigma_t \right|}. \tag{1}
$$

The first term encodes the expected risk, while the log-determinant term captures
expected ambiguity through the conditional observation covariance.

### Horizon-based control optimization

For a planning horizon of $T$ steps, the optimal control sequence is obtained as

$$
\begin{align}
\hat u
&= \arg\max_{u \in \mathcal U} q^\star(u) \tag{1}\\
&= \arg\min_{u \in \mathcal U}
\sum_{t=1}^{T} \mathcal J_t(u_t)
- \ln p(u_t). \tag{2}
\end{align}
$$

<!-- markdown-cell 11 -->
## Model: EFE(ET2)

<!-- markdown-cell 12 -->
## GP Visibility Map + GP-aware EFE Experiment

This section adds a GP visibility model and runs ET1 vs ET2 experiments with visibility-aware planning/correction.
It is self-contained within this notebook and does not depend on the occlusion notebook.

---

## Source: `botnav-bev2image-efe.ipynb`

<!-- markdown-cell 01 -->
# Expected Free energy minimization for mobile robot navigation

<!-- markdown-cell 02 -->
### Joost Leliveld, last update: 06-02-2026  
Heavily inspired from Wouter Kouw

<!-- markdown-cell 03 -->
### Usecase:

A robot that moves according to the constant-velocity model is observed by an external static camera. The camera provides image measurements (pixel location) to help the robot navigate from point A to point B. The ground is assumed to be planar and the camera intrinsics as well as the homography known. This makes the observation model nonlinear and the effects of this will be studied for different control types.

<!-- markdown-cell 04 -->
### Dynamics (linear constant-velocity model)

Consider a robot that moves according to the constant-velocity model:

$$\underbrace{\begin{bmatrix} x_{1,k} \\ x_{2,k} \\ \dot{x}_{1,k} \\ \dot{x}_{2,k} \end{bmatrix}}_{z_k}
=
\underbrace{\begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}}_{A}
\underbrace{\begin{bmatrix} x_{1,k-1} \\ x_{2,k-1} \\ \dot{x}_{1,k-1} \\ \dot{x}_{2,k-1} \end{bmatrix}}_{z_{k-1}}
+
\underbrace{\begin{bmatrix}
0 & 0 \\
0 & 0 \\
\Delta t & 0 \\
0 & \Delta t
\end{bmatrix}}_{B}
\underbrace{\begin{bmatrix} u_{1,k} \\ u_{2,k} \end{bmatrix}}_{u_k}
+ q_k\, .$$

<!-- markdown-cell 05 -->
Process noise is white, $q_k \sim \mathcal{N}(0,Q)$, with

$$Q =
\begin{bmatrix}
\frac{\Delta t^3}{3}\rho_1 & 0 & \frac{\Delta t^2}{2}\rho_1 & 0 \\
0 & \frac{\Delta t^3}{3}\rho_2 & 0 & \frac{\Delta t^2}{2}\rho_2 \\
\frac{\Delta t^2}{2}\rho_1 & 0 & \Delta t\rho_1 & 0 \\
0 & \frac{\Delta t^2}{2}\rho_2 & 0 & \Delta t\rho_2
\end{bmatrix}\, .$$

<!-- markdown-cell 06 -->
### Observations: external oblique camera (pixels)

A static external camera observes the robot in the image plane. The measurement is the pixel location:

$$o_k \triangleq \begin{bmatrix} u_k^{\mathrm{pix}} \\ v_k^{\mathrm{pix}} \end{bmatrix} \in \mathbb{R}^2.$$

The camera measurement is generated by a nonlinear projective mapping:

$$o_k = g(z_k) + r_k,\qquad r_k \sim \mathcal{N}(0,R),$$

where $g(z_k)$ is defined as the composition of (i) ground-plane embedding into 3D, (ii) rigid transform into the camera frame, and (iii) perspective projection to pixels. Since the camera observes position on the ground plane, the observation model depends only on the position components. 

#### Projective mapping (state $\rightarrow$ pixels)
https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html 
1) **Embed the ground-plane point into 3D (world frame).**  
Assuming a planar ground ($Z=0$):

$$P_w(p_k) \triangleq \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
=
\begin{bmatrix} x_{1,k} \\ x_{2,k} \\ 0 \\ 1 \end{bmatrix}.$$

2) **Transform world coordinates to camera coordinates.**  
Let $(R_{cw},t_{cw})$ be the camera extrinsics (world-to-camera):

$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
=
R_{cw}\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} + t_{cw}.$$

3) **Perspective division (nonlinearity).**  
Normalized image coordinates are obtained by:

$$x_n = \frac{X_c}{Z_c},\qquad y_n = \frac{Y_c}{Z_c}.$$

4) **Intrinsics map normalized coordinates to pixels.**  
With camera intrinsics $(f_x,f_y,c_x,c_y)$:

$$u_k^{\mathrm{pix}} = f_x x_n + c_x,\qquad
v_k^{\mathrm{pix}} = f_y y_n + c_y.$$

Putting these steps together defines $g(z_k)$:

$$g(z_k) \triangleq 
\begin{bmatrix}
f_x \frac{X_c(p_k)}{Z_c(p_k)} + c_x \\
f_y \frac{Y_c(p_k)}{Z_c(p_k)} + c_y
\end{bmatrix},
\qquad p_k = C z_k.$$

The mapping is nonlinear because of the perspective division by $Z_c(p_k)$, which induces state-dependent sensitivity of pixels to changes in ground-plane position.

<!-- markdown-cell 07 -->
### Probabilistic model

We use a probabilistic state-space model with linear-Gaussian dynamics and a nonlinear camera observation model:

$$\begin{align}
p(z_0) &= \mathcal{N}(z_0 \mid m_0, S_0) \\
p(z_k \mid z_{k-1}, u_k) &= \mathcal{N}(z_k \mid A z_{k-1} + B u_k, Q) \\
p(o_k \mid z_k) &= \mathcal{N}(o_k \mid g(z_k), R)\, .
\end{align}$$

Here, $z_k \in \mathbb{R}^4$ is the ground-plane constant-velocity state and $o_k$ is the external camera measurement in the image plane.

<!-- markdown-cell 08 -->
### Gaussian approximation to likelihood

Because $g(\cdot)$ is nonlinear, the likelihood $p(o_k \mid z_k)$ is generally non-Gaussian as a function of $z_k$:

Using the projective observation model,

$$
g(z_k)
=
\begin{bmatrix}
f_x \dfrac{X_c(p_k)}{Z_c(p_k)} + c_x \\
f_y \dfrac{Y_c(p_k)}{Z_c(p_k)} + c_y
\end{bmatrix},
$$

the likelihood can be written (up to a normalization constant) as

$$
p(o_k \mid z_k)
=
\frac{1}{\sqrt{(2\pi)^2 \det R}}
\exp\!\Big(-\tfrac12 (o_k-g(z_k))^\top R^{-1}(o_k-g(z_k))\Big).
$$


$$
\log p(o_k \mid z_k)
=
-\tfrac12 \big(o_k - g(z_k)\big)^\top R^{-1} \big(o_k - g(z_k)\big).
$$

Since the term is not just a power of o or z the likelihood is non-Gaussian, however we approximate the joint $(z_k, o_k)$ as a Gaussian:

$$p(o_k, z_k) \approx \mathcal{N}\!\left(
\begin{bmatrix} z_k \\ o_k \end{bmatrix}
\Bigg|\,
\begin{bmatrix} m_k \\ \mu_k \end{bmatrix},
\begin{bmatrix} S_k & \Gamma_k \\ \Gamma_k^{\top} & \Sigma_k \end{bmatrix}
\right)\, .$$

Here:
- $m_k, S_k$ are the predicted state mean and covariance,
- $\mu_k, \Sigma_k$ are the predicted observation mean and covariance,
- $\Gamma_k$ is the state--observation cross-covariance.

Because this joint is Gaussian, conditioning yields an approximate Gaussian likelihood:

$$\begin{align}
p(o_k \mid z_k)
&\approx \mathcal{N}\!\left(
o_k \,\Big|\,
\mu_k + \Gamma_k^{\top} S_k^{-1}(z_k - m_k),
\ \Sigma_k - \Gamma_k^{\top} S_k^{-1}\Gamma_k
\right)\, .
\end{align}$$

The parameters $(\mu_k, \Sigma_k, \Gamma_k)$ are obtained by a Gaussian approximation method:
first-order Taylor (ET1/EFE1), second-order Taylor (ET2/EFE2), or sigma-point (UT) approximation.

<!-- markdown-cell 09 -->
## Posterior predictive

The posterior predictive distribution of the observation is obtained by marginalizing
over the latent state:

$$
\begin{align}
p(o_k \mid o_{1:k-1})
&= \int p(o_k \mid z_k)\, p(z_k \mid o_{1:k-1}) \, dz_k \tag{1}\\
&= \int \mathcal N\!\left(
\begin{bmatrix} z_k \\ o_k \end{bmatrix}
\Bigg|\,
\begin{bmatrix} m_k \\ \mu_k \end{bmatrix},
\begin{bmatrix}
S_k & \Gamma_k \\
\Gamma_k^\top & \Sigma_k
\end{bmatrix}
\right) dz_k \tag{2}\\
&= \mathcal N(o_k \mid \mu_k, \Sigma_k). \tag{3}
\end{align}
$$

That is, marginalizing a joint Gaussian yields a Gaussian predictive distribution
with mean $\mu_k$ and covariance $\Sigma_k$.

<!-- markdown-cell 10 -->
## Planning and control

### Expected Free Energy (EFE) objective

The Expected Free Energy objective for a single time step is given by

$$
\mathcal J_k(u_t)
=
\mathcal C
+
\frac{1}{2}\,\mathrm{tr}\!\left(S_*^{-1}(\Sigma_t + \Sigma_\star)\right)
+
\frac{1}{2}\,\ln
\frac{\left| \Sigma_t - \Gamma_t^\top S_t^{-1} \Gamma_t \right|}
{\left| \Sigma_t \right|}. \tag{1}
$$

The first term encodes the expected risk, while the log-determinant term captures
expected ambiguity through the conditional observation covariance.

### Horizon-based control optimization

For a planning horizon of $T$ steps, the optimal control sequence is obtained as

$$
\begin{align}
\hat u
&= \arg\max_{u \in \mathcal U} q^\star(u) \tag{1}\\
&= \arg\min_{u \in \mathcal U}
\sum_{t=1}^{T} \mathcal J_t(u_t)
- \ln p(u_t). \tag{2}
\end{align}
$$

<!-- markdown-cell 11 -->
## Model: EFE(ET2)

---

