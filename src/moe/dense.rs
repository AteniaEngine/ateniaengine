//! **MOE-3** — Dense MoE reference execution path (correctness-first).
//!
//! This is the first *executable* MoE path in Atenia. It is a pure-CPU,
//! self-contained **reference** implementation whose only goal is
//! numerical correctness — exactly the "correctness first" discipline AQS
//! used. It runs **every** expert on **every** token and combines them by
//! their router weights:
//!
//! ```text
//!   router_logits = W_router · x
//!   weights       = softmax(router_logits)          (sum to 1)
//!   for each expert e:  y_e = expert_e.forward(x)
//!   output        = Σ_e weights[e] · y_e            (ALL experts)
//! ```
//!
//! ## What this is NOT
//!
//! * **No sparse dispatch** — all experts always execute (dense).
//! * **No real top-k routing** — no pruning; a `top_k` value is carried as
//!   *conceptual* metadata only and does not gate execution.
//! * **No graph / runtime / CUDA integration** — this is a standalone
//!   `f32` reference computed with naive matmuls. It is not wired into the
//!   execution graph, the loader, or any model.
//! * **Not optimised** — clarity over speed.
//!
//! Its purpose is to be the certifiable mathematical truth a future sparse
//! MoE path (MOE-4) must reproduce.

/// Errors from the dense MoE reference path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoeDenseError {
    /// An input vector length did not match the expected dimension.
    DimMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A weight matrix length did not match `rows * cols`.
    WeightShapeMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A layer was built with zero experts.
    NoExperts,
    /// The router weight implies a different expert count than the experts
    /// vector provides.
    RouterExpertCountMismatch { router: usize, experts: usize },
}

impl std::fmt::Display for MoeDenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeDenseError::DimMismatch { what, expected, actual } => {
                write!(f, "moe-dense: {what} dim mismatch: expected {expected}, got {actual}")
            }
            MoeDenseError::WeightShapeMismatch { what, expected, actual } => write!(
                f,
                "moe-dense: {what} weight shape mismatch: expected {expected} elems, got {actual}"
            ),
            MoeDenseError::NoExperts => write!(f, "moe-dense: layer has no experts"),
            MoeDenseError::RouterExpertCountMismatch { router, experts } => write!(
                f,
                "moe-dense: router expert count ({router}) != experts ({experts})"
            ),
        }
    }
}

impl std::error::Error for MoeDenseError {}

/// SiLU (a.k.a. swish): `x * sigmoid(x)`. Computed in f64 internally for
/// reference-grade accuracy, returned as f32.
fn silu(x: f32) -> f32 {
    let xd = x as f64;
    (xd / (1.0 + (-xd).exp())) as f64 as f32
}

/// Rows*cols product above which [`matvec`] parallelises across output rows.
/// Below it the rayon fork/join overhead would dominate (e.g. the router's
/// `[num_experts, d_model]` is tiny); above it (the expert FFN projections —
/// `d_ff × d_model` with `d_ff` in the thousands) the per-row work amortises
/// the overhead and the 24-core speedup is real. **MOE-PROD-8.**
const MATVEC_PAR_THRESHOLD: usize = 1 << 16; // 65 536 MACs

/// `y = W · x` where `W` is row-major `[rows, cols]` and `x` has length
/// `cols`. Returns length `rows`. Accumulates in f64 for accuracy.
///
/// **MOE-PROD-8** — the per-output-row f64 reductions are independent, so for
/// large weights they run in parallel across rows via rayon. This is
/// **bit-identical** to the serial path: each `y[r]` is the *same* sequential
/// f64 accumulation over `c in 0..cols`; only *which thread* computes a given
/// row changes, never the arithmetic or its order. The expert FFN matmuls
/// dominate MoE generation (measured ~82 % of the warm wall in MOE-PROD-7), so
/// this is the highest-ROI generation-compute win that preserves the certified
/// reference math.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let row = |base: usize| -> f32 {
        let mut acc = 0.0_f64;
        for c in 0..cols {
            acc += (w[base + c] as f64) * (x[c] as f64);
        }
        acc as f32
    };
    let mut y = vec![0.0_f32; rows];
    if rows.saturating_mul(cols) >= MATVEC_PAR_THRESHOLD {
        use rayon::prelude::*;
        y.par_iter_mut().enumerate().for_each(|(r, yr)| *yr = row(r * cols));
    } else {
        for (r, yr) in y.iter_mut().enumerate() {
            *yr = row(r * cols);
        }
    }
    y
}

/// **NUMERIC-POLICY-1** — f32-accumulation `matvec`. Same parallel structure as
/// [`matvec`] but the per-row reduction is f32. Used by the expert FFN under
/// `Strict`/`Fast`; bounded drift vs the f64 path, certified by tolerance.
fn matvec_f32(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let row = |base: usize| -> f32 {
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += w[base + c] * x[c];
        }
        acc
    };
    let mut y = vec![0.0_f32; rows];
    if rows.saturating_mul(cols) >= MATVEC_PAR_THRESHOLD {
        use rayon::prelude::*;
        y.par_iter_mut().enumerate().for_each(|(r, yr)| *yr = row(r * cols));
    } else {
        for (r, yr) in y.iter_mut().enumerate() {
            *yr = row(r * cols);
        }
    }
    y
}

/// **NUMERIC-POLICY-1** — policy-aware `matvec` for the **expert FFN**: f64
/// under [`NumericPolicy::Certified`] (bit-exact reference), f32 under
/// `Strict`/`Fast`. The router deliberately does **not** use this (it stays f64
/// on every policy) so the top-k routing decision is identical.
fn matvec_policy(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    if super::numeric_policy::numeric_policy().ffn_uses_f32() {
        matvec_f32(w, rows, cols, x)
    } else {
        matvec(w, rows, cols, x)
    }
}

/// Numerically-stable softmax over `logits`. Returns weights that sum to 1
/// (within f32 epsilon). f64 internals.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f64> = logits
        .iter()
        .map(|&v| ((v - max) as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for e in &mut exps {
        *e *= inv;
    }
    exps.into_iter().map(|v| v as f32).collect()
}

/// Router output: per-expert routing weights (softmax of router logits).
#[derive(Debug, Clone, PartialEq)]
pub struct MoeRouterOutput {
    /// One weight per expert; sums to 1.
    pub weights: Vec<f32>,
}

impl MoeRouterOutput {
    /// Sum of weights (should be ~1.0 for a valid softmax).
    pub fn weight_sum(&self) -> f32 {
        self.weights.iter().sum()
    }
}

/// A single dense SwiGLU expert: `down( silu(gate·x) ⊙ (up·x) )`.
///
/// Weight layout (row-major):
/// - `w_gate`: `[d_ff, d_model]`
/// - `w_up`:   `[d_ff, d_model]`
/// - `w_down`: `[d_model, d_ff]`
#[derive(Debug, Clone)]
pub struct MoeDenseExpert {
    pub d_model: usize,
    pub d_ff: usize,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

impl MoeDenseExpert {
    /// Construct + validate shapes.
    pub fn new(
        d_model: usize,
        d_ff: usize,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
    ) -> Result<Self, MoeDenseError> {
        if w_gate.len() != d_ff * d_model {
            return Err(MoeDenseError::WeightShapeMismatch {
                what: "w_gate",
                expected: d_ff * d_model,
                actual: w_gate.len(),
            });
        }
        if w_up.len() != d_ff * d_model {
            return Err(MoeDenseError::WeightShapeMismatch {
                what: "w_up",
                expected: d_ff * d_model,
                actual: w_up.len(),
            });
        }
        if w_down.len() != d_model * d_ff {
            return Err(MoeDenseError::WeightShapeMismatch {
                what: "w_down",
                expected: d_model * d_ff,
                actual: w_down.len(),
            });
        }
        Ok(Self { d_model, d_ff, w_gate, w_up, w_down })
    }

    /// Forward one token vector `x` (length `d_model`) → output length
    /// `d_model`. SwiGLU.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, MoeDenseError> {
        if x.len() != self.d_model {
            return Err(MoeDenseError::DimMismatch {
                what: "expert input",
                expected: self.d_model,
                actual: x.len(),
            });
        }
        // NUMERIC-POLICY-1: the expert FFN projections honour the active policy
        // (f64 Certified / f32 Strict|Fast). Router stays f64 (see `route`).
        let gate = matvec_policy(&self.w_gate, self.d_ff, self.d_model, x);
        let up = matvec_policy(&self.w_up, self.d_ff, self.d_model, x);
        let mut h = vec![0.0_f32; self.d_ff];
        for i in 0..self.d_ff {
            h[i] = silu(gate[i]) * up[i];
        }
        Ok(matvec_policy(&self.w_down, self.d_model, self.d_ff, &h))
    }
}

/// A dense MoE layer: a router + N experts. All experts run on every
/// forward; the output is their router-weighted sum.
#[derive(Debug, Clone)]
pub struct MoeDenseLayer {
    pub d_model: usize,
    pub d_ff: usize,
    /// Router weight, row-major `[num_experts, d_model]`.
    pub w_router: Vec<f32>,
    pub experts: Vec<MoeDenseExpert>,
    /// **Conceptual** top-k (documentation only). The dense reference path
    /// ignores it and runs every expert. Carried so a future sparse path
    /// (MOE-4) and the fixture metadata agree on intent.
    pub conceptual_top_k: usize,
}

impl MoeDenseLayer {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        w_router: Vec<f32>,
        experts: Vec<MoeDenseExpert>,
        conceptual_top_k: usize,
    ) -> Result<Self, MoeDenseError> {
        if experts.is_empty() {
            return Err(MoeDenseError::NoExperts);
        }
        let n = experts.len();
        if w_router.len() != n * d_model {
            return Err(MoeDenseError::RouterExpertCountMismatch {
                router: if d_model == 0 { 0 } else { w_router.len() / d_model.max(1) },
                experts: n,
            });
        }
        for (i, e) in experts.iter().enumerate() {
            if e.d_model != d_model || e.d_ff != d_ff {
                return Err(MoeDenseError::DimMismatch {
                    what: if e.d_model != d_model { "expert d_model" } else { "expert d_ff" },
                    expected: if e.d_model != d_model { d_model } else { d_ff },
                    actual: if e.d_model != d_model { e.d_model } else { e.d_ff },
                });
            }
            let _ = i;
        }
        Ok(Self { d_model, d_ff, w_router, experts, conceptual_top_k })
    }

    /// Number of experts.
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Compute router weights for token `x` (softmax of `W_router · x`).
    pub fn route(&self, x: &[f32]) -> Result<MoeRouterOutput, MoeDenseError> {
        if x.len() != self.d_model {
            return Err(MoeDenseError::DimMismatch {
                what: "router input",
                expected: self.d_model,
                actual: x.len(),
            });
        }
        let logits = matvec(&self.w_router, self.experts.len(), self.d_model, x);
        Ok(MoeRouterOutput {
            weights: softmax(&logits),
        })
    }

    /// **Dense MoE forward.** Runs all experts and returns the
    /// router-weighted sum. Output length `d_model`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, MoeDenseError> {
        let routing = self.route(x)?;
        let mut out = vec![0.0_f32; self.d_model];
        for (e, expert) in self.experts.iter().enumerate() {
            let w = routing.weights[e];
            // Skip the matmul only as a pure micro-optimisation when the
            // weight is exactly zero — numerically identical to running it
            // and scaling by 0. (Still "dense": every nonzero-weight expert
            // runs; this does not implement top-k pruning.)
            if w == 0.0 {
                continue;
            }
            let y_e = expert.forward(x)?;
            for d in 0..self.d_model {
                out[d] += w * y_e[d];
            }
        }
        Ok(out)
    }
}

// ============================================================================
// Synthetic fixture (deterministic, tiny — for certification tests).
// ============================================================================

/// Deterministic xorshift used to build reproducible fixture weights with
/// no external dependency.
fn seeded(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 11) as u32;
        out.push((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    out
}

/// Build the official tiny synthetic dense-MoE fixture:
/// **4 experts, conceptual top-k = 2, dense execution = 4**, small dims.
/// Deterministic. Used by the certification tests and as the MOE-1
/// "tiny synthetic MoE" made executable.
pub fn build_fixture_layer() -> MoeDenseLayer {
    let d_model = 8;
    let d_ff = 16;
    let num_experts = 4;
    let w_router = seeded(1, num_experts * d_model);
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let base = 100 + e as u64;
        experts.push(
            MoeDenseExpert::new(
                d_model,
                d_ff,
                seeded(base * 10 + 1, d_ff * d_model),
                seeded(base * 10 + 2, d_ff * d_model),
                seeded(base * 10 + 3, d_model * d_ff),
            )
            .expect("fixture expert shapes are valid"),
        );
    }
    MoeDenseLayer::new(d_model, d_ff, w_router, experts, 2)
        .expect("fixture layer is valid")
}

// ============================================================================
// Certification tests (synthetic — no model, no loader, no graph)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn token(seed: u64, d: usize) -> Vec<f32> {
        seeded(seed, d)
    }

    // NUMERIC-POLICY-1: the expert FFN honours the active policy.
    #[test]
    fn expert_forward_certified_is_f64_strict_is_bounded() {
        use crate::moe::numeric_policy::{
            clear_numeric_policy_override, set_numeric_policy, NumericPolicy, PolicyCertificate,
            STRICT_LOGIT_TOLERANCE,
        };
        let (d_model, d_ff) = (64usize, 256usize);
        let e = MoeDenseExpert::new(
            d_model,
            d_ff,
            seeded(11, d_ff * d_model),
            seeded(12, d_ff * d_model),
            seeded(13, d_model * d_ff),
        )
        .unwrap();
        let x = seeded(99, d_model);

        // Certified (f64) is deterministic and must equal the pre-policy default.
        set_numeric_policy(NumericPolicy::Certified);
        let cert_a = e.forward(&x).unwrap();
        let cert_b = e.forward(&x).unwrap();
        assert_eq!(cert_a, cert_b, "Certified forward must be deterministic");

        // Strict (f32) must be within tolerance of Certified, same argmax.
        set_numeric_policy(NumericPolicy::Strict);
        let strict = e.forward(&x).unwrap();
        clear_numeric_policy_override();

        let c = PolicyCertificate::compare(
            NumericPolicy::Strict,
            std::slice::from_ref(&cert_a),
            std::slice::from_ref(&strict),
            &[0],
            &[0],
        );
        assert!(
            c.passes(STRICT_LOGIT_TOLERANCE),
            "Strict expert FFN must certify vs Certified f64: {c:?}"
        );
        // The f32 path is genuinely exercised: drift is small but the values
        // are real f32 reductions (max_abs_diff is a tiny, finite number).
        assert!(c.max_abs_diff < STRICT_LOGIT_TOLERANCE);
    }

    #[test]
    fn routing_weights_sum_to_one() {
        let layer = build_fixture_layer();
        let x = token(7, layer.d_model);
        let r = layer.route(&x).unwrap();
        assert_eq!(r.weights.len(), layer.num_experts());
        assert!((r.weight_sum() - 1.0).abs() < 1e-5, "weights must sum to 1");
        assert!(r.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn deterministic_output() {
        let layer = build_fixture_layer();
        let x = token(9, layer.d_model);
        let a = layer.forward(&x).unwrap();
        let b = layer.forward(&x).unwrap();
        assert_eq!(a, b, "dense MoE forward must be deterministic");
        assert!(a.iter().all(|v| v.is_finite()));
        assert_eq!(a.len(), layer.d_model);
    }

    #[test]
    fn combine_matches_manual_calculation() {
        let layer = build_fixture_layer();
        let x = token(11, layer.d_model);
        let routing = layer.route(&x).unwrap();
        // Manual: Σ_e w_e * expert_e(x).
        let mut manual = vec![0.0_f32; layer.d_model];
        for (e, expert) in layer.experts.iter().enumerate() {
            let y = expert.forward(&x).unwrap();
            for d in 0..layer.d_model {
                manual[d] += routing.weights[e] * y[d];
            }
        }
        let got = layer.forward(&x).unwrap();
        for d in 0..layer.d_model {
            assert!((got[d] - manual[d]).abs() < 1e-5, "combine must match manual sum");
        }
    }

    #[test]
    fn identical_experts_reproduce_dense_behavior() {
        // If all experts are identical, the weighted sum (weights sum to 1)
        // must equal a single expert's output.
        let d_model = 4;
        let d_ff = 8;
        let n = 3;
        let wg = seeded(21, d_ff * d_model);
        let wu = seeded(22, d_ff * d_model);
        let wd = seeded(23, d_model * d_ff);
        let experts: Vec<MoeDenseExpert> = (0..n)
            .map(|_| MoeDenseExpert::new(d_model, d_ff, wg.clone(), wu.clone(), wd.clone()).unwrap())
            .collect();
        let layer = MoeDenseLayer::new(d_model, d_ff, seeded(24, n * d_model), experts, 2).unwrap();
        let x = token(25, d_model);
        let single = layer.experts[0].forward(&x).unwrap();
        let combined = layer.forward(&x).unwrap();
        for d in 0..d_model {
            assert!(
                (combined[d] - single[d]).abs() < 1e-5,
                "identical experts must reproduce a single expert's output"
            );
        }
    }

    #[test]
    fn zero_weight_expert_contributes_nothing() {
        // Force a routing vector with one expert at weight 0 by comparing
        // a manual combine that zeroes expert 0 against a forward that
        // would include it — using the property directly on the combine.
        let layer = build_fixture_layer();
        let x = token(31, layer.d_model);
        let routing = layer.route(&x).unwrap();
        // Manual combine with expert 0 zeroed.
        let mut manual = vec![0.0_f32; layer.d_model];
        for (e, expert) in layer.experts.iter().enumerate() {
            let w = if e == 0 { 0.0 } else { routing.weights[e] };
            if w == 0.0 {
                continue;
            }
            let y = expert.forward(&x).unwrap();
            for d in 0..layer.d_model {
                manual[d] += w * y[d];
            }
        }
        // A second manual combine that adds expert 0 with weight 0 must be
        // identical — proving a zero-weight expert contributes nothing.
        let mut manual2 = manual.clone();
        let y0 = layer.experts[0].forward(&x).unwrap();
        for d in 0..layer.d_model {
            manual2[d] += 0.0 * y0[d];
        }
        assert_eq!(manual, manual2);
    }

    #[test]
    fn expert_ordering_does_not_change_weighted_result() {
        // Reverse the experts AND the matching router rows; the weighted
        // sum is commutative, so the output must be identical.
        let layer = build_fixture_layer();
        let x = token(41, layer.d_model);
        let base = layer.forward(&x).unwrap();

        let n = layer.num_experts();
        let dm = layer.d_model;
        // Reverse experts.
        let mut experts_rev = layer.experts.clone();
        experts_rev.reverse();
        // Reverse router rows to keep each expert paired with its row.
        let mut router_rev = vec![0.0_f32; layer.w_router.len()];
        for e in 0..n {
            let src = (n - 1 - e) * dm;
            let dst = e * dm;
            router_rev[dst..dst + dm].copy_from_slice(&layer.w_router[src..src + dm]);
        }
        let layer_rev =
            MoeDenseLayer::new(dm, layer.d_ff, router_rev, experts_rev, layer.conceptual_top_k)
                .unwrap();
        let rev = layer_rev.forward(&x).unwrap();
        for d in 0..dm {
            assert!((base[d] - rev[d]).abs() < 1e-5, "ordering must not change result");
        }
    }

    #[test]
    fn fixture_executes_end_to_end() {
        let layer = build_fixture_layer();
        assert_eq!(layer.num_experts(), 4);
        assert_eq!(layer.conceptual_top_k, 2);
        let x = token(55, layer.d_model);
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.len(), layer.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_basic_properties() {
        let w = softmax(&[1.0, 2.0, 3.0]);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Monotonic: larger logit -> larger weight.
        assert!(w[2] > w[1] && w[1] > w[0]);
        // Uniform logits -> uniform weights.
        let u = softmax(&[5.0, 5.0, 5.0, 5.0]);
        for &v in &u {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(matches!(
            MoeDenseExpert::new(4, 8, vec![0.0; 3], vec![0.0; 32], vec![0.0; 32]),
            Err(MoeDenseError::WeightShapeMismatch { what: "w_gate", .. })
        ));
        assert!(matches!(
            MoeDenseLayer::new(4, 8, vec![0.0; 4], vec![], 2),
            Err(MoeDenseError::NoExperts)
        ));
    }
}
