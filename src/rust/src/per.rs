/*!
 * per.rs — Prioritized Experience Replay (PER) buffer.
 *
 * Reference:
 *   Schaul et al., "Prioritized Experience Replay" (2016)
 *   https://arxiv.org/abs/1511.05952
 *
 * Key ideas:
 *   - Transitions are sampled with probability proportional to |TD error|^α
 *   - Importance-sampling weights w_i = (1/N * 1/P(i))^β correct for bias
 *   - β is annealed from β_start → 1.0 over training
 *   - Priorities are stored in a SumTree for O(log N) ops
 *
 * Python usage:
 *   buf = mujuco_replay.PERBuffer(
 *       capacity=1_000_000, obs_dim=11, act_dim=3,
 *       alpha=0.6, beta_start=0.4, beta_steps=1_000_000,
 *   )
 *   buf.add(obs, action, reward, next_obs, done)
 *   batch = buf.sample(256)
 *   # batch["weights"] contains IS weights for loss scaling
 *   buf.update_priorities(batch["indices"], new_td_errors)
 */

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use rand::prelude::*;
use rand::rngs::SmallRng;

use crate::sum_tree::SumTree;

/// Prioritized Experience Replay buffer backed by a SumTree.
#[pyclass]
pub struct PERBuffer {
    // ── Dimensions ───────────────────────────────────────────────────────
    capacity: usize,
    obs_dim:  usize,
    act_dim:  usize,

    // ── Storage ───────────────────────────────────────────────────────────
    obs:      Vec<f32>,
    actions:  Vec<f32>,
    rewards:  Vec<f32>,
    next_obs: Vec<f32>,
    dones:    Vec<f32>,

    // ── Priority tree ─────────────────────────────────────────────────────
    tree:     SumTree,
    max_prio: f64,

    // ── PER hyper-parameters ──────────────────────────────────────────────
    alpha:       f64,   // priority exponent (0 = uniform, 1 = full PER)
    beta:        f64,   // IS weight exponent (annealed 0.4 → 1.0)
    beta_start:  f64,
    beta_end:    f64,
    beta_steps:  u64,

    // ── Ring-buffer state ─────────────────────────────────────────────────
    ptr:   usize,
    size:  usize,
    steps: u64,

    // ── RNG ───────────────────────────────────────────────────────────────
    rng: SmallRng,
}

#[pymethods]
impl PERBuffer {
    #[new]
    #[pyo3(signature = (
        capacity, obs_dim, act_dim,
        alpha = 0.6,
        beta_start = 0.4,
        beta_end = 1.0,
        beta_steps = 1_000_000,
        seed = 0,
    ))]
    pub fn new(
        capacity:   usize,
        obs_dim:    usize,
        act_dim:    usize,
        alpha:      f64,
        beta_start: f64,
        beta_end:   f64,
        beta_steps: u64,
        seed:       u64,
    ) -> Self {
        PERBuffer {
            capacity,
            obs_dim,
            act_dim,
            obs:      vec![0.0_f32; capacity * obs_dim],
            actions:  vec![0.0_f32; capacity * act_dim],
            rewards:  vec![0.0_f32; capacity],
            next_obs: vec![0.0_f32; capacity * obs_dim],
            dones:    vec![0.0_f32; capacity],
            tree:     SumTree::new(capacity),
            max_prio: 1.0,
            alpha,
            beta:       beta_start,
            beta_start,
            beta_end,
            beta_steps,
            ptr:   0,
            size:  0,
            steps: 0,
            rng:   SmallRng::seed_from_u64(seed),
        }
    }

    /// Add a transition. Priority defaults to max seen so far.
    pub fn add(
        &mut self,
        obs:      Vec<f32>,
        action:   Vec<f32>,
        reward:   f32,
        next_obs: Vec<f32>,
        done:     bool,
    ) -> PyResult<()> {
        if obs.len() != self.obs_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("obs.len()={} != obs_dim={}", obs.len(), self.obs_dim)
            ));
        }
        let i = self.ptr;
        let od = self.obs_dim;
        let ad = self.act_dim;

        self.obs[i * od .. (i + 1) * od].copy_from_slice(&obs);
        self.actions[i * ad .. (i + 1) * ad].copy_from_slice(&action);
        self.rewards[i] = reward;
        self.next_obs[i * od .. (i + 1) * od].copy_from_slice(&next_obs);
        self.dones[i]   = if done { 1.0 } else { 0.0 };

        self.tree.update(i, self.max_prio.powf(self.alpha));

        self.ptr  = (self.ptr + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
        Ok(())
    }

    /// Sample a mini-batch using priority-proportional sampling.
    pub fn sample<'py>(&mut self, py: Python<'py>, batch_size: usize) -> PyResult<PyObject> {
        if self.size < batch_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("buffer has {} samples, requested {}", self.size, batch_size)
            ));
        }

        // Anneal beta
        let frac = (self.steps as f64 / self.beta_steps as f64).min(1.0);
        self.beta = self.beta_start + frac * (self.beta_end - self.beta_start);
        self.steps += 1;

        let total = self.tree.total();
        let segment = total / batch_size as f64;

        let od = self.obs_dim;
        let ad = self.act_dim;
        let b  = batch_size;

        let mut obs_out      = vec![0.0_f32; b * od];
        let mut actions_out  = vec![0.0_f32; b * ad];
        let mut rewards_out  = vec![0.0_f32; b];
        let mut next_obs_out = vec![0.0_f32; b * od];
        let mut dones_out    = vec![0.0_f32; b];
        let mut weights_out  = vec![0.0_f32; b];
        let mut indices_out: Vec<i64> = vec![0; b];

        let min_prio = self.tree.min_priority().max(1e-8);
        let max_weight = ((self.size as f64 * min_prio / total).powf(-self.beta)) as f32;

        for k in 0..b {
            let lo = segment * k as f64;
            let hi = segment * (k + 1) as f64;
            let value = lo + self.rng.gen::<f64>() * (hi - lo);
            let idx = self.tree.find(value).min(self.size - 1);

            let prio = self.tree.get(idx);
            let prob = prio / total;
            let w = ((self.size as f64 * prob).powf(-self.beta) / max_weight as f64) as f32;

            obs_out[k * od .. (k + 1) * od]
                .copy_from_slice(&self.obs[idx * od .. (idx + 1) * od]);
            actions_out[k * ad .. (k + 1) * ad]
                .copy_from_slice(&self.actions[idx * ad .. (idx + 1) * ad]);
            rewards_out[k]  = self.rewards[idx];
            next_obs_out[k * od .. (k + 1) * od]
                .copy_from_slice(&self.next_obs[idx * od .. (idx + 1) * od]);
            dones_out[k]    = self.dones[idx];
            weights_out[k]  = w;
            indices_out[k]  = idx as i64;
        }

        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("obs",
            numpy::PyArray2::from_vec2(py, &obs_out.chunks(od)
                .map(|s| s.to_vec()).collect::<Vec<_>>()).unwrap())?;
        dict.set_item("actions",
            numpy::PyArray2::from_vec2(py, &actions_out.chunks(ad)
                .map(|s| s.to_vec()).collect::<Vec<_>>()).unwrap())?;
        dict.set_item("rewards",
            numpy::PyArray1::from_vec(py, rewards_out))?;
        dict.set_item("next_obs",
            numpy::PyArray2::from_vec2(py, &next_obs_out.chunks(od)
                .map(|s| s.to_vec()).collect::<Vec<_>>()).unwrap())?;
        dict.set_item("dones",
            numpy::PyArray1::from_vec(py, dones_out))?;
        dict.set_item("weights",
            numpy::PyArray1::from_vec(py, weights_out))?;
        dict.set_item("indices",
            numpy::PyArray1::from_vec(py, indices_out))?;

        Ok(dict.into())
    }

    /// Update priorities after computing new TD errors.
    ///
    /// Args:
    ///     indices:  array of indices returned by sample()
    ///     td_errors: absolute TD errors for each sampled transition
    pub fn update_priorities(
        &mut self,
        indices:   Vec<i64>,
        td_errors: Vec<f32>,
    ) -> PyResult<()> {
        if indices.len() != td_errors.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "indices and td_errors must have the same length"
            ));
        }
        for (&idx, &err) in indices.iter().zip(td_errors.iter()) {
            let prio = (err.abs() as f64 + 1e-6).powf(self.alpha);
            self.tree.update(idx as usize, prio);
            if prio > self.max_prio {
                self.max_prio = prio;
            }
        }
        Ok(())
    }

    #[getter] pub fn len(&self)           -> usize  { self.size }
    #[getter] pub fn capacity(&self)      -> usize  { self.capacity }
    #[getter] pub fn beta(&self)          -> f64    { self.beta }
    #[getter] pub fn is_ready(&self)      -> bool   { self.size >= 256 }
    #[getter] pub fn fill_fraction(&self) -> f32    { self.size as f32 / self.capacity as f32 }

    fn __len__(&self)  -> usize  { self.size }
    fn __repr__(&self) -> String {
        format!(
            "PERBuffer(capacity={}, obs_dim={}, act_dim={}, size={}, beta={:.3})",
            self.capacity, self.obs_dim, self.act_dim, self.size, self.beta
        )
    }
}
