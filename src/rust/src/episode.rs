/*!
 * episode.rs — Episode-based replay buffer for sequence models.
 *
 * Stores complete episodes rather than individual transitions.
 * Useful for recurrent policies (RNNs, Transformers) and world models
 * (e.g. DreamerV3) that require contiguous sequences for training.
 *
 * Features:
 *   - Stores variable-length episodes up to `max_episode_len`
 *   - Returns randomly sampled sub-sequences of fixed `seq_len`
 *   - Supports "valid mask" so padding is identifiable
 *
 * Python usage:
 *   buf = mujuco_replay.EpisodeBuffer(
 *       capacity=500,       # max number of complete episodes
 *       obs_dim=64,
 *       act_dim=4,
 *       max_episode_len=500,
 *       seq_len=50,         # length returned by sample()
 *   )
 *   buf.add_episode(obs_seq, action_seq, reward_seq, done_seq)
 *   batch = buf.sample(16)   # batch of 16 sequences
 */

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use rand::prelude::*;
use rand::rngs::SmallRng;

/// A single stored episode.
struct Episode {
    obs:     Vec<f32>,    // [T × obs_dim]
    actions: Vec<f32>,    // [T × act_dim]
    rewards: Vec<f32>,    // [T]
    dones:   Vec<f32>,    // [T]
    length:  usize,
}

/// Episode-based replay buffer for sequence-based RL methods.
#[pyclass]
pub struct EpisodeBuffer {
    capacity:        usize,
    obs_dim:         usize,
    act_dim:         usize,
    max_episode_len: usize,
    seq_len:         usize,

    episodes: Vec<Option<Episode>>,
    ptr:      usize,
    size:     usize,

    rng: SmallRng,
}

#[pymethods]
impl EpisodeBuffer {
    #[new]
    #[pyo3(signature = (capacity, obs_dim, act_dim, max_episode_len = 1000, seq_len = 50, seed = 0))]
    pub fn new(
        capacity:        usize,
        obs_dim:         usize,
        act_dim:         usize,
        max_episode_len: usize,
        seq_len:         usize,
        seed:            u64,
    ) -> Self {
        let mut episodes = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            episodes.push(None);
        }
        EpisodeBuffer {
            capacity,
            obs_dim,
            act_dim,
            max_episode_len,
            seq_len,
            episodes,
            ptr:  0,
            size: 0,
            rng:  SmallRng::seed_from_u64(seed),
        }
    }

    /// Store a complete episode.
    ///
    /// Args:
    ///     obs:     float32 array of shape (T, obs_dim)
    ///     actions: float32 array of shape (T, act_dim)
    ///     rewards: float32 array of shape (T,)
    ///     dones:   bool/float array of shape (T,)
    pub fn add_episode(
        &mut self,
        obs:     Vec<f32>,
        actions: Vec<f32>,
        rewards: Vec<f32>,
        dones:   Vec<f32>,
    ) -> PyResult<()> {
        let t = rewards.len();
        if obs.len() != t * self.obs_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("obs has {} elements, expected {}×{}={}", obs.len(), t, self.obs_dim, t*self.obs_dim)
            ));
        }
        if actions.len() != t * self.act_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("actions has {} elements, expected {}×{}", actions.len(), t, self.act_dim)
            ));
        }
        if dones.len() != t {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("dones has {} elements, expected {}", dones.len(), t)
            ));
        }
        // Truncate to max_episode_len
        let t = t.min(self.max_episode_len);
        let ep = Episode {
            obs:     obs[..t * self.obs_dim].to_vec(),
            actions: actions[..t * self.act_dim].to_vec(),
            rewards: rewards[..t].to_vec(),
            dones:   dones[..t].to_vec(),
            length:  t,
        };
        self.episodes[self.ptr] = Some(ep);
        self.ptr  = (self.ptr + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
        Ok(())
    }

    /// Sample a batch of fixed-length sub-sequences.
    ///
    /// Returns a dict with keys:
    ///   obs      — (batch, seq_len, obs_dim)
    ///   actions  — (batch, seq_len, act_dim)
    ///   rewards  — (batch, seq_len)
    ///   dones    — (batch, seq_len)
    ///   mask     — (batch, seq_len)  1 = valid, 0 = padding
    pub fn sample<'py>(&mut self, py: Python<'py>, batch_size: usize) -> PyResult<PyObject> {
        if self.size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("buffer is empty"));
        }
        let sl  = self.seq_len;
        let od  = self.obs_dim;
        let ad  = self.act_dim;
        let b   = batch_size;

        let mut obs_out     = vec![0.0_f32; b * sl * od];
        let mut actions_out = vec![0.0_f32; b * sl * ad];
        let mut rewards_out = vec![0.0_f32; b * sl];
        let mut dones_out   = vec![0.0_f32; b * sl];
        let mut mask_out    = vec![0.0_f32; b * sl];

        // Collect valid episode indices
        let valid: Vec<usize> = (0..self.capacity)
            .filter(|&i| self.episodes[i].is_some())
            .collect();

        for k in 0..b {
            let ep_idx  = valid[self.rng.gen_range(0..valid.len())];
            let ep      = self.episodes[ep_idx].as_ref().unwrap();
            let ep_len  = ep.length;

            // Start index for the sub-sequence (allow partial overlap at end)
            let start = if ep_len <= sl {
                0
            } else {
                self.rng.gen_range(0..ep_len - sl + 1)
            };

            for t in 0..sl {
                let src = start + t;
                let valid_step = src < ep_len;

                let obs_dst = k * sl * od + t * od;
                if valid_step {
                    obs_out[obs_dst..obs_dst + od]
                        .copy_from_slice(&ep.obs[src * od..(src + 1) * od]);
                }
                let act_dst = k * sl * ad + t * ad;
                if valid_step {
                    actions_out[act_dst..act_dst + ad]
                        .copy_from_slice(&ep.actions[src * ad..(src + 1) * ad]);
                    rewards_out[k * sl + t] = ep.rewards[src];
                    dones_out[k * sl + t]   = ep.dones[src];
                    mask_out[k * sl + t]    = 1.0;
                }
            }
        }

        // Reshape to 3D arrays
        let obs_3d: Vec<Vec<Vec<f32>>> = (0..b).map(|k| {
            (0..sl).map(|t| obs_out[k*sl*od + t*od .. k*sl*od + (t+1)*od].to_vec()).collect()
        }).collect();
        let act_3d: Vec<Vec<Vec<f32>>> = (0..b).map(|k| {
            (0..sl).map(|t| actions_out[k*sl*ad + t*ad .. k*sl*ad + (t+1)*ad].to_vec()).collect()
        }).collect();
        let rew_2d: Vec<Vec<f32>> = (0..b).map(|k| rewards_out[k*sl..(k+1)*sl].to_vec()).collect();
        let don_2d: Vec<Vec<f32>> = (0..b).map(|k| dones_out[k*sl..(k+1)*sl].to_vec()).collect();
        let msk_2d: Vec<Vec<f32>> = (0..b).map(|k| mask_out[k*sl..(k+1)*sl].to_vec()).collect();

        let dict = pyo3::types::PyDict::new_bound(py);

        // Flatten to 2D then let Python reshape if needed
        let obs_flat: Vec<Vec<f32>> = obs_3d.into_iter().flatten().collect();
        let act_flat: Vec<Vec<f32>> = act_3d.into_iter().flatten().collect();

        dict.set_item("obs",
            numpy::PyArray2::from_vec2(py, &obs_flat).unwrap())?;
        dict.set_item("actions",
            numpy::PyArray2::from_vec2(py, &act_flat).unwrap())?;
        dict.set_item("rewards",
            numpy::PyArray2::from_vec2(py, &rew_2d).unwrap())?;
        dict.set_item("dones",
            numpy::PyArray2::from_vec2(py, &don_2d).unwrap())?;
        dict.set_item("mask",
            numpy::PyArray2::from_vec2(py, &msk_2d).unwrap())?;
        dict.set_item("seq_len", sl)?;
        dict.set_item("batch_size", b)?;

        Ok(dict.into())
    }

    /// Number of stored episodes.
    #[getter] pub fn len(&self)      -> usize { self.size }
    #[getter] pub fn capacity(&self) -> usize { self.capacity }
    #[getter] pub fn is_ready(&self) -> bool  { self.size >= 1 }

    fn __len__(&self)  -> usize  { self.size }
    fn __repr__(&self) -> String {
        format!(
            "EpisodeBuffer(capacity={}, episodes={}, obs_dim={}, act_dim={}, seq_len={})",
            self.capacity, self.size, self.obs_dim, self.act_dim, self.seq_len
        )
    }
}
