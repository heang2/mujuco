/*!
 * mujoco_replay — High-performance replay buffer in Rust
 *
 * Exposed to Python via PyO3.  Provides:
 *   - `UniformBuffer`  — O(1) add, O(batch) sample, ring-buffer layout
 *   - `PERBuffer`      — Prioritized Experience Replay with sum-tree
 *   - `EpisodeBuffer`  — Stores full episodes (for sequence-based methods)
 *
 * Build:
 *   cd src/rust
 *   maturin develop --release          # installs into the active venv
 *   # or
 *   maturin build --release            # produces a .whl
 *
 * Install maturin:
 *   pip install maturin
 *
 * Python usage after build:
 *   import mujoco_replay
 *   buf = mujoco_replay.UniformBuffer(capacity=1_000_000, obs_dim=11, act_dim=3)
 *   buf.add(obs, action, reward, next_obs, done)
 *   batch = buf.sample(256)
 *   # batch is a dict of numpy arrays: obs, actions, rewards, next_obs, dones, weights, indices
 */

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::sync::atomic::{AtomicUsize, Ordering};

mod uniform;
mod per;
mod episode;
mod sum_tree;

pub use uniform::UniformBuffer;
pub use per::PERBuffer;
pub use episode::EpisodeBuffer;

// ── Module registration ────────────────────────────────────────────────────

#[pymodule]
fn mujoco_replay(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UniformBuffer>()?;
    m.add_class::<PERBuffer>()?;
    m.add_class::<EpisodeBuffer>()?;
    m.add("__version__", "0.1.0")?;
    m.add("__author__", "MuJoCo Playground")?;
    Ok(())
}
