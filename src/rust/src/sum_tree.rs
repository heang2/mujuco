/*!
 * sum_tree.rs — Segment tree for O(log N) priority updates and sampling.
 *
 * Used by PERBuffer to implement Prioritized Experience Replay.
 * The tree stores priorities; sampling is proportional to priority.
 *
 * Layout: 1-indexed binary heap stored in a flat Vec<f64>.
 *   - Node 1:          root (sum of all priorities)
 *   - Nodes 2..=3:     children of root
 *   - Leaves n..=2n-1: individual priorities
 */

/// A binary sum-tree supporting O(log N) update and O(log N) prefix-sum query.
pub struct SumTree {
    n:    usize,       // number of leaf nodes (capacity)
    tree: Vec<f64>,    // size = 2*n; index 0 unused, root at 1
    min:  Vec<f64>,    // parallel min-tree for IS-weight normalization
}

impl SumTree {
    /// Create a new sum-tree with `capacity` leaves.
    pub fn new(capacity: usize) -> Self {
        let n = capacity.next_power_of_two();
        SumTree {
            n,
            tree: vec![0.0_f64; 2 * n],
            min:  vec![f64::INFINITY; 2 * n],
        }
    }

    /// Total sum of all priorities (the root value).
    #[inline]
    pub fn total(&self) -> f64 {
        self.tree[1]
    }

    /// Minimum priority among all stored values.
    #[inline]
    pub fn min_priority(&self) -> f64 {
        self.min[1]
    }

    /// Update the priority at leaf `idx` (0-indexed).
    pub fn update(&mut self, idx: usize, priority: f64) {
        assert!(idx < self.n, "SumTree index {} out of range {}", idx, self.n);
        let mut i = self.n + idx; // leaf position in the 1-indexed tree
        self.tree[i] = priority;
        self.min[i]  = priority;
        // Propagate up
        i >>= 1;
        while i >= 1 {
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1];
            self.min[i]  = self.min[2 * i].min(self.min[2 * i + 1]);
            if i == 1 { break; }
            i >>= 1;
        }
    }

    /// Find the leaf index whose cumulative-sum prefix first exceeds `value`.
    ///
    /// Returns the 0-indexed leaf position.
    pub fn find(&self, mut value: f64) -> usize {
        let mut i = 1_usize;
        while i < self.n {
            let left = 2 * i;
            if value <= self.tree[left] {
                i = left;
            } else {
                value -= self.tree[left];
                i = left + 1;
            }
        }
        i - self.n  // convert to 0-indexed leaf
    }

    /// Retrieve the priority at leaf `idx` (0-indexed).
    pub fn get(&self, idx: usize) -> f64 {
        self.tree[self.n + idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_tree_basic() {
        let mut t = SumTree::new(4);
        t.update(0, 1.0);
        t.update(1, 2.0);
        t.update(2, 3.0);
        t.update(3, 4.0);
        assert!((t.total() - 10.0).abs() < 1e-9);
        assert!((t.min_priority() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sum_tree_find() {
        let mut t = SumTree::new(4);
        t.update(0, 1.0);
        t.update(1, 1.0);
        t.update(2, 1.0);
        t.update(3, 1.0);
        // With uniform priorities, each quadrant maps to one leaf.
        assert_eq!(t.find(0.5), 0);
        assert_eq!(t.find(1.5), 1);
        assert_eq!(t.find(2.5), 2);
        assert_eq!(t.find(3.5), 3);
    }
}
