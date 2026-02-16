# Sequence Prediction Interference

## Problem Statement

The PredictiveLattice (coupled oscillator network) shows degraded accuracy when learning multiple sequences:
- Single sequence: Works
- Multiple sequences: ~67% → drops to <10% in stress tests
- "What comes after X?" predictions become unreliable

## Root Cause Analysis

### Diagnostic Methodology

We implemented a 4-condition A/B test to separate interference modes:

| Condition | ω (frequency) | Tokens | Tests |
|-----------|---------------|--------|-------|
| A | Random | Random (overlapping) | Baseline |
| B | Identical | Random | Phase drift |
| C | Random | Collision-free | Overlap |
| D | Identical | Collision-free | Upper bound |

### Test Results

```
Baseline (A): 0.0%
Fixing drift only (B-A): +6.7%
Fixing overlap only (C-A): +6.7%
Fixing both (D-A): +6.7%

Diagnosis: SATURATION/NORMALIZATION is dominant
```

### Three Root Causes Identified

#### 1. Weight Saturation (DOMINANT)

The weight matrix normalizes rows to prevent explosion:
```python
def _normalize(self):
    norms = np.linalg.norm(self.W, axis=1, keepdims=True)
    self.W = np.where(norms > 1.0, self.W / (norms + 1e-8), self.W)
```

**Problem:** As sequences are added, each edge gets diluted. The signal-to-noise ratio drops below the discrimination threshold.

**Evidence:** Even with zero overlap and identical frequencies (optimal conditions), accuracy is only 6.7%.

#### 2. Concept Overlap (CONTRIBUTING)

With K=10 neurons per concept and L=256 lattice:
```
P(overlap) ≈ 1 - (1 - K/L)^(V*K)
For V=20 concepts: ~53% pairs have overlap
```

**Impact:** Shared neurons create "shortcut" paths between unrelated sequences.

#### 3. Phase Drift (MINOR)

Random natural frequencies (ω) cause phase desynchronization:
```python
self.omega = np.random.uniform(0.5, 2.0, n)  # Different for each neuron
```

**Impact:** Patterns learned at one phase alignment may not trigger at another.

## Solution Approaches

### Approach 1: Hippocampus-Based Sequences (RECOMMENDED)

Store sequences as explicit facts instead of learned dynamics:
```python
# Instead of training oscillators:
store_fact("after_wake", "next", "shower")
store_fact("after_shower", "next", "dress")

# Prediction becomes fact lookup:
def predict_next(current):
    return query_fact(f"after_{current}", "next")
```

**Pros:**
- 100% accuracy (uses proven HippocampusVSA)
- No interference between sequences
- Instant learning

**Cons:**
- Loses emergent dynamics
- Explicit rather than implicit

### Approach 2: Mitigation Stack (For Oscillator Approach)

If keeping coupled oscillators:

```python
# 1. Hard sparsity in updates
active_sources = np.argsort(np.abs(self.e))[-top_k:]
active_targets = np.argsort(np.abs(self.z))[-top_k:]
# Only update edges between these

# 2. Per-edge decay + cap (instead of row normalization)
self.W *= (1 - decay_rate)
self.W = np.clip(self.W, -max_weight, max_weight)

# 3. Phase reset on stimulation
def stimulate(self, indices, amplitude=1.0):
    for idx in indices:
        self.z[idx] = amplitude * np.exp(1j * 0)  # Fixed phase

# 4. Bounded out-degree
for i in range(n):
    row = np.abs(self.W[i, :])
    weak = np.argsort(row)[:-max_outgoing]
    self.W[i, weak] = 0
```

### Approach 3: Larger Lattice + Sparser Encoding

```python
L = 1024  # Instead of 256
K = 5     # Instead of 10

# Overlap probability drops dramatically
P(overlap) ≈ 1 - (1 - 5/1024)^(20*5) ≈ 37% (vs 53%)
```

## Verification Method

```bash
python3 test_interference_diagnosis.py
```

### Test Files

- `test_interference_diagnosis.py` - 4-condition A/B test
- `test_sequence_analysis.py` - Overall impact assessment

## Current Status

| Metric | Value | Target |
|--------|-------|--------|
| Single sequence | 100% | 100% |
| Non-overlapping | 100% | 100% |
| Multiple overlapping | <10% | 80%+ |

### Recommendation

**Use Hippocampus-based sequences for production.** The coupled oscillator approach is valuable for research but not reliable enough for multi-sequence deployment.

## Lessons Learned

1. **Diagnose before fixing** - The A/B/C/D test revealed saturation was dominant, not overlap
2. **Normalization is a tradeoff** - Prevents explosion but causes dilution
3. **Simpler often wins** - Explicit storage outperforms learned dynamics for this use case

## Related

- [03-weight-saturation](03-weight-saturation.md) - Detailed saturation analysis
- [01-vsa-cleanup-memory](01-vsa-cleanup-memory.md) - The reliable alternative

---

*Diagnosed: 2026-02-16*
*Status: DIAGNOSED - Mitigation strategies identified*
