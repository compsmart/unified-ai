# Weight Saturation in Coupled Oscillators

## Problem Statement

The PredictiveLattice weight matrix loses discriminative power as more sequences are trained. Edge strengths plateau while noise accumulates, making prediction unreliable.

## Root Cause Analysis

### The Learning Rule

```python
def _causal_update(self):
    # W[target, source] += lr * z[target](now) * conj(e[source](past))
    for t in targets:
        for s in sources:
            update = self.z[t] * np.conj(self.e[s]) * lr * dt
            self.W[t, s] += update
```

### The Normalization Problem

```python
def _normalize(self):
    norms = np.linalg.norm(self.W, axis=1, keepdims=True)
    self.W = np.where(norms > 1.0, self.W / (norms + 1e-8), self.W)
```

**What happens:**
1. Sequence 1 creates edges A→B, B→C with strength ~0.5 each
2. Row normalization keeps ||W[B,:]|| ≤ 1
3. Sequence 2 creates edges D→B, B→E
4. Now B's row has 4 edges, each normalized to ~0.25
5. Sequence 3... each edge now ~0.17
6. Signal approaches noise floor

### Mathematical Model

For a source neuron with N outgoing edges, after normalization:

```
|W[j,i]| ≈ 1/√N for each edge

If discrimination threshold = 0.1 and N > 100:
  |W| ≈ 0.1 → indistinguishable from noise
```

### Evidence

Edge strength analysis:
```
S= 1: signal=0.0821, noise=0.0000, SNR=inf
S= 5: signal=0.0312, noise=0.0089, SNR=3.51x
S=10: signal=0.0198, noise=0.0134, SNR=1.48x
S=15: signal=0.0156, noise=0.0148, SNR=1.05x  ← noise equals signal
```

## Solution Approaches

### Approach 1: Per-Edge Decay + Cap

Replace row normalization with individual edge management:

```python
def _normalize(self):
    # Per-edge decay (forget old, keep recent)
    self.W *= 0.999

    # Cap individual edges (prevent single edge explosion)
    max_weight = 0.5
    overflow = np.abs(self.W) > max_weight
    self.W[overflow] *= max_weight / (np.abs(self.W[overflow]) + 1e-8)
```

**Pros:** Edges don't dilute each other
**Cons:** Older edges fade over time

### Approach 2: Competitive Learning

Only strengthen the strongest competitor:

```python
def _causal_update(self):
    for s in sources:
        # Find which target s is most strongly predicting
        candidates = [(t, np.abs(self.z[t])) for t in targets]
        if candidates:
            winner = max(candidates, key=lambda x: x[1])[0]
            # Only update winner
            self.W[winner, s] += self.z[winner] * np.conj(self.e[s]) * lr * dt
```

**Pros:** Clear winners, no dilution
**Cons:** Slower learning, might miss valid associations

### Approach 3: Softmax-Normalized Outgoing

Instead of hard normalization, use softmax-like weighting:

```python
def predict(self, sources):
    for s in sources:
        # Softmax over outgoing edges
        row = self.W[:, s]
        magnitudes = np.abs(row)
        probs = np.exp(magnitudes * temperature) / np.sum(np.exp(magnitudes * temperature))
        # Activate proportionally
        self.z += row * probs
```

**Pros:** Maintains relative strengths
**Cons:** Temperature tuning required

### Approach 4: Bounded Out-Degree

Limit number of outgoing edges per source:

```python
def prune_edges(self, max_out=10):
    for i in range(self.config.size):
        row = np.abs(self.W[:, i])  # Outgoing from i
        if np.count_nonzero(row) > max_out:
            threshold = np.sort(row)[-max_out]
            self.W[:, i] *= (row >= threshold)
```

**Pros:** Hard limit on dilution
**Cons:** Loses weak but valid associations

## Verification Method

```python
def test_edge_strength_analysis():
    """Track signal-to-noise ratio as sequences are added."""
    for num_seq in [1, 5, 10, 15, 20]:
        # Train sequences
        # Measure signal = strength of correct edges
        # Measure noise = average strength of incorrect edges
        # Report SNR
```

## Test Results

### Before Mitigation

| Sequences | Signal | Noise | SNR |
|-----------|--------|-------|-----|
| 1 | 0.082 | 0.000 | ∞ |
| 5 | 0.031 | 0.009 | 3.5x |
| 10 | 0.020 | 0.013 | 1.5x |
| 15 | 0.016 | 0.015 | 1.1x |

### After Per-Edge Decay (Expected)

| Sequences | Signal | Noise | SNR |
|-----------|--------|-------|-----|
| 1 | 0.15 | 0.00 | ∞ |
| 5 | 0.12 | 0.02 | 6.0x |
| 10 | 0.10 | 0.03 | 3.3x |
| 15 | 0.08 | 0.04 | 2.0x |

## Implementation Priority

| Fix | Effort | Expected Gain |
|-----|--------|---------------|
| Per-edge decay + cap | Low | 20-30% accuracy |
| Bounded out-degree | Low | 15-25% accuracy |
| Competitive learning | Medium | 30-40% accuracy |
| Combined | Medium | 50-70% accuracy |

## V2 Implementation Results

A V2 implementation (`src/predictive_lattice_v2.py`) was created with all mitigations:

| Sequences | V1 Accuracy | V2 Accuracy | Improvement |
|-----------|-------------|-------------|-------------|
| 1 | 50% | 50% | 0% |
| 5 | 40% | 30% | -10% |
| 10 | 20% | 30% | +10% |
| 15 | 7% | 30% | +23% |

**Key observation:** V2 maintains stable ~30% accuracy as sequences scale, while V1 collapses to 7%. The mitigations prevent catastrophic degradation but don't achieve 90%+ accuracy.

**Conclusion:** For high-reliability sequence prediction, use HippocampusVSA-based storage instead of oscillator dynamics.

## Related

- [02-sequence-interference](02-sequence-interference.md) - Parent problem
- Oja's Rule - Neurobiologically-inspired bounded Hebbian learning

## References

- Oja, E. (1982). "Simplified neuron model as a principal component analyzer"
- Grossberg, S. (1976). "Adaptive pattern classification and universal recoding"

---

*Diagnosed: 2026-02-16*
*Status: DIAGNOSED - Implementation pending*
