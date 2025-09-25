---
name: Meta Optimization Task
about: Template for tasks where we can’t directly optimize ops but apply meta-level
  improvements
title: "[Module]: "
labels: meta-optimization
assignees: ''

---

### Task
<short description, e.g. Apply meta-optimizations to ConvGRU module.>

### Strategy
- [ ] Wrap module / function with GraphCachedModule
- [ ] Test CUDA Graph capture in inference loop
- [ ] Remove Python overhead (buffer reuse, prealloc, etc.)
- [ ] Validate correctness

### Expected Outcome
Improved runtime efficiency (latency, throughput, or memory) without rewriting core kernels.  

### Reference
- Related code snippet or class:
```python
class ConvGRU(nn.Module):
    ...
