# PyMixEF native-core ABI skeleton

This C++20 library establishes the stable C ABI boundary specified by the
blueprint. The 0.1 Python engines remain independent NumPy/SciPy reference
implementations; they do not silently change calculations based on whether this
library is present.

The skeleton currently supplies version negotiation, a compensated weighted
sum-of-squares kernel, and a small dense Cholesky reference. Future sparse
Cholesky, quadrature, conditional-mode, and ODE sensitivity kernels must preserve
the public ABI or introduce an explicit versioned symbol set.

Build and test:

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

