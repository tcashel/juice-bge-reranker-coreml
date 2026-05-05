"""
Vendored slice of Apple's `ane_transformers` reference implementation.

Source: https://github.com/apple/ml-ane-transformers
License: Apple sample-code license — see ./LICENSE.

We vendor (rather than pip-depending on `ane-transformers`) because the upstream
package strict-pins `torch<=1.11` (2022-era), but the actual reference code is
plain PyTorch ops that run fine on torch 2.x. Vendoring also keeps the (B, C, 1, S)
layout primitives near our XLM-RoBERTa adaptation without dragging the whole
package surface area in.
"""
