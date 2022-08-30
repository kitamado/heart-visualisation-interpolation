# Creating a interpolation video between 3D meshes

Pick time points 1.0s, 5.0s, 11.0s and reconstruct a interpolation video of heart development from beginning to end. 32 interpolations in between two points.

#  Matching

Regularized Brenier’s map

```python
a, x = source  # weights, locations
b, y = target  # weights, locations
z = x.clone()  # Moving point cloud

...

nits = 4 

for it in range(nits):
    wasserstein_zy = Loss(a, z, b, y)
    [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
    z -= grad_z / a[:, None]  # Apply the regularized Brenier map
```

# Interpolation

Linear interpolation.

#### lerp function
```c
// Precise method, which guarantees v = v1 when t = 1. This method is monotonic only when v0 * v1 < 0.
// Lerping between same values might not produce the same value
lerp(v0, v1, t) {
  return (1 - t) * v0 + t * v1;
}
```