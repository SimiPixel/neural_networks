# RNN-based Observer (RNNO)

**Currently only supports kinematic chains. No branching!**

## Networks are maps from

```python
params, state = network.init(key, X)
y, state = network.apply(params, state, X)
```
where `X` has no batchsize dimension. Batching is done via `vmap`

## What the networks expect as `X` and returns as `y`
Both `rnno_network` and `rnno_network_local` expect the input `X` to be

```python
X = {
    # the Tree of jax.Arrays should be of shape (n_timesteps, M)
    # where M matches accross nodes that can measure
    # so currently mixing 9D and 6D IMUs would not be possible
    0: Tree,
    N-1: Tree
}
```

and what it returns is
```python
y = {
    # the jax.Array should be of shape (n_timesteps, 4)
    1: jax.Array
    2: jax.Array
    ...
    N-1: jax.Array 
}
```

## Dependencies
- `x_xy`
- `dm-haiku`
- `tree_utils`
