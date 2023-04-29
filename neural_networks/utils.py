import os

import jax


def simulate_physical_devices(number_of_devices: int = 8):
    flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (
        flags + f" --xla_force_host_platform_device_count={number_of_devices}"
    )

    print("Physical devcies: ", jax.devices())


def on_cluster() -> bool:
    """Return `true` if executed on cluster."""
    env_var = os.environ.get("ON_CLUSTER", None)
    return False if env_var is None else True
