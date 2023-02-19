def simulate_physical_devices(number_of_devices: int = 8):
    import os

    flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (
        flags + f" --xla_force_host_platform_device_count={number_of_devices}"
    )

    import jax

    print("Physical devcies: ", jax.devices())
