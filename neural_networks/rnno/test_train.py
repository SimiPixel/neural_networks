import jax
import jax.numpy as jnp
from x_xy import base
from x_xy.rcmg import rcmg, rcmg_callbacks, rcmg_old_3Seg
from x_xy.utils import add_floating_base

from neural_networks.rnno import rnno_network, rnno_network_local, train
from neural_networks.rnno.optimizer import ranger


def three_segment_system() -> base.System:
    """Nodes: 6--(5)--7"""
    joint1 = base.Link.create(
        base.Transform.zero(), base.Joint(base.JointType.RevoluteY)
    )
    joint2 = base.Link.create(
        base.Transform.zero(),
        base.Joint(base.JointType.RevoluteZ),
    )
    sys = base.System(jnp.array([-1, -1]), joint1.batch(joint2))
    return add_floating_base(sys)


def rcmg_new(T, Ts, batchsize):
    sys = three_segment_system()

    @jax.jit
    def generator(key):
        return rcmg.rcmg(
            key,
            sys,
            T,
            Ts,
            batchsize=batchsize,
            params=rcmg.RCMG_Parameters(),
            flags=rcmg.RCMG_Flags(),
            callbacks=(
                rcmg_callbacks.RCMG_Callback_better_random_joint_axes(),
                rcmg_callbacks.RCMG_Callback_randomize_middle_segment_length(),
                rcmg_callbacks.RCMG_Callback_random_sensor2segment_position(),
                rcmg_callbacks.RCMG_Callback_6D_IMU_at_nodes(
                    [6, 7], [0, 2], sys.gravity, Ts
                ),
                rcmg_callbacks.RCMG_Callback_qrel_to_parent([5, 7], [6, 5], [1, 2]),
                rcmg_callbacks.RCMG_Callback_noise_and_bias(),
            ),
        )

    return generator


def test_train(batchsize=1):
    T = 10
    Ts = 0.01

    for generator in [
        rcmg_new(T, Ts, batchsize),
        rcmg_old_3Seg.rcmg_3Seg(batchsize, T=T, Ts=Ts),
    ]:
        for network in [
            rnno_network(),
            rnno_network_local(n_hidden_units=50, message_dim=30),
        ]:
            train(generator, 2, network, optimizer=ranger())
