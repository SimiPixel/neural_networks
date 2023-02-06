import jax
from x_xy import rcmg, rcmg_callbacks
from x_xy.examples.three_segments import three_segment_system

from neural_networks.logging import NeptuneLogger
from neural_networks.rnno import RNNO_Config, rnno_network, train


def three_segment_generator():
    sys = three_segment_system()

    T = 60
    Ts = 0.01

    @jax.jit
    def generator(key):
        return rcmg.rcmg(
            key,
            sys,
            T,
            Ts,
            batchsize=16,
            callbacks=(
                rcmg_callbacks.RCMG_Callback_randomize_middle_segment_length(),
                rcmg_callbacks.RCMG_Callback_random_sensor2segment_position(),
                rcmg_callbacks.RCMG_Callback_6D_IMU_at_nodes(
                    [6, 7], [0, 2], sys.gravity, Ts
                ),
                rcmg_callbacks.RCMG_Callback_qrel_to_parent([5, 7], [6, 5], [1, 2]),
                rcmg_callbacks.RCMG_Callback_noise_and_bias([0, 2]),
            ),
        )

    return generator


def main():
    logger = NeptuneLogger("iss/social-rnno", name="my fancy run..")
    generator = three_segment_generator()

    network = rnno_network()
    train(
        generator,
        network,
        RNNO_Config(n_episodes=100),
        [logger],
        eval_dustin_exp_every=2,
        network_dustin=network,
    )


if __name__ == "__main__":
    main()
