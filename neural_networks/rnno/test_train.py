from x_xy.examples.three_segments import three_segment_generator

from neural_networks.rnno import RNNO_Config, rnno_network, rnno_network_local, train


def test_train():
    T = 10
    Ts = 0.01
    generator = three_segment_generator(T, Ts)

    for network in [
        rnno_network(),
        rnno_network_local(n_hidden_units=50, message_dim=30),
    ]:
        train(
            generator,
            network,
            RNNO_Config(2),
            [],
            eval_dustin_exp_every=1,
            network_dustin=network,
        )
