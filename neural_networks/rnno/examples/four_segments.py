from x_xy.rcmg.rcmg_old_4Seg import rcmg_4Seg

from neural_networks.rnno.network_local import rnno_network_local
from neural_networks.rnno.train import train

batchsize = 1024

generator = rcmg_4Seg(
    batchsize,
    t_min=0.1,
    t_max=0.5,
    dang_min=0.1,
    dang_max=2.4,
)

network = rnno_network_local(length_of_chain=4)
network_dustin = rnno_network_local(length_of_chain=3)

train(
    generator,
    1500,
    network,
    network_dustin,
    project_name="rnno-4Seg",
    run_name="first trial",
    log_to_neptune=False,
)
