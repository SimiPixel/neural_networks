from x_xy.rcmg_old import param_ident_dustin, rcmg

from neural_networks.rnno.network_local import rnno_network_local
from neural_networks.rnno.train import train

batchsize = 1024

generator = rcmg(
    batchsize,
    t_min=0.1,
    t_max=0.3,
    dang_min=0.1,
    dang_max=3.0,
    param_ident=param_ident_dustin,
)

forward_fn = rnno_network_local()

train(generator, forward_fn, 1500)
