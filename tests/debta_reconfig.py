import pandas as pd
import tools
from denta.debta_tools import WindowedSequenceDataset
from ml4cps.examples import simple_conveyor_8_states
from ml4cps import denta

if __name__ == "__main__":
    ha = simple_conveyor_8_states()
    q, d, t, y, finish_time = ha.simulate(finish_time=1000)
    ds = WindowedSequenceDataset([y])

    # mdl = denta.DENTA(num_y=2, first_hidden_size=10, num_sigm_layers=2, num_h=3, sigma=0.1)
    # mdl.pretrain_layers(ds, valid_data=ds, lr=[0.001, 0.01], verbose=True, max_epoch=50, batch_size=1000)

    mdl = denta.DENTA(num_y=2, num_sigm_layers=1, num_h=3, sigma=0.5)
    mdl.pretrain_layers(ds, valid_data=ds, lr=[0.01, 0.01], num_k=5, verbose=True, max_epoch=50, batch_size=128)


    generated = pd.DataFrame(mdl.ebm.generate(1000)[:,0,:].detach().numpy())
    mdl.ebm.plot_input_space(y, samples=generated,
                             xmin=-0.5, xmax=1, ymin=-0.5, ymax=1.5).show('browser') # show_gaussian_components=True,