from ml4cps import examples
from ml4cps.vis import plot_timeseries, add_time_frames_to_subplots

ta = examples.simple_conveyor_8_states()

discr_state_data, discr_output_data, cont_state_data, cont_output_data, finish_time = ta.simulate(finish_time=3)



fig_ts = plot_timeseries(cont_output_data, mode_data=discr_state_data, height=300)
fig_anim = add_time_frames_to_subplots(fig_ts)
fig_anim.show()