import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from ml4cps import vis
from ml4cps.debta.rbms.gb import GaussianBinaryRBM


def plot_discretization(self, time, target, prediction, data=None, data_time=None):
    target = np.asarray(target)
    target = target[:, 0]
    if data is not None:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        for i in range(data.shape[1]):
            fig.add_trace(go.Scatter(x=data_time, y=data[:, i], name=f'Signal{i + 1}'), row=1, col=1)

        if torch.is_tensor(data):
            data = data.to(self.device)
        else:
            data = torch.tensor(data, device=self.device)

        fig.add_trace(go.Scatter(x=np.asarray(time), y=target.astype(str), name='Mode Target',
                                 mode='lines+markers'), row=2, col=1)
        fig.add_trace(go.Scatter(x=np.asarray(time), y=np.asarray(prediction).astype(str), name='Mode Prediction',
                                 mode='lines+markers'), row=2, col=1)

        error = self.recon_error(self.prepare_data(data), per_point=True)
        error_rounding = self.recon_error(self.prepare_data(data), per_point=True, round=True)
        fig.add_trace(go.Scatter(x=np.asarray(time), y=error.cpu().detach().numpy(), name='Reconstruction error',
                                 mode='lines+markers'), row=3, col=1)
        fig.add_trace(go.Scatter(x=np.asarray(time), y=error_rounding.cpu().detach().numpy(),
                                 name='Reconstruction error from rounded',
                                 mode='lines+markers'), row=3, col=1)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.asarray(time), y=target.astype(str), name='Mode Target',
                                 mode='lines+markers'))
        fig.add_trace(go.Scatter(x=np.asarray(time), y=np.asarray(prediction).astype(str), name='Mode Prediction',
                                 mode='lines+markers'))
    fig.update_layout(height=1200)
    return fig


def plot_input_space(self, data=None, samples=None, show_gaussian_components=False, data_limit=10000,
                     xmin=None, xmax=None, ymin=None, ymax=None, figure_width=600, figure_height=600,
                     show_axis_titles=True, show_energy_contours=False, showlegend=True,
                     show_recon_error_contours=False, ncontours=None,
                     plot_code_positions=True, show_recon_error_heatmap=False, plot_bias_vector=False,
                     show_reconstructions=False, **kwargs):
    fig = go.Figure()
    if show_recon_error_heatmap:
        if xmin is None and xmax is None and ymin is None and ymax is None:
            if data is None:
                xmin, xmax = -5, 5
                ymin, ymax = -5, 5
            else:
                xmin = ymin = data.min().min()
                xmax = ymax = data.max().max()

        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(x, y)
        d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
        with torch.no_grad():
            fe = self.recon_error(torch.Tensor(d)).numpy()

        trace = go.Heatmap(x=x, y=y, z=np.reshape(fe, xv.shape),
                           name="Reconstruction Error", showlegend=True, showscale=False)
        fig.add_trace(trace)

    if show_recon_error_contours and data.shape[0] == 2:
        if xmin is None and xmax is None and ymin is None and ymax is None:
            if data is None:
                xmin, xmax = -5, 5
                ymin, ymax = -5, 5
            else:
                xmin = ymin = data.min().min()
                xmax = ymax = data.max().max()

        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(x, y)
        d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
        with torch.no_grad():
            fe = self.recon_error(torch.Tensor(d)).numpy()
            fe = np.reshape(fe, xv.shape)

            trace = go.Contour(x=x, y=y, z=fe, contours=dict(coloring='lines'), name="Reconstruction Error",
                               showlegend=True, showscale=False, ncontours=ncontours)
            fig.add_trace(trace)

    if show_energy_contours:
        if xmin is None and xmax is None and ymin is None and ymax is None:
            if data is None:
                xmin, xmax = -5, 5
                ymin, ymax = -5, 5
            else:
                xmin = ymin = data.min().min()
                xmax = ymax = data.max().max()

        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(x, y)
        d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
        fe = self.free_energy(torch.Tensor(d)).detach().numpy()

        trace = go.Contour(x=x, y=y, z=np.reshape(fe, xv.shape),
                           contours=dict(coloring='lines'), name="Free energy", ncontours=ncontours, showlegend=True,
                           showscale=False)
        fig.add_trace(trace)

    if data is not None:
        if data_limit is not None and data.shape[0] > data_limit:
            data = data.sample(data_limit)
        fig.add_trace(vis.plot2d(data, x=data.columns[0], y=data.columns[1], name='Data',
                                 marker=dict(size=3, opacity=0.2, color='MediumPurple')))
        if show_reconstructions:
            recon, _ = self.recon(torch.Tensor(data.values)).detach().numpy()
            fig.add_trace(vis.plot2d(recon[:, 0], recon[:, 1], name='Reconstruction',
                                     marker=dict(size=3, opacity=0.2, color='limegreen')))
    if samples is not None:
        fig.add_trace(vis.plot2d(samples, x=samples.columns[0],
                                 y=samples.columns[1], name='Samples',
                                 marker=dict(size=3, opacity=0.2, color='darkgreen')))

    if show_axis_titles:
        fig.update_layout(
            xaxis_title="$x_1$",
            yaxis_title="$x_2$",
        )
    if plot_code_positions:
        num_h = self.num_h()
        num_v = self.num_y
        num_components = 2 ** num_h
        # Initialize
        means = np.zeros((num_components, num_v))
        hid_states = np.zeros((num_components, num_h))
        for i in range(0, num_components):
            hs = list(bin(i)[2:])
            hid_states[i, -len(hs):] = hs
            hs = hid_states[[i], :]
            # Calc means
            mean = self.decode(torch.Tensor(hs))
            means[i, :] = mean.detach().numpy()

        hm_mapping = dict()
        for h, m in zip(list(hid_states), list(means)):
            hm_mapping[str(h)] = m
        for i in range(means.shape[0]):
            mean = means[i, :]
            hid = hid_states[i, :]
            for i, hi in enumerate(hid):
                if hi == 1:
                    hid_prev = hid.copy()
                    hid_prev[i] = 0
                    mean_start = hm_mapping[str(hid_prev)]
                    fig.add_annotation(xref="x", yref="y", axref="x", ayref="y",
                                       ax=mean_start[0], ay=mean_start[1], x=mean[0], y=mean[1],
                                       showarrow=True, arrowhead=2, arrowsize=1.5)

        fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], text=hid_states, mode='text+markers',
                                 name='Codes', textfont_size=12,
                                 textposition="top left", marker_color='orange', marker_size=4))
    if plot_bias_vector:
        bx = self.bx.detach().numpy()
        fig.add_annotation(xref="x", yref="y", axref="x", ayref="y",
                           x=bx[0][0], y=bx[0][1], ax=0, ay=0,
                           showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1,
                           arrowcolor="#636363")
    if show_gaussian_components:
        weights, means, gmm_sigmas, hid_states, Z = self.gmm_model()
        hm_mapping = dict()
        for h, m in zip(list(hid_states), list(means)):
            hm_mapping[str(h)] = m
        for i in range(weights.shape[0]):
            weight = weights[i, 0]
            mean = means[i, :]
            sigma = gmm_sigmas[i, :]
            hid = hid_states[i, :]
            fig.add_shape(type="circle",
                          xref="x", yref="y",
                          x0=mean[0] - 2 * sigma[0], y0=mean[1] - 2 * sigma[1],
                          x1=mean[0] + 2 * sigma[0], y1=mean[1] + 2 * sigma[1],
                          # opacity=weight/max(max(weights)),
                          fillcolor='rgba(23, 156, 125, {:.2f})'.format(0.7 * weight / max(max(weights))),
                          line_color='rgba(23, 156, 125)',
                          line_width=1,
                          layer='below')
            for i, hi in enumerate(hid):
                if hi == 1:
                    hid_prev = hid.copy()
                    hid_prev[i] = 0
                    mean_start = hm_mapping[str(hid_prev)]
                    fig.add_annotation(xref="x", yref="y", axref="x", ayref="y",
                                       ax=mean_start[0], ay=mean_start[1], x=mean[0], y=mean[1],
                                       showarrow=True, arrowhead=2, arrowsize=1.5)

        weights = list(weights[i, :] for i in range(weights.shape[0]))
        hid_states = [' '.join(list(hid_states[i, :].astype(int).astype(str))) for i in range(hid_states.shape[0])]
        # fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], text=hid_states, mode='text+markers',
        #                          hovertext=weights,
        #                          name='GMM',
        #                          textposition="top left", marker_color='orange'))
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        title_standoff=0,
        range=[ymin, ymax]
    )
    fig.update_xaxes(
        title_standoff=0,
        range=[xmin, xmax]
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      width=figure_width,
                      height=figure_height,
                      showlegend=showlegend,
                      legend=dict(yanchor="bottom", y=1, xanchor="left", x=0.01, orientation="h",
                                  font=dict(size=8)))
    fig.update_layout(**kwargs)
    return fig

def plot_learning_curve(model):
    curves = [pd.DataFrame(model.learning_curve), pd.DataFrame(model.valid_curve)]
    columns = list(dict.fromkeys(column for curve in curves for column in curve.columns))
    # Optimization losses exist only for training; leave validation series empty.
    curves = [curve.reindex(columns=columns) for curve in curves]
    return vis.plot_timeseries(curves,
                               title='Learning curve', names=['Train', 'Valid'], xaxis_title='Epoch')

def plot_error_histogram(self, d, v=None):
    s = self.anomaly_score(d)
    fig = go.Figure(data=[go.Histogram(x=s, name="Anomaly score", histnorm="density")], layout_title="Histogram of scores")
    if v is not None:
        s = self.anomaly_score(v)
        fig.add_trace(go.Histogram(x=s, name="Anomaly score - validation set", histnorm="density"))
    if self.threshold is not None:
        fig.add_vline(x=self.threshold, line_width=2, line_dash="dash", line_color="red")
    return fig


def plot_frequency_of_latent_combinations(self, d):
    d = d.to(self.device)
    h = self.predict_discrete_mode(d)
    if type(h) is list:
        h = np.concatenate(h, axis=0)
    fig = go.Figure(data=[go.Histogram(x=h, name="Frequency", histnorm="percent")],
                    layout_title="Frequency of latent combinations")
    return fig


def plot_input_space(self, data=None, samples=None, show_gaussian_components=False, data_limit=10000,
                     xmin=None, xmax=None, ymin=None, ymax=None, figure_width=600, figure_height=600,
                     show_axis_titles=True, show_energy_contours=False, showlegend=True,
                     show_recon_error_contours=False, ncontours=None, plot_code_positions=True,
                     show_recon_error_heatmap=False, plot_bias_vector=False, show_reconstructions=False,
                     plot_separation_lines=False,
                     samples_opacity=0.2, **kwargs):
    fig = go.Figure()

    if plot_separation_lines:
        if xmin is None and xmax is None and ymin is None and ymax is None:
            if data is None:
                xmin, xmax = -5, 5
                ymin, ymax = -5, 5
            else:
                xmin = ymin = data.min().min()
                xmax = ymax = data.max().max()

        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(x, y)
        d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
        with torch.no_grad():
            binary_codes = torch.round(self.v2h(torch.Tensor(d))).numpy()

            # Step 1: Find unique binary combinations
            unique_combinations, indices = np.unique(binary_codes, axis=0, return_inverse=True)

            # Step 2: Generate colors for each unique combination
            num_combinations = len(unique_combinations)
            colors = plt.cm.get_cmap('viridis', num_combinations)(range(num_combinations))

            # Map each row in the binary_codes array to its corresponding color
            color_map = [colors[idx] for idx in indices]

            # Convert RGBA colors to Plotly-compatible hex
            color_map_hex = [f"rgba({r * 255:.0f},{g * 255:.0f},{b * 255:.0f},{a:.2f})" for r, g, b, a in color_map]

            # Step 3: Create scatter plot
            scatter = go.Scatter(
                    x=d[:, 0],
                    y=d[:, 1],
                    mode='markers',
                    opacity=0.5,
                    marker=dict(
                        symbol='square',
                        size=3,
                        color=color_map_hex  # Assign colors dynamically
                    ),
                )
            fig.add_trace(scatter)

    if show_recon_error_heatmap:
        if xmin is None and xmax is None and ymin is None and ymax is None:
            if data is None:
                xmin, xmax = -5, 5
                ymin, ymax = -5, 5
            else:
                xmin = ymin = data.min().min()
                xmax = ymax = data.max().max()

        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(x, y)
        d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
        with torch.no_grad():
            fe = self.recon_error(torch.Tensor(d)).numpy()

        trace = go.Heatmap(x=x, y=y, z=np.reshape(fe, xv.shape),
                           name="Reconstruction Error", showlegend=True, showscale=False)
        fig.add_trace(trace)

    if show_recon_error_contours and data.shape[1] == 2:
        if xmin is None and xmax is None and ymin is None and ymax is None:
            if data is None:
                xmin, xmax = -5, 5
                ymin, ymax = -5, 5
            else:
                xmin = ymin = data.min().min()
                xmax = ymax = data.max().max()

        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(x, y)
        d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])
        with torch.no_grad():
            fe = self.recon_error(torch.Tensor(d), per_point=True).numpy()
            fe = np.reshape(fe, xv.shape)

            trace = go.Contour(x=x, y=y, z=fe, contours=dict(coloring='lines'), name="Reconstruction Error",
                               showlegend=True, showscale=False, ncontours=ncontours)
            fig.add_trace(trace)

    if show_energy_contours:
        if xmin is None and xmax is None and ymin is None and ymax is None:
            if data is None:
                xmin, xmax = -5, 5
                ymin, ymax = -5, 5
            else:
                xmin = ymin = data.min().min()
                xmax = ymax = data.max().max()

        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(x, y)
        d = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1)])

        d = self.prepare_input(torch.Tensor(d))
        fe = self.free_energy(d).detach().numpy()

        trace = go.Contour(x=x, y=y, z=np.reshape(fe, xv.shape), contours=dict(coloring='lines'),
                           name="Free energy", ncontours=ncontours, showlegend=True, showscale=False)
        fig.add_trace(trace)

    if data is not None:
        if data_limit is not None and data.shape[0] > data_limit:
            data = data.sample(data_limit)
        fig.add_trace(vis.plot2d(data, data.columns[0], data.columns[1], name='Data',
                                 marker=dict(size=3, opacity=0.2, color='MediumPurple')))
        if show_reconstructions:
            recon, h_recon = self.recon(torch.tensor(data.values).float(), round=True)
            recon = pd.DataFrame(recon.detach().numpy())
            fig.add_trace(vis.plot2d(recon, recon.columns[0], recon.columns[1], name='Reconstruction',
                                     marker=dict(size=3, opacity=0.2, color='limegreen')))
    if samples is not None:
        col1 = samples.columns[0]
        col2 = samples.columns[1]
        samples = self.decode_input(torch.tensor(samples.to_numpy())).detach().numpy()
        samples = pd.DataFrame(samples, columns=[col1, col2])
        fig.add_trace(vis.plot2d(samples, col1, col2, name='Samples',
                                 marker=dict(size=3, opacity=samples_opacity, color='darkgreen')))

    if show_axis_titles:
        fig.update_layout(
            xaxis_title="$x_1$",
            yaxis_title="$x_2$",
        )
    if plot_code_positions:
        num_h = self.n_hidden
        num_v = self.n_visible
        num_components = 2 ** num_h
        # Initialize
        means = np.zeros((num_components, num_v))
        hid_states = np.zeros((num_components, num_h))
        for i in range(0, num_components):
            hs = list(bin(i)[2:])
            hid_states[i, -len(hs):] = hs
            hs = hid_states[[i], :]
            # Calc means
            with torch.no_grad():
                mean = self.h2v(torch.Tensor(hs))
                mean = self.decode_input(mean)
                means[i, :] = mean.detach().numpy().reshape(1, -1)

        hm_mapping = dict()
        for h, m in zip(list(hid_states), list(means)):
            hm_mapping[str(h)] = m
        for i in range(means.shape[0]):
            mean = means[i, :]
            hid = hid_states[i, :]
            for i, hi in enumerate(hid):
                if hi == 1:
                    hid_prev = hid.copy()
                    hid_prev[i] = 0
                    mean_start = hm_mapping[str(hid_prev)]
                    fig.add_annotation(xref="x", yref="y", axref="x", ayref="y", ax=mean_start[0], ay=mean_start[1],
                                       x=mean[0], y=mean[1], showarrow=True, arrowhead=2, arrowsize=1.5)

        fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], text=hid_states, mode='text+markers', name='Codes',
                                 textfont_size=12, textposition="top left", marker_color='orange', marker_size=4))
    if plot_bias_vector:
        bx = self.h2v(torch.tensor(np.zeros((1, self.n_hidden)), requires_grad=False).float()) # self.visible_bias.detach()[None] #* np.exp(self.log_sigma_x.detach().numpy()).flatten()
        bx = self.decode_input(bx.detach()).numpy().flatten()
        fig.add_annotation(xref="x", yref="y", axref="x", ayref="y", x=bx[0], y=bx[1], ax=0, ay=0,
                           showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1, arrowcolor="#636363")
    if show_gaussian_components and isinstance(self, GaussianBinaryRBM):
        weights, means, gmm_sigmas, hid_states, Z = self.gmm_model()
        hm_mapping = dict()
        for h, m in zip(list(hid_states), list(means)):
            hm_mapping[str(h)] = m
        for i in range(weights.shape[0]):
            weight = weights[i, 0]
            mean = means[i, :]
            sigma = gmm_sigmas[i, :]
            hid = hid_states[i, :]
            fig.add_shape(type="circle",
                          xref="x", yref="y",
                          x0=mean[0] - 2 * sigma[0], y0=mean[1] - 2 * sigma[1],
                          x1=mean[0] + 2 * sigma[0], y1=mean[1] + 2 * sigma[1],
                          # opacity=weight/max(max(weights)),
                          fillcolor='rgba(23, 156, 125, {:.2f})'.format(0.7 * weight / max(max(weights))),
                          line_color='rgba(23, 156, 125)',
                          line_width=1,
                          layer='below')
            for i, hi in enumerate(hid):
                if hi == 1:
                    hid_prev = hid.copy()
                    hid_prev[i] = 0
                    mean_start = hm_mapping[str(hid_prev)]
                    fig.add_annotation(xref="x", yref="y", axref="x", ayref="y",
                                       ax=mean_start[0], ay=mean_start[1], x=mean[0], y=mean[1],
                                       showarrow=True, arrowhead=2, arrowsize=1.5)

        weights = list(weights[i, :] for i in range(weights.shape[0]))
        hid_states = [' '.join(list(hid_states[i, :].astype(int).astype(str))) for i in range(hid_states.shape[0])]
        # fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], text=hid_states, mode='text+markers',
        #                          hovertext=weights,
        #                          name='GMM',
        #                          textposition="top left", marker_color='orange'))
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        title_standoff=0,
        range=[ymin, ymax]
    )
    fig.update_xaxes(
        title_standoff=0,
        range=[xmin, xmax]
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      width=figure_width,
                      height=figure_height,
                      showlegend=showlegend,
                      legend=dict(yanchor="bottom", y=1, xanchor="left", x=0.01, orientation="h",
                                  font=dict(size=8)))
    fig.update_layout(**kwargs)


    return fig
