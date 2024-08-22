"""
    The module implements the novel DENTA algorithm for the learning of hybrid automata from data.

    Author:
    Nemanja Hranisavljevic, hranisan@hsu-hh.de, nemanja@ai4cps.com
"""

import pprint
import time
import pandas as pd
from plotly import graph_objects as go
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from datetime import timedelta
from multiprocessing import cpu_count
from automata4cps import automata, tools
from automata4cps import learn as automata_learn
from sklearn.preprocessing import LabelEncoder
from plotly.subplots import make_subplots
import mlflow
from torch import optim


class DENTA(nn.Module, automata.Automaton):
    def __init__(self, num_signals, num_hidden_units, first_hidden_size=None, sigma=1., sigma_learnable=False,
                 sparsity_weight=0.01, persistence=True, num_hidden_layers=1,
                 window_size=1, window_step=1, sparsity_target=0.1, use_derivatives=0, device='cpu', log_mlflow=False):
        super(DENTA, self).__init__()
        self.variant = 'gbrbm'
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        self.persistence = persistence
        self.use_derivatives = np.asarray(use_derivatives).reshape(-1).tolist()
        self.window_size = window_size
        self.window_step = window_step
        self.device = device
        self._mean = None
        self._std = None

        self.log_mlflow = log_mlflow

        if use_derivatives:
            input_size = num_signals * len(self.use_derivatives)
        else:
            input_size = num_signals

        if sigma_learnable:
            self.is_sigma_learnable = True
            self.log_sigma_x = nn.Parameter(np.log(sigma) * torch.ones(1, input_size, requires_grad=True)).to(device)
        else:
            self.is_sigma_learnable = False
            self.log_sigma_x = np.log(sigma) * torch.ones(1, input_size, requires_grad=False).to(device)

        if self.window_size > 1:
            input_size = input_size * self.window_size

        self._encoder = nn.Sequential().to(device)
        self._decoder = nn.Sequential().to(device)

        # bias of the input we take as separate parameter for implementation reasons
        # self.bx = nn.Parameter(torch.zeros(1, num_sig, requires_grad=True)).to(device)

        # add encoder layers
        # if self.use_rnn:
        #     enc_layer = nn.LSTMCell(num_signals, num_hidden, device=device)
        #     self._encoder.add_module('rnn_v2h', enc_layer)
        #     # self._encoder.add_module('sigmoid_v2h', nn.Sigmoid())
        # else:

        enc_layers = []
        if first_hidden_size is None:
            if num_hidden_layers == 1:
                first_hidden_size = num_hidden_units
            else:
                first_hidden_size = int(input_size * 1.5)

        layer_sizes = ([input_size] +
                       np.round(np.linspace(first_hidden_size, num_hidden_units, num_hidden_layers)).astype(int).tolist())
        for i in range(num_hidden_layers):
            enc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], device=device))
            self._encoder.add_module(f'linear_v2h{i}', enc_layers[i])
            self._encoder.add_module(f'sigmoid_v2h{i}', nn.Sigmoid())

        dec_layers = []
        layer_sizes.reverse()
        for i in range(num_hidden_layers):
            dec_lin = nn.Linear(layer_sizes[i], layer_sizes[i+1], device=device)
            dec_lin.weight = nn.Parameter(enc_layers[-i-1].weight.transpose(0, 1))
            dec_layers.append(dec_lin)
            self._decoder.add_module(f'linear_h2v{i}', dec_lin)
            if i != num_hidden_layers - 1:
                self._decoder.add_module(f'sigmoid_h2v{i}', nn.Sigmoid())


        # self.free_energy_components = nn.Sequential()
        # self.free_energy_components.add_module('linear_energy', enc_layer)
        # self.free_energy_components.add_module('softplus_energy', nn.Softplus())

        self.threshold = None
        self.learning_curve = []
        self.valid_curve = []
        self.num_epoch = 0

        if log_mlflow:
            mlflow.log_param("model", "denta")
            mlflow.log_param("num_signals", num_signals)
            mlflow.log_param("input_size", input_size)
            mlflow.log_param("num_hidden_layers", num_hidden_layers)
            mlflow.log_param("num_hidden_units", num_hidden_units)
            mlflow.log_param("sparsity_target", self.sparsity_target)
            mlflow.log_param("sparsity_weight", self.sparsity_weight)
            mlflow.log_param("sigma", sigma)
            mlflow.log_param("use_derivatives", self.use_derivatives)
            mlflow.log_param("window_size", self.window_size)
            mlflow.log_param("first_hidden_size", first_hidden_size)

    def encode(self, x, rounding=False, x_level=0, h_level=None):
        if h_level is None:
            h_level = self.num_hidden_layers()
        if type(x) is list:
            return [self.encode(xx, rounding=rounding) for xx in x]
        else:
            sigma_x = torch.exp(self.log_sigma_x)
            if x.dim() == 3:
                sigma_x = sigma_x.unsqueeze(2)
            x = torch.div(x, sigma_x)
            x = x.reshape(x.size(0), -1)
            h = self._encoder(x)
            if rounding:
                h = torch.round(h)
            return h

    def prepare_data(self, x, update_mean_std=False):
        x = self._extend_derivative(x)
        x = self._standardize(x, fit=update_mean_std)
        x = self._window(x)
        return x

    def _standardize(self, x, fit=False):
        if type(x) is list:
            if fit:
                all_x = torch.vstack(x)
                self._mean = all_x.mean(dim=0)
                self._std = all_x.std(dim=0)

            return [(xx - self._mean) / self._std for xx in x]
        else:
            if fit:
                self._mean = x.mean(dim=0)
                self._std = x.std(dim=0)
            return (x - self._mean) / self._std

    def _window(self, x):
        if type(x) is list:
            return [self._window(xx) for xx in x]
        else:
            return x.unfold(dimension=0, size=self.window_size, step=self.window_step)

    def _extend_derivative(self, signals):
        if type(signals) is list:
            return [self._extend_derivative(x) for x in signals]
        else:
            new_signals = [signals]
            for ord in range(0, max(self.use_derivatives)):
                # Initialize a tensor to hold the derivatives, same shape as the input
                derivatives = torch.zeros_like(signals)

                # Use central differences for the interior points
                derivatives[1:-1, :] = (signals[2:, :] - signals[:-2, :]) / 2

                # Use forward difference for the first point
                derivatives[0, :] = signals[1, :] - signals[0, :]

                # Use backward difference for the last point
                derivatives[-1, :] = signals[-1, :] - signals[-2, :]

                new_signals.append(derivatives)
                signals = derivatives

            new_signals = [new_signals[i] for i in self.use_derivatives]
            new_signals = torch.hstack(new_signals)
        return new_signals

    def predict_discrete_mode(self, data, prepare_data=True):
        if type(data) is list:
            return [self.predict_discrete_mode(d) for d in data]
        else:
            if prepare_data:
                data = self.prepare_data(data)
            h = self.encode(data, rounding=True)
            h = self.binary_vector_to_mode(h)
            return np.concatenate([np.full((self.window_size-1,), np.nan), h], axis=0)

    def binary_vector_to_mode(self, h):
        if type(h) is list:
            return [self.binary_vector_to_mode(x) for x in h]
        else:
            df = pd.DataFrame(h.cpu().detach().numpy())
            df = df.astype(int).astype(str)
            return df.agg(''.join, axis=1).to_numpy()

    def decode(self, h, x_level=0, h_level=None):
        if h_level is None:
            h_level = self.num_hidden_layers()
        y = self._decoder(h)
        y = y.reshape(y.size(0), -1, self.window_size)
        sigma_x = torch.exp(self.log_sigma_x)
        if y.dim() == 3:
            sigma_x = sigma_x.unsqueeze(2)
        y = torch.mul(y, sigma_x)
        return y

    def forward(self, x):
        x = self.prepare_data(x)
        return self.decode(self.encode(x))

    def energy(self, x, h):
        vis = torch.sum(torch.div(torch.square(x - self.bx), (2 * torch.square(torch.exp(self.log_sigma_x)))), dim=1)
        hid = torch.matmul(h, self._encoder[-2].bias)
        xWh = torch.sum(torch.matmul(x, self._encoder[-2].weight.T) * h, dim=1)
        return vis - hid - xWh

    def free_energy(self, x, level=1):
        bx = self._decoder[-level][f"linear_h2v{level}"]
        vis = torch.sum(torch.div(torch.square(x - bx), (2 * torch.square(torch.exp(self.log_sigma_x)))), dim=1)
        x = torch.div(x, torch.exp(self.log_sigma_x))
        return vis - torch.sum(self.free_energy_components(x), dim=1)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logp = -self.free_energy_components(x).sum()
        grad = torch.autograd.grad(logp, x, create_graph=True)[0] # Create graph True to allow later backprop
        return grad

    def encode_ordinal(self, x, columns, order=None):
        return encode_ordinal(x, columns, order)

    def encode_nominal(self, x, columns=None):
        return encode_nominal(x, columns)

    def dsm_loss(self, x, v, sigma=0.1):
        """DSM loss from
        A Connection Between Score Matching
            and Denoising Autoencoders
        The loss is computed as
        x_ = x + v   # noisy samples
        s = -dE(x_)/dx_
        loss = 1/2*||s + (x-x_)/sigma^2||^2
        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises
            sigma (float, optional): noise scale. Defaults to 0.1.

        Returns:
            DSM loss
        """
        x = x.requires_grad_()
        v = v * sigma
        x_ = x + v
        s = self.score(x_)
        loss = torch.norm(s + v / (sigma ** 2), dim=-1) ** 2
        loss = loss.mean() / 2.
        return loss

    def num_x(self):
        l = self._encoder[0]
        return l.in_features

    def num_h(self):
        l = self._decoder[0]
        return l.in_features

    def num_hidden_layers(self):
        return len(self._encoder)

    # def sample_h(self, x):
    #     p_h_given_v = self.encode(x)
    #     return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_h(self, h):
        return torch.bernoulli(h)

    # def sample_x(self, h):
    #     p_v_given_h = self.decode(h)
    #     return p_v_given_h, torch.normal(p_v_given_h, std=torch.exp(self.log_sigma_x))
    def sample_x(self, x):
        return torch.normal(x, std=torch.exp(self.log_sigma_x))

    def generate(self, num_examples, num_steps=10):
        x = torch.randn([num_examples, self.num_x()])
        for k in range(num_steps):
            ph, h = self.sample_h(x)
            px, x = self.sample_x(h)
        return x

    def contrastive_divergence(self, v0, h0, vk, hk):
        # return self.free_energy(v0) - self.free_energy(vk)
        sparsity_penalty = self.sparsity_weight * torch.sum((self.sparsity_target - hk) ** 2)
        return self.energy(v0, h0) - self.energy(vk, hk) + sparsity_penalty

    def recon(self, v, round=False, x_level=0, h_level=None):
        # if self.variant == 'dsebm':
        #     v = v.requires_grad_()
        #     logp = -self.free_energy_components(v).sum()
        #     return torch.autograd.grad(logp, v, create_graph=True)[0]
        if h_level is None:
            h_level = self.num_hidden_layers()
        h = self.encode(v, x_level=x_level, h_level=h_level)
        if round:
            h = torch.round(h)
        r = self.decode(h, x_level, h_level)
        return r, h

    def pretrain_dbn(self, train_data, valid_data, **kwargs):
        for layer in range(1, self.num_hidden_layers() + 1):
            self.train_rbm(train_data, valid_data, level=layer, **kwargs)

    def train_rbm(self, train_data, valid_data, level=1, learning_rule='cd', min_epoch=0, max_epoch=10, weight_decay=0.,
                  batch_size=128, shuffle=True, num_k=1, verbose=True, early_stopping=False, early_stopping_patience=3,
                  use_probability_last_x_update=False):
        if self.log_mlflow:
            mlflow.log_param("min_epoch", min_epoch)
            mlflow.log_param("max_epoch", max_epoch)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("shuffle", shuffle)
            mlflow.log_param("weight_decay", weight_decay)
            mlflow.log_param("early_stopping_patience", early_stopping_patience)
            mlflow.log_param("early_stopping", early_stopping)

        train_data = self.prepare_data(train_data, update_mean_std=level == 1)
        valid_data = self.prepare_data(valid_data)

        train_data = torch.vstack(train_data)
        valid_data = torch.vstack(valid_data)

        if valid_data is not None:
            self.valid_curve.append(self._get_progress_rbm(valid_data, level=level))

        self.learning_curve.append(self._get_progress_rbm(train_data, level=level))

        data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
        opt = torch.optim.RMSprop(self.parameters(), weight_decay=weight_decay)
        t_start = time.time()
        for epoch in range(1, max_epoch + 1):
            for i, d in enumerate(data_loader):
                xk = d
                if self.variant in ['dsebm', 'dae']:
                    xk = torch.normal(mean=xk, std=torch.exp(self.log_sigma_x))
                x0 = d
                if learning_rule == 'cd':
                    with torch.no_grad():
                        eh0 = self.encode(x0)
                        h0 = self.sample_h(eh0)
                        hk = h0
                        for k in range(num_k):
                            exk = self.decode(hk)
                            xk = self.sample_x(exk)
                            ehk = self.encode(xk)
                            hk = self.sample_h(ehk)

                    if use_probability_last_x_update:
                        cd = torch.mean(self.contrastive_divergence(x0, h0, exk, ehk))
                    else:
                        cd = torch.mean(self.contrastive_divergence(x0, h0, xk, ehk))
                    opt.zero_grad()
                    cd.backward()
                    opt.step()
                elif learning_rule == 'sm':
                    r = self.recon(xk)
                    loss = (r - x0).pow(2).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                elif learning_rule == 'dsm':
                    x0noise = torch.normal(torch.zeros_like(x0))
                    loss = self.dsm_loss(x0, x0noise)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            with torch.no_grad():
                progress = self._get_progress_rbm(train_data)
                if verbose:
                    print(f'\n############### Epoch {epoch} ###############')
                    print('Train: ')
                    pprint.pp(progress)
                self.learning_curve.append(progress)
            if valid_data is not None:
                with torch.no_grad():
                    progress = self._get_progress_rbm(valid_data)
                    self.valid_curve.append(progress)
                if verbose:
                    print('Valid: ')
                    pprint.pp(progress)

                if early_stopping and epoch > min_epoch and epoch > early_stopping_patience:
                    valid_metrics = np.array([v['MSE'] for v in self.valid_curve[-early_stopping_patience - 1:]])
                    if np.all(valid_metrics[1:] > valid_metrics[0]):
                        print('Early stop after valid metrics: ', valid_metrics)
                        break

            if self.is_sigma_learnable:
                print(torch.exp(self.log_sigma_x))
        self.eval()
        self.num_epoch = epoch
        print('Training finished after ', timedelta(seconds=time.time() - t_start))

    def mse(self, x, r, per_point=False):
        if per_point:
            return torch.mean(torch.square(x - r), dim=1)
        else:
            return torch.mean(torch.square(x - r))

    def _get_progress_learn(self, d):
        r, h = self.recon(d)
        progress = dict(MSE=self.mse(d, r).item(),
                        Sparsity=self.sparsity(h).item())
        if False:
            valid_energy = torch.mean(self.free_energy(valid_data)).item()
            progress['Energy'] = valid_energy
        return progress

    def _get_progress_rbm(self, d, level):
        r, h = self.recon(d, x_level=level-1, h_level=level)
        e = self.free_energy(d)
        progress = dict(MSE=self.mse(d, r).item(),
                        Sparsity=self.sparsity(h).item(),
                        Energy=torch.mean(e).item())
        return progress

    def _is_set_mean_std(self):
        return self._std is not None and self._mean is not None

    def learn_denta_network(self, train_data, valid_data, max_epoch=10, min_epoch=0,
                            weight_decay=0., batch_size=128, shuffle=True, verbose=True, early_stopping=False,
                            early_stopping_patience=3, round_latent_during_learning=False):
        if self.log_mlflow:
            mlflow.log_param("min_epoch", min_epoch)
            mlflow.log_param("max_epoch", max_epoch)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("shuffle", shuffle)
            mlflow.log_param("weight_decay", weight_decay)
            mlflow.log_param("early_stopping_patience", early_stopping_patience)
            mlflow.log_param("early_stopping", early_stopping)
            mlflow.log_param('round_latent_during_learning', round_latent_during_learning)

        train_data = self.prepare_data(train_data, update_mean_std=not self._is_set_mean_std())
        valid_data = self.prepare_data(valid_data)

        train_data = torch.vstack(train_data)
        valid_data = torch.vstack(valid_data)

        if valid_data is not None:
            self.valid_curve.append(self._get_progress_learn(valid_data))

        train_data_loaded = train_data
        self.learning_curve.append(self._get_progress_learn(train_data_loaded))

        data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
        opt = torch.optim.RMSprop(self.parameters(), weight_decay=weight_decay)
        t_start = time.time()
        for epoch in range(1, max_epoch + 1):
            for i, d in enumerate(data_loader):
                xk = d
                if self.variant in ['dsebm', 'dae']:
                    xk = torch.normal(mean=xk, std=torch.exp(self.log_sigma_x))
                x0 = d
                r, h = self.recon(xk, round=round_latent_during_learning)
                loss = self.mse(x0, r) + self.sparsity_weight * self.sparsity_loss(h)
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                progress = self._get_progress_learn(train_data_loaded)
                if verbose:
                    print(f'\n############### Epoch {epoch} ###############')
                    print('Train: ')
                    pprint.pp(progress)
                self.learning_curve.append(progress)
            if valid_data is not None:
                with torch.no_grad():
                    progress = self._get_progress_learn(valid_data)
                    self.valid_curve.append(progress)
                if verbose:
                    print('Valid: ')
                    pprint.pp(progress)

                if early_stopping and epoch > min_epoch and epoch > early_stopping_patience:
                    valid_metrics = np.array([v['MSE'] for v in self.valid_curve[-early_stopping_patience-1:]])
                    if np.all(valid_metrics[1:] > valid_metrics[0]):
                        print('Early stop after valid metrics: ', valid_metrics)
                        break

            if self.is_sigma_learnable:
                print(torch.exp(self.log_sigma_x))
        self.eval()
        self.num_epoch = epoch

        if self.log_mlflow:
            mlflow.log_metrics(self.valid_curve[-1])
            mlflow.log_metric("num_epoch", self.num_epoch)
            run_id = mlflow.active_run().info.run_id
            mlflow.log_figure(self.plot_learning_curve(), f'figures/learning_curve_denta_{run_id}.html')
            mlflow.log_figure(self.plot_activation_probabilities(valid_data, prepare_data=False),
                              f'figures/activations_hist_denta_{run_id}.html')
            mlflow.log_figure(self.plot_frequency_of_latent_combinations(valid_data, prepare_data=False),
                              f'figures/latent_frequencies_denta_{run_id}.html')

        print('Training finished after ', timedelta(seconds=time.time() - t_start))

    def learn_latent_automaton(self, train_data, valid_data):
        sig_names = [f'h{i + 1}' for i in range(self.num_h())]
        h = []
        for d in train_data:
            d = self.prepare_data(d)
            hh = torch.round(self.encode(d)).cpu().detach().numpy()
            hh = pd.DataFrame(hh, columns=sig_names, index=np.arange(hh.shape[0]))
            hh.reset_index(inplace=True)
            h.append(hh)
        a = automata_learn.simple_learn_from_signal_vectors(h, drop_no_changes=True, sig_names=sig_names)
        self._G = a._G

        if self.log_mlflow:
            mlflow.log_metric("num_modes", self.num_modes)
            mlflow.log_figure(self.view_plotly(), f'figures/denta_automaton_{mlflow.active_run().info.run_id}.html')

    # def sparsity_loss(self, h):
    #     sparsity_penalty = self.sparsity_weight * torch.sum((self.sparsity_target - h) ** 2)
    #     return sparsity_penalty
    def sparsity(self, activations, per_point=False):
        if per_point:
            return torch.mean(activations, dim=1)
        else:
            return torch.mean(activations)

    def sparsity_loss(self, activations):
        # Mean activation per neuron
        mean_activations = self.sparsity(activations, per_point=True)

        # KL divergence between the target sparsity level and the means
        kl_div = self.sparsity_target * torch.log(self.sparsity_target / mean_activations) + \
                 (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - mean_activations))
        kl_div = torch.mean(kl_div)  # Sum over all neurons

        return kl_div

    def recon_error(self, data, input=None, per_point=False, round=False):
        if input is None:
            input = data
        recon, _ = self.recon(input, round=round)
        if per_point:
            dim = (1, 2)
        else:
            dim = None
        squared_error = torch.mean(torch.square(data - recon), dim=dim)
        return squared_error

    def anomaly_score(self, s):
        # d = d[:]
        # t, s, = d['time'], d
        # if self.variant in ['gbrbm']:
        #     score_in_time = self.free_energy(s)
        # else:
        #     score_in_time = self.recon_error(s)
        # score = score_in_time.cpu().detach().numpy()

        s = self.prepare_data(s)
        score_in_time = self.recon_error(s, per_point=True).cpu().detach().numpy()
        return score_in_time

    def calculate_ad_threshold(self, d, quantile=0.95):
        scores = self.anomaly_score(d)
        self.threshold = np.sort(scores)[int((len(scores) - 1) * quantile)]
        if self.log_mlflow:
            mlflow.log_metric('threshold', self.threshold)

    def plot_activation_probabilities(self, d, prepare_data=True):
        if prepare_data:
            d = self.prepare_data(d)
        h = self.encode(d).cpu().detach().numpy()
        fig = go.Figure(data=[go.Histogram(x=h.reshape(-1), name="Probability", histnorm="percent", nbinsx=20)],
                        layout_title="Histogram of activation probabilities")
        return fig

    def plot_error_histogram(self, d, v=None):
        s = self.anomaly_score(d)
        fig = go.Figure(data=[go.Histogram(x=s, name="Anomaly score", histnorm="density")], layout_title="Histogram of scores")
        if v is not None:
            s = self.anomaly_score(v)
            fig.add_trace(go.Histogram(x=s, name="Anomaly score - validation set", histnorm="density"))
        if self.threshold is not None:
            fig.add_vline(x=self.threshold, line_width=2, line_dash="dash", line_color="red")
        return fig

    def anomaly_detection(self, s):
        s = self.prepare_data(s)
        score_in_time = self.recon_error(s, per_point=True).cpu().detach().numpy()
        return (score_in_time > self.threshold).astype(float), score_in_time

    def plot_learning_curve(self):
        return tools.plot_data([pd.DataFrame(self.learning_curve), pd.DataFrame(self.valid_curve)],
                             title='Learning curve', names=['Train', 'Valid'], xaxis_title='Epoch')

    # Transforms the p(v) into mixture of Gaussians and returns the weight, mean and sigma for each Gaussian component as
    # well the corresponding hidden states.This function is for use with very small models.Otherwize it will last forever
    def gmm_model(self):
        def gbrbm_h2v(type, h, W, bv, sigma):
            x = np.matmul(np.atleast_2d(h), W)
            if type == 'gbrbm':
                x *= sigma
            return x + bv
        sigma = np.exp(self.log_sigma_x.detach().numpy())
        bv = self.bx.detach().numpy()
        bh = self._encoder[-2].bias.detach().numpy()
        W = self._encoder[-2].weight.detach().numpy()

        num_components = 2 ** self.num_h()

        if sigma.size == 1:
            sigma = np.repeat(sigma, self.num_x(), axis=1)
            sigma = sigma[None]

        gmm_sigmas = np.repeat(sigma, num_components, axis=0)

        # Initialize
        weights = np.zeros((num_components, 1))
        means = np.zeros((num_components, self.num_x()))
        hid_states = np.zeros((num_components, self.num_h()))

        phi0 = np.prod(np.sqrt(2 * np.pi) * sigma)

        weights[0] = phi0
        means[0, :] = bv
        for i in range(1, num_components):
            hs = list(bin(i)[2:])
            hid_states[i, -len(hs):] = hs
            hs = hid_states[i, :]
            # Calc means
            mean = gbrbm_h2v(self.variant, hs, W, bv, sigma)
            means[i, :] = mean

            # Calc phi
            phi = (np.sum(mean ** 2 / (2 * sigma ** 2)) - np.sum(bv ** 2 / (2 * sigma ** 2)))
            phi = np.sum(phi) + np.sum(bh * hs)
            phi = phi0 * np.exp(phi)

            weights[i] = phi

        # Normalize weights
        Z = sum(weights)
        weights = weights / Z
        return weights, means, gmm_sigmas, hid_states, Z

    def plot_frequency_of_latent_combinations(self, d, prepare_data=True):
        h = self.predict_discrete_mode(d, prepare_data)
        if type(h) is list:
            h = np.concatenate(h, axis=0)
        fig = go.Figure(data=[go.Histogram(x=h, name="Frequency", histnorm="percent")],
                        layout_title="Frequency of latent combinations")
        return fig

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
                               contours=dict(coloring='lines'), name="Free energy", ncontours=ncontours, showlegend=True, showscale=False)
            fig.add_trace(trace)

        if data is not None:
            if data_limit is not None and data.shape[0] > data_limit:
                data = data.sample(data_limit)
            fig.add_trace(vis.plot2d(data[data.columns[0]], data[data.columns[1]], name='Data',
                                     marker=dict(size=3, opacity=0.2, color='MediumPurple')))
            if show_reconstructions:
                recon, _ = self.recon(torch.Tensor(data.values)).detach().numpy()
                fig.add_trace(vis.plot2d(recon[:,0], recon[:, 1], name='Reconstruction',
                                         marker=dict(size=3, opacity=0.2, color='limegreen')))
        if samples is not None:
            fig.add_trace(vis.plot2d(samples[:, 0], samples[:, 1], name='Samples',
                                     marker=dict(size=3, opacity=0.2, color='darkgreen')))

        if show_axis_titles:
            fig.update_layout(
                xaxis_title="$x_1$",
                yaxis_title="$x_2$",
            )
        if plot_code_positions:
            num_h = self.num_h()
            num_v = self.num_x()
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

    def find_optimal_threshold_for_f1(self, data, search_every=1, plot=False):
        scores_unsorted = self.anomaly_score(data[:])
        sort_ind = np.argsort(scores_unsorted)
        sort_ind = sort_ind[0::search_every]
        thresholds = scores_unsorted[sort_ind]
        labels_unsorted = data[:]['label'].cpu().detach().numpy()

        f1_scores = [f1_score(labels_unsorted != 0, scores_unsorted > th) for th in thresholds]
        opt_ind = np.argmax(f1_scores)
        opt_th = thresholds[opt_ind]
        max_f1 = f1_scores[opt_ind]
        if plot:
            fig = vis.plot2d(np.arange(0, scores_unsorted.shape[0]), scores_unsorted, return_figure=True)
            fig.add_trace(vis.plot2d(np.arange(0, labels_unsorted.shape[0]), labels_unsorted))
            fig.show()
            vis.plot2d(thresholds, f1_scores, return_figure=True).show()
        return opt_th, max_f1

    def get_auroc(self, data):
        scores = self.anomaly_score(data[:])
        labels = data[:]['label'].cpu().detach().numpy()
        labels = labels != 0
        return roc_auc_score(labels, scores)

    def compute_purity(cluster_assignments, class_assignments):
        """Computes the purity between cluster and class assignments.
        Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

        Args:
            cluster_assignments (list): List of cluster assignments for every point.
            class_assignments (list): List of class assignments for every point.

        Returns:
            float: The purity value.
        """

        assert len(cluster_assignments) == len(class_assignments)

        num_samples = len(cluster_assignments)
        num_clusters = len(np.unique(cluster_assignments))
        num_classes = len(np.unique(class_assignments))

        cluster_class_counts = {cluster_: {class_: 0 for class_ in np.unique(class_assignments)}
                                for cluster_ in np.unique(cluster_assignments)}

        for cluster_, class_ in zip(cluster_assignments, class_assignments):
            cluster_class_counts[cluster_][class_] += 1

        total_intersection = sum(
            [max(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()])

        purity = total_intersection / num_samples

        return purity

if __name__ == '__main__':
    # Preprocess the data
    num_sequences = 1  # Number of sequences
    sequence_length = 60  # Length of each sequence
    num_features = 3  # Number of features per time step

    # Generate synthetic data
    data = torch.randn(sequence_length, num_features, device='cuda')
    valid = torch.randn(sequence_length, num_features, device='cuda')
    timestamp = torch.arange(sequence_length)
    valid_states = torch.randint(0, 10, (sequence_length, 1))

    # Hyperparameters
    sparsity_target = 1 / 3
    sparsity_weight = 0
    sigma = 0.3
    max_epoch = 2
    window_size = 5
    window_step = 1

    # Parameters
    latent_dim = 7
    num_hidden_layers = 3

    # Train model
    model = DENTA(data.shape[1], latent_dim, sigma=sigma, sigma_learnable=False, sparsity_weight=sparsity_weight,
                  num_hidden_layers=num_hidden_layers, window_size=window_size, window_step=window_step,
                  sparsity_target=sparsity_target, use_derivatives=0, device='cuda')
    model.pretrain_dbn([data], [valid])
    # model.learn_denta_network([data], [valid], max_epoch=max_epoch, verbose=False)
    model.learn_latent_automaton([data], [valid])
    model.view_plotly().show()
    model.plot_learning_curve().show('browser')

    valid_mode_prediction = model.predict_discrete_mode([valid])
    # model.plot_frequency_of_latent_combinations([data]).show()
    model.plot_activation_probabilities(valid).show()
    #model.plot_discretization(timestamp, valid_states.cpu(), valid_mode_prediction[0], data.cpu(), timestamp.cpu()).show()


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def train_model(self, X, y, num_epochs=1000, lr=0.01):
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.SGD(self.parameters(), lr=lr)  # Stochastic Gradient Descent with learning rate of 0.01

        for epoch in range(num_epochs):
            # Forward pass: Compute predicted y by passing X to the model
            y_pred = self(X)

            # Compute and print loss
            loss = criterion(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')