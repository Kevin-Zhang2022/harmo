import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from numpy import (pi, cos, exp, log10, sqrt)
import snntorch as snn
import pyfilterbank.gammatone as gt
import copy

class innerhaircell(nn.Module):
    def __init__(
            self,
    ):
        super(innerhaircell, self).__init__()

    def forward(self, x):
        out = (x > 0) * x
        return out

class auditorynerve(nn.Module):
    def __init__(self,
                 uth
    ):
        super(auditorynerve, self).__init__()
        self.uth = uth
        self.sleaky = snn.Leaky(beta=0.95, threshold=uth)

    def forward(self, inp):
        mem = self.sleaky.init_leaky()
        inp = torch.tensor(inp, dtype=torch.float32)
        spikes = []
        for t in range(inp.size(1)):
            x = inp[:, t]
            spk, mem = self.sleaky(x, mem)
            spikes.append(spk)
        spikes = torch.stack(spikes, dim=1)
        return spikes.detach().numpy()


class stellate(nn.Module):
    def __init__(
            self,
            channels=224,
            thr=0.8,
            sigma=10,
    ):
        super(stellate, self).__init__()
        self.channels = channels
        self.w = torch.tensor(self.get_w(sigma=sigma), dtype=torch.float32)
        self.sleaky = snn.Leaky(beta=0.95, threshold=thr)
        # plt.plot(self.w_1[20,:])

    def forward(self, inp):
        inp = torch.tensor(inp, dtype=torch.float32)
        spks = []
        mem = self.sleaky.init_leaky()
        for t in range(inp.shape[1]):
            x = inp[:, t]
            I = torch.matmul(self.w, x.unsqueeze(-1)).squeeze(-1)
            spk, mem = self.sleaky(x-I, mem)
            spks.append(spk)
        spks = torch.stack(spks, dim=1)
        return spks.detach().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(inp)
        # plt.imshow(spks)

    def get_w(self, sigma, radius=6):
        x = np.linspace(0, self.channels-1, self.channels)
        out = []
        for i in range(self.channels):
            p = np.exp(-0.5 * ((x - i) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            out.append(p)
            # # 只保留均值左右6个数的数值，其余置为0
            # mask = np.zeros_like(p)
            # start = max(0, i - radius)
            # end = min(self.channels, i + radius+1)
            # mask[start:end] = 1
            # p *= mask
        out = np.stack(out)
        return out


class cochlea(nn.Module):
    def __init__(self,
                 channels,
                 window,
                 frequency_range=(20, 20000),
                 bandwidth_factor=0.05,
                 sample_rate=44100,
                 ):
        super(cochlea, self).__init__()
        start_band = gt.hertz_to_erbscale(frequency_range[0])
        end_band = gt.hertz_to_erbscale(frequency_range[1])
        self.gtfb = gt.GammatoneFilterbank(samplerate=sample_rate, bandwidth_factor=bandwidth_factor,
                                           startband=start_band, endband=end_band, channels=channels)
        self.cf = self.gtfb.centerfrequencies
        self.window = window
        self.hop = window

    def forward(self, inp):
        out = []

        results = self.gtfb.analyze(inp)
        for (band, state) in results:
            out.append(np.real(band))
        out = np.array(out)

        out = (out > 0)*out
        temp = []
        for t in range(0, out.shape[1]-self.window, self.hop):
            temp.append(np.mean(out[:, t:t+self.window], axis=1))
        out = np.stack(temp, axis=1)
        # np.stach(out.append(temp))

        return out


class harmolearn(nn.Module):
    def __init__(
            self,
            channels=224,
            window=16,
            w_range=(-1, 3)
    ):
        super(harmolearn, self).__init__()
        self.channels = channels
        self.window = window
        self.w = np.ones(channels)
        self.w_range = w_range
        self.sleaky = snn.Leaky(beta=0.95, threshold=1)
        # plt.plot(self.w_1[20,:])

    def forward(self, inp, **kwargs):
        out = copy.deepcopy(inp)
        # pad_size = self.window//2
        # inp_padded = np.pad(inp, pad_width=((0, 0), (pad_size, pad_size)), mode='constant', constant_values=0)
        # inp_padded = torch.nn.functional.pad(inp, (pad_size, pad_size, 0, 0), mode='constant', value=0)

        mem = self.sleaky.init_leaky()
        cc_rec = []
        w_rec = []
        for t in range(self.window, inp.shape[1]):
            B = inp[:, t-self.window:t]
            cc_mat = B @ B.T
            # cc_mat = self.calculate_xnor_matrix(B.astype(np.bool_))
            cc_vec = np.sum(cc_mat, axis=-1).astype(np.float_)
            cc_vec /= (1e-5+np.max(cc_vec))
            cc_rec.append(cc_vec)
            self.w = 0.5*self.w + 0.5*cc_vec
            w = self.w.clip(self.w_range[0], self.w_range[1])
            w_rec.append(w)
            x = inp[:, t]
            spk, mem = self.sleaky(torch.tensor(self.w*x), mem)
            out[:, t] = spk
        if len(kwargs):
            kwargs['w_rec'] = np.stack(w_rec, axis=-1)
            kwargs['h_rec'] = np.stack(cc_rec, axis=-1)
            return out, kwargs['w_rec'], kwargs['h_rec']
        else:
            return out
        # self.window =
        # self.w_range = (-1.5, 1.5)
        # fig, axes = plt.subplots(2, 2, figsize=(8,8))
        # axes[0,0].imshow(inp)
        # axes[0, 0].set_title('inp')
        # axes[0, 1].imshow(out)
        # axes[0, 1].set_title('out')
        # axes[1, 0].imshow(cc_rec)
        # axes[1, 0].set_title('cc_rec')
        # axes[1, 1].imshow(w_rec)
        # axes[1, 1].set_title('w_rec')






