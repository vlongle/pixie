import numpy
import torch.nn as nn
import torch.nn.functional as F
import pytorch_wavelets.dwt.lowlevel as lowlevel
from pytorch_wavelets.dwt.transform2d import DWTForward
import torch
from torch.autograd import Function
import time
import pywt

### modify from https://github.com/fbcotter/pytorch_wavelets/blob/master/pytorch_wavelets/dwt/lowlevel.py
from models.module.dwt_utils import prep_filt_sfb3d, prep_filt_afb3d


class DWTInverse3d_Laplacian(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col


        # Prepare the filters
        filts = prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_dep', filts[0])
        self.register_buffer('g1_dep', filts[1])
        self.register_buffer('g0_col', filts[2])
        self.register_buffer('g1_col', filts[3])
        self.register_buffer('g0_row', filts[4])
        self.register_buffer('g1_row', filts[5])
        self.J = J
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:

            # 'Unpad' added dimensions
            ll_diff = SFB3D_Laplacian.apply(
                ll, self.g0_dep, self.g0_col, self.g0_row, mode)
            if ll_diff.shape[-3] > h.shape[-3]:
                ll_diff = ll_diff[...,:-1, :, :]
            if ll_diff.shape[-2] > h.shape[-2]:
                ll_diff = ll_diff[...,:-1,:]
            if ll_diff.shape[-1] > h.shape[-1]:
                ll_diff = ll_diff[...,:-1]

            ll = ll_diff + h

        return ll

class DWTInverse3d(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col


        # Prepare the filters
        filts = prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_dep', filts[0])
        self.register_buffer('g1_dep', filts[1])
        self.register_buffer('g0_col', filts[2])
        self.register_buffer('g1_col', filts[3])
        self.register_buffer('g0_row', filts[4])
        self.register_buffer('g1_row', filts[5])
        self.J = J
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 7, ll.shape[-3], ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-3] > h.shape[-3]:
                ll = ll[...,:-1, :, :]
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            ll = SFB3D.apply(
                ll, h, self.g0_dep, self.g1_dep, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
        return ll

class DWTForward3d_Laplacian(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
            h0_dep, h1_dep = h0_col, h1_col


        # Prepare the filters
        filts = prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_dep', filts[0])
        self.register_buffer('h1_dep', filts[1])
        self.register_buffer('h0_col', filts[2])
        self.register_buffer('h1_col', filts[3])
        self.register_buffer('h0_row', filts[4])
        self.register_buffer('h1_row', filts[5])
        self.J = J
        self.mode = mode

        ## Need for inverse
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col


        # Prepare the filters
        filts = prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_dep', filts[0])
        self.register_buffer('g1_dep', filts[1])
        self.register_buffer('g0_col', filts[2])
        self.register_buffer('g1_col', filts[3])
        self.register_buffer('g0_row', filts[4])
        self.register_buffer('g1_row', filts[5])

    def forward(self, x):
        """ Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll_new = AFB3D_Laplacian.apply(
                ll, self.h0_dep, self.h0_col, self.h0_row, mode)
            reversed_ll = SFB3D_Laplacian.apply(ll_new, self.g0_dep, self.g0_col, self.g0_row, mode)
            if ll.shape[-1] < reversed_ll.shape[-1]:
                reversed_ll = reversed_ll[..., :-1]
            if ll.shape[-2] < reversed_ll.shape[-2]:
                reversed_ll = reversed_ll[..., :-1, :]
            if ll.shape[-3] < reversed_ll.shape[-3]:
                reversed_ll = reversed_ll[..., :-1, :, :]
            yh.append(ll - reversed_ll)
            ll = ll_new

        return ll, yh

class DWTForward3d(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
            h0_dep, h1_dep = h0_col, h1_col


        # Prepare the filters
        filts = prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_dep', filts[0])
        self.register_buffer('h1_dep', filts[1])
        self.register_buffer('h0_col', filts[2])
        self.register_buffer('h1_col', filts[3])
        self.register_buffer('h0_row', filts[4])
        self.register_buffer('h1_row', filts[5])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = AFB3D.apply(
                ll, self.h0_dep, self.h1_dep, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh


def afb1d_laplacian(x, h0, mode='zero', dim = -1):
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 5
    s = [1, 1, 1]
    s[d - 2] = 2
    s = tuple(s)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1, 1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)

    h = torch.cat([h0] * C, dim=0)

    assert mode in ['zero']

    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)
    p = 2 * (outsize - 1) - N + L

    # Sadly, pytorch only allows for same padding before and after, if
    # we need to do more padding after for odd length signals, have to
    # prepad

    padding_mode = None
    if mode == 'zero':
        padding_mode = 'zero'
    else:
        raise Exception("Unknown mode")

    if p % 2 == 1:
        pad = [0, 0, 0, 0, 0, 0]
        pad[(4 - d) * 2 + 1] = 1
        pad = tuple(pad)
        function_padding = 'constant'
        x = F.pad(x, pad, mode = function_padding)
    pad = [0, 0, 0]
    pad[d - 2] = p // 2
    pad = tuple(pad)
    # Calculate the high and lowpass
    if padding_mode == 'zero':
        lo = F.conv3d(x, h, padding=pad, stride=s, groups=C)
    else:
        pad_new = [ pad[2 - i // 2] for i in range(6)]
        x = F.pad(x, pad_new, mode = padding_mode)
        lo = F.conv3d(x, h, stride=s, groups=C)

    return lo

def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image
    Inputs:
        x (tensor): 5D input with the last two dimensions the spatial input
        h0 (tensor): 5D input for the lowpass filter. Should have shape (1, 1,
            h, 1, 1) or (1, 1, 1, w, 1) or (1, 1, 1, 1, d)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w, 1) or (1, 1, 1, 1, d)
        mode (str): padding method can only be zero
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).
    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """

    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 5
    s = [1, 1, 1]
    s[d-2] = 2
    s = tuple(s)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    assert mode in ['zero']

    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)
    p = 2 * (outsize - 1) - N + L

    # Sadly, pytorch only allows for same padding before and after, if
    # we need to do more padding after for odd length signals, have to
    # prepad
    padding_mode = None
    if mode == 'zero':
        padding_mode = 'zero'
    else:
        raise Exception("Unknown mode")

    if p % 2 == 1:
        pad = [0, 0, 0, 0, 0, 0]
        pad[(4 - d)*2+1] = 1
        pad = tuple(pad)
        function_padding = 'constant'
        x = F.pad(x, pad, mode = function_padding)
    pad = [0, 0, 0]
    pad[d - 2] = p // 2
    pad = tuple(pad)
    # Calculate the high and lowpass
    if padding_mode == 'zero':
        lohi = F.conv3d(x, h, padding=pad, stride=s, groups=C)
    else:
        pad_new = [ pad[2 - i // 2] for i in range(6)]
        x = F.pad(x, pad_new, mode = padding_mode)
        lohi = F.conv3d(x, h, stride=s, groups=C)

    return lohi

def sfb1d_laplacian(lo, g0, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 5
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]

    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)

    s = [1, 1, 1]
    s[d-2] = 2

    g0 = torch.cat([g0]*C,dim=0)

    assert mode in ['zero']

    pad = [0, 0, 0]
    pad[d-2] = L - 2
    pad = tuple(pad)

    y = F.conv_transpose3d(lo, g0, stride=s, padding=pad, groups=C)

    return y

def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 5
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]

    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = [1, 1, 1]
    s[d-2] = 2

    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)

    assert mode in ['zero']

    pad = [0, 0, 0]
    pad[d-2] = L - 2
    pad = tuple(pad)

    y = F.conv_transpose3d(lo, g0, stride=s, padding=pad, groups=C) + \
        F.conv_transpose3d(hi, g1, stride=s, padding=pad, groups=C)

    return y

class SFB3D_Laplacian(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    @staticmethod
    def forward(ctx, low, g0_dep, g0_col, g0_row, mode):
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0_dep, g0_col, g0_row)
        lll = low
        ## first level
        ll = sfb1d_laplacian(lll, g0_dep, mode=mode, dim=2)

        ## second level
        l = sfb1d_laplacian(ll, g0_col, mode=mode, dim=3)

        ## last level
        y = sfb1d_laplacian(l, g0_row, mode=mode, dim=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_dep, g0_col, g0_row = ctx.saved_tensors
            dx = afb1d_laplacian(dy, g0_row, mode=mode, dim=4)
            dx = afb1d_laplacian(dx, g0_col, mode=mode, dim=3)
            dx = afb1d_laplacian(dx, g0_dep, mode = mode, dim = 2)
            s = dx.shape
            dlow = dx.reshape(s[0], -1, s[-3], s[-2], s[-1])
        return dlow, None, None, None, None, None

class SFB3D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    @staticmethod
    def forward(ctx, low, highs, g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row, mode):
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        hll, lhl, hhl, llh, hlh, lhh, hhh = torch.unbind(highs, dim=2)
        lll = low
        ## first level
        ll = sfb1d(lll, hll, g0_dep, g1_dep, mode=mode, dim=2)
        hl = sfb1d(lhl, hhl, g0_dep, g1_dep, mode=mode, dim=2)
        lh = sfb1d(llh, hlh, g0_dep, g1_dep, mode=mode, dim=2)
        hh = sfb1d(lhh, hhh, g0_dep, g1_dep, mode=mode, dim=2)

        ## second level
        l = sfb1d(ll, hl, g0_col, g1_col, mode=mode, dim=3)
        h = sfb1d(lh, hh, g0_col, g1_col, mode=mode, dim=3)

        ## last level
        y = sfb1d(l, h, g0_row, g1_row, mode=mode, dim=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row = ctx.saved_tensors
            dx = afb1d(dy, g0_row, g1_row, mode=mode, dim=4)
            dx = afb1d(dx, g0_col, g1_col, mode=mode, dim=3)
            dx = afb1d(dx, g0_dep, g1_dep, mode = mode, dim = 2)
            s = dx.shape
            dx = dx.reshape(s[0], -1, 8, s[-3], s[-2], s[-1])
            dlow = dx[:,:,0].contiguous()
            dhigh = dx[:,:,1:].contiguous()
        return dlow, dhigh, None, None, None, None, None, None, None

class AFB3D_Laplacian(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        h0_dep: depth lowpass
        h1_dep: depth highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W, D)
    """

    @staticmethod
    def forward(ctx, x, h0_dep, h0_col, h0_row, mode):
        ctx.save_for_backward(h0_dep, h0_col, h0_row)
        ctx.shape = x.shape[-3:]
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        lohi_dim_last = afb1d_laplacian(x, h0_row, mode=mode, dim=4)
        lohi_dim_last_2 = afb1d_laplacian(lohi_dim_last, h0_col, mode=mode, dim=3)
        y = afb1d_laplacian(lohi_dim_last_2, h0_dep, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 1, s[-3], s[-2], s[-1])
        low = y[:, :, 0].contiguous()
        return low

    @staticmethod
    def backward(ctx, lll):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_dep, h0_row, h0_col = ctx.saved_tensors

            ## first level
            ll = sfb1d_laplacian(lll, h0_dep, mode=mode, dim=2)

            ## second level
            l = sfb1d_laplacian(ll, h0_col, mode=mode, dim=3)

            ## last level
            dx = sfb1d_laplacian(l, h0_row, mode = mode, dim = 4)

            if dx.shape[-3] > ctx.shape[-3]:
                dx = dx[:, :, :ctx.shape[-3]]
            if dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:, :, :, :ctx.shape[-2]]
            if dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:, :, :, :, :ctx.shape[-1]]

        return dx, None, None, None, None

class AFB3D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        h0_dep: depth lowpass
        h1_dep: depth highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W, D)
    """

    @staticmethod
    def forward(ctx, x, h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row, mode):
        ctx.save_for_backward(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row)
        ctx.shape = x.shape[-3:]
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        lohi_dim_last = afb1d(x, h0_row, h1_row, mode=mode, dim=4)
        lohi_dim_last_2 = afb1d(lohi_dim_last, h0_col, h1_col, mode=mode, dim=3)
        y = afb1d(lohi_dim_last_2, h0_dep, h1_dep, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 8, s[-3], s[-2], s[-1])
        low = y[:, :, 0].contiguous()
        highs = y[:, :, 1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, lll, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_dep, h1_dep, h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            hll, lhl, hhl, llh, hlh, lhh, hhh = torch.unbind(highs, dim=2)

            ## first level
            ll = sfb1d(lll, hll, h0_dep, h1_dep, mode=mode, dim=2)
            hl= sfb1d(lhl, hhl, h0_dep, h1_dep, mode=mode, dim=2)
            lh = sfb1d(llh, hlh, h0_dep, h1_dep, mode=mode, dim=2)
            hh= sfb1d(lhh, hhh, h0_dep, h1_dep, mode=mode, dim=2)

            ## second level
            l = sfb1d(ll, hl, h0_col, h1_col, mode=mode, dim=3)
            h = sfb1d(lh, hh, h0_col, h1_col, mode=mode, dim=3)

            ## last level
            dx = sfb1d(l, h, h0_row, h1_row, mode = mode, dim = 4)

            if dx.shape[-3] > ctx.shape[-3]:
                dx = dx[:, :, :ctx.shape[-3]]
            if dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:, :, :, :ctx.shape[-2]]
            if dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:, :, :, :, :ctx.shape[-1]]

        return dx, None, None, None, None, None


if __name__ == '__main__':
    pass