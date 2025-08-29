from pytorch_wavelets.dwt import lowlevel as lowlevel
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row):
    g0_row, g1_row = lowlevel.prep_filt_sfb1d(g0_row, g1_row, device)
    g0_col, g1_col = lowlevel.prep_filt_sfb1d(g0_col, g1_col, device)
    g0_dep, g1_dep = lowlevel.prep_filt_sfb1d(g0_dep, g1_dep, device)

    g0_dep = g0_dep.reshape((1, 1, -1, 1, 1))
    g1_dep = g1_dep.reshape((1, 1, -1, 1, 1))
    g0_col = g0_col.reshape((1, 1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, 1, -1))

    return g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row


def prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row):
    h0_row, h1_row = lowlevel.prep_filt_afb1d(h0_row, h1_row, device)
    h0_col, h1_col = lowlevel.prep_filt_afb1d(h0_col, h1_col, device)
    h0_dep, h1_dep = lowlevel.prep_filt_afb1d(h0_dep, h1_dep, device)

    h0_dep = h0_dep.reshape((1, 1, -1, 1, 1))
    h1_dep = h1_dep.reshape((1, 1, -1, 1, 1))
    h0_col = h0_col.reshape((1, 1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, 1, -1))

    return h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row