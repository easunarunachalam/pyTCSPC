import numpy as np
import matplotlib.pyplot as plt
import pyTCSPC as pc

def test_zoom(plot=False):

    xi = np.linspace(-1,1,1001)
    X, Y = np.meshgrid(xi, xi)
    syn_im = X**2 + Y**2

    if plot:
        fig, ax = plt.subplots(ncols=4, figsize=(16,4))
        # p = [None]*4
        p[0] = ax[0].imshow(X)
        p[1] = ax[1].imshow(Y)
        p[2] = ax[2].imshow(syn_im)
        p[3] = ax[3].imshow(syn_im)
        _ = [plt.colorbar(p[i], ax=ax[i]) for i in range(len(p))]
        plt.tight_layout()

    zoom_im = pc.zoom_image(syn_im, zoom_factor=2)

    if plot:
        plt.imshow(zoom_im)
        plt.colorbar()

    assert zoom_im.shape == (501,501)

    assert np.allclose(np.max(zoom_im), 0.5, atol=1e-3)


# flim_image = xr.DataArray(
#     data=np.random.rand(1,2,10,10,16),
#     dims=["file_info", "channel", "x", "y", "microtime_ns"]
# )
#
# flim_image2 = xr.DataArray(
#     data=np.random.rand(10,10),
#     dims=["x", "y"]
# )
#
# xarr_multiply = flim_image * flim_image2
