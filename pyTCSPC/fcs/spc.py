import numpy as np
import os, sys


def bits(f):
    bytes = (ord(chr(b)) for b in f)
    for b in bytes:
        for i in range(8):
            yield (b >> i) & 1

def SPCReader(filename):

    iimg = np.zeros((1024,1024))
    pixel = line = frame = 0

    nphotons = os.path.getsize(filename)/4

    if nphotons.is_integer():
        nphotons = int(nphotons)
    else:
        print("Error: number of bytes in file not evenly divisible by 4.")
        sys.exit()

    macrotime = np.empty((nphotons, 1))
    microtime = np.empty((nphotons, 1))
    hvpixel   = np.empty((nphotons, 1))
    hvline    = np.empty((nphotons, 1))

    overflow = 0

    with open(filename, "rb") as f:
        for ii, i in enumerate(range(nphotons)):

            scale = 100
            if (ii+1) % (nphotons//scale) == 0:
                print(int(scale*ii/nphotons), end="% complete \r")

            # data for single photon, 4 bytes / 2 words
            photondata = list(bits(f.read(4)))
            MT      = int("".join(reversed([str(i) for i in photondata[0:12]])),2)
            ROUT    = list([i for i in photondata[12:16]])

            if ROUT[0] != 0:
                pixel += 1
            if ROUT[1] != 0:
                line += 1
                pixel = 0
            if ROUT[2] != 0:
                frame += 1
                line = 0
                pixel = 0

#             if ii%(nphotons//100) == 0:
# #                 pbar.value += 1
#                 print("{:d}/{:d} photons processed\r".format(ii,nphotons));

            ADC     = int("".join(reversed([str(i) for i in photondata[16:28]])),2)
            MARK    = photondata[28]
            GAP     = photondata[29]
            MTOV    = photondata[30]
            INVALID = photondata[31]

            macrotime[i] = MT + overflow
            microtime[i] = 4095 - ADC

            if INVALID == 1:
                macrotime[i] = microtime[i] = hvpixel[i] = hvline[i] = np.nan
            else:
                iimg[pixel,line] += 1
                hvpixel[i], hvline[i] = pixel, line

            if MTOV == 1: # Macrotime clock overflow
                if INVALID == 1:
                    overflow += MT*(2**12)
                    macrotime[i] += MT*(2**12)
                else:
                    overflow += 2**12
                    macrotime[i] += 2**12

        valid = np.logical_not(np.isnan(macrotime))
        macrotime, microtime, hvpixel, hvline = macrotime[valid], microtime[valid], hvpixel[valid], hvline[valid]
#         microtime = microtime[np.logical_not(np.isnan(microtime))]

        return macrotime, microtime, hvpixel, hvline, iimg
