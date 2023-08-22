from pyrism.rism_ctrl import *
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft

mol = RismController("../data/cSPCE_XRISM_methane.toml")

mol.initialise_controller()

mol.do_rism(verbose=True)

fft_sfed = rfft(mol.SFED['KH'])
freqs = rfftfreq(mol.SFED['KH'].shape[0], mol.vv.grid.d_r)

a = 0
b = np.argmax(fft_sfed) + 50
new_signal = np.zeros_like(fft_sfed)
new_signal[a:b] = fft_sfed[a:b]
plt.plot(freqs, np.abs(fft_sfed))
plt.plot(freqs, np.abs(new_signal))
plt.show()


new_sfed = irfft(new_signal)
plt.plot(mol.vv.grid.ri, mol.SFED['KH'])
plt.plot(mol.vv.grid.ri, new_sfed)
plt.show()