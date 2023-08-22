from pyrism.rism_ctrl import *
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

mol = RismController("../data/cSPCE_XRISM_methane.toml")

mol.initialise_controller()

mol.do_rism(verbose=True)

print(mol.SFED['KH'])

fft_sfed = rfft(mol.SFED['KH'])
freqs = rfftfreq(mol.SFED['KH'].shape[0], mol.vv.grid.d_r)

plt.plot(mol.vv.grid.ri, mol.SFED['KH'])
plt.plot(freqs, np.abs(fft_sfed))
plt.show()