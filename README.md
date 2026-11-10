# lyaforest

Synthetic Lyman-α forest spectra from the skewers of any simulation, computed by summing Voigt profiles along the line of sight.

```bash
pip install -r requirements.txt
```

```python
import numpy as np
import lyaforest

# n_HI [cm^-3], T [K], v_pec [km/s]: shape (N_pixel,) or (N_pixel, N_los)
# z: redshift of each pixel, or one value; L_box in cMpc/h
tau = lyaforest.compute_tau(n_HI, T, v_pec, z, L_box=40)
flux = np.exp(-tau)
```

Cosmology (`hubble_param`, `omega_m`, `omega_l`), periodic boundaries (`periodic=True`) and the skewer axis (`axis=0`) can be set as keyword arguments.

`example.ipynb` shows a hand-made skewer and skewers from a small example dataset (`example_data/`, read with `lyaforest.read_skewer`), compared with the simulation's own optical depth.

## Acknowledgements

I wrote this code as a participant in the tutorial *Numerical Modeling of Absorption Spectra* by Prakash Gaikwad at the [ACE 2026 Summer Course "Physics of Galaxy Formation"](https://ace.iucaa.in/index.php?option=com_content&view=article&id=22&catid=2), IUCAA, Pune. The simulation data were provided in the tutorial ([materials](https://people.ast.cam.ac.uk/~pgaikwad/IUCAA_SCHOOL/)).
