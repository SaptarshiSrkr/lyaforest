"""
Extract a small example dataset from the full Spectra_Fields.npy
(simulation skewers provided in the ACE-IUCAA tutorial "Numerical Modeling
of Absorption Spectra" by Prakash Gaikwad; see README.md).

Keeps a few snapshots and the first few skewers, and saves them as a
compressed .npz file (no pickle) with keys '<redshift>/<field>',
e.g. '2.0/n_HI_arr'. Usage, from the repository directory:

    python example_data/make_example_data.py [path/to/Spectra_Fields.npy]
"""

import os
import sys

import numpy as np

REDSHIFT_KEYS = ['2.0', '3.0', '4.0', '5.0']
N_SKEWERS = 10

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectra_fields_example.npz')

def main(full_data_file):

    spectra_fields = np.load(full_data_file, allow_pickle=True, encoding='latin1').item()

    example = {}
    for redshift_key in REDSHIFT_KEYS:
        for field, arr in spectra_fields[redshift_key].items():
            example[f'{redshift_key}/{field}'] = arr if arr.ndim == 1 else arr[:, :N_SKEWERS]

    np.savez_compressed(OUTPUT_FILE, **example)
    print(f'Wrote {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB)')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'Spectra_Fields.npy')
