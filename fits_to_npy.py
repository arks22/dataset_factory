import os
import sys
import numpy as np
import argparse
from astropy.io import fits

parser = argparse.ArgumentParser(description='simple fits to npy')
parser.add_argument('source', type=str)
parser.add_argument('--out_dir', type=str, default='.')
args = parser.parse_args()

fits_data = fits.open(args.source)
img = np.array(fits_data[1].data)

basename = os.path.splitext(os.path.basename(args.source))[0]
np.save(os.path.join(args.out_dir,basename), img)
