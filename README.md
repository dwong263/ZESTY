# About ZESTY
ZESTY stands for **Z**-spectrum **E**valuation and **S**pectral **T**ools in p**Y**thon.  This set of software tools were developed as a simple way to post-process and fit CEST imaging data and produce pH-weighted [AACID](https://pmc.ncbi.nlm.nih.gov/articles/PMC3982091/) maps.

# Dependencies
This program is written and tested in Python 3.13.5.  Dependencies include:

* PySide6
* PyQtGraph
* NiBabel
* Numpy
* SciPy
* Lmfit
* Joblib
* Natsort

These dependencies can be easily installed using `pip`.

# Usage
Navigate to the folder containing the ZESTY files and launch the program with the following commands:

```
cd <filepath to ZESTY>
python zesty.py
```
You can also launch the indvidual programs directly using:

```
python concat.py
python fitwassr.py
python fitcest.py
```

# Included Tools
## Concatenate NifTI Files
This application concatenates 3D CEST images and saves a 4D volume with the 4th dimension being the frequency axis.  During this process, it will normalize the signal by the -300 ppm image, assumed to be the first image in the loaded series.  It will also apply a provided mask to the data.

## Fit WASSR Data
This application allows you to load a normalized and masked WASSR image, a corresponding list of frequency offsets and fits a single Gaussian function voxelwise.  It produces a file containing the fitted curve, a file containing the fit parameters, and a ∆B<sub>0</sub> map.  The application also includes a viewer that allows you to inspect the normalized and masked WASSR image, ∆B<sub>0</sub> map, and the Gaussian fit.

<img width="1552" height="982" alt="image" src="https://github.com/user-attachments/assets/394f94c1-d0be-4434-bf80-44790e7dd29e" />

## Fit CEST Data
This application allows you to:

* load a normalized and masked CEST image
* load a ∆B<sub>0</sub> map
* load the corresponding list of frequency offsets
* load and edit a multi-Lorentzian model with bounds
* load and edit an initial guess for the parameters
* run a voxelwise multi-Lorentzian fit

It will produce a file containing the fitted curve, a file contianing the fit parameters, and an AACID map.  The application also icnludes a viewer that allows you to inspect the normalized and masked CEST image, the AACID map, and the multi-Lorentzian fit.  You can also smooth the AACID map and saved the smoothed map.

<img width="1552" height="1089" alt="image" src="https://github.com/user-attachments/assets/ae8100d0-bbee-498d-b913-10d738a028d3" />

# Documentation
*More detailed documentation to come ...*

# Credits
Dickson Wong ([dwong2022@meds.uwo.ca](mailto:dwong2022@meds.uwo.ca))

# License
This software is licensed uner the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).

In its current state, it is a minimum viable product developed for internal use by the [Bartha Lab](https://cfmm.uwo.ca/people/core_scientists/Rob.html) and Bartha Lab collaborators.  It is not intended for commercial use.
