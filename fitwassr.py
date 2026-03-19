from PySide6.QtWidgets import (
    QApplication, QMainWindow,
    QFileDialog
)
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

import os, sys
import nibabel as nib
import numpy as np
import pyqtgraph as pg

from util.viewer import TriPlanarViewer
from util.plotter import ZSpectrumPlotter
from util.fitutils import _1Gaussian, fit_volume_parallel_gaussian

class FitWASSRApp(QMainWindow):
    def __init__(self):
        # Initialize Window
        super(FitWASSRApp, self).__init__()
        self.resize(1200, 900)
        self._load_ui()
        self._bind_widgets()
    
    def _load_ui(self):
        loader  = QUiLoader()
        ui_file = QFile("ui/fitwassr.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()

        self.setCentralWidget(self.ui.centralWidget())
        
        # Layout Viewing Tab
        self.plotter = ZSpectrumPlotter()
        self.ui.display_box_fit.layout().addWidget(self.plotter)

        self.wassr_viewer = TriPlanarViewer()
        self.ui.display_box_wassr.layout().addWidget(self.wassr_viewer)

        self.b0_viewer = TriPlanarViewer(cbar=True)
        self.ui.display_box_b0.layout().addWidget(self.b0_viewer)

    def _bind_widgets(self):
        # Fitting Tab
        self.ui.SelectFileButton.clicked.connect(self._select_file)
        self.ui.SelectOffsetsButton.clicked.connect(self._select_offsets)
        self.ui.FitFileButton.clicked.connect(self._fit_file)
        self.ui.FilenameLineEdit.textChanged.connect(self._validate_filename)

        # Viewing Tab
        self.ui.LoadWASSRButton.clicked.connect(self._load_wassr_data)
        self.ui.LoadFitButton.clicked.connect(self._load_wassr_fit)
        self.ui.LoadWASSROffsetsButton.clicked.connect(self._load_wassr_offsets)
        self.ui.LoadB0Button.clicked.connect(self._load_dB0_data)

        self.ui.frequencySlider.valueChanged.connect(self._on_slider_change)
        self.wassr_viewer.voxelSelected.connect(self._on_voxel_select)
        self.b0_viewer.voxelSelected.connect(self._on_voxel_select)

    # -------------------------
    # Fitter Methods
    # -------------------------
    # ---- WASSR File Selector ---- #
    def _select_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select WASSR File",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.f_wassr_file = file
            self.f_wassr_img = nib.load(self.f_wassr_file)
            self.f_wassr_data = self.f_wassr_img.get_fdata()
            self.ui.ConsoleTextBrowser.append("The following mask file was loaded.")
            self._display_file_tree([file])
            self.ui.ConsoleTextBrowser.append("\n")
    
    # ---- WASSR Offsets Selector ---- #
    def _select_offsets(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select WASSR Offsets",
            "",
            "Text Files (*.txt)"
        )
        if file:   # user didn't cancel
            with open(file, 'r') as f:
                self.f_wassr_offsets = [float(line.strip()) for line in f if not('#' in line)]
            self.ui.ConsoleTextBrowser.append("The following offset file was loaded.")
            self._display_file_tree([file])
            self.ui.ConsoleTextBrowser.append(f"Offsets: {str(self.f_wassr_offsets)}")
            self.ui.ConsoleTextBrowser.append("\n")
    
    # --- Fit File Methods ---- #
    def _fit_file(self):
        wassr_img = self.f_wassr_img
        wassr_data = self.f_wassr_data
        wassr_offsets = np.array(self.f_wassr_offsets)

        wassr_offsets_interp = np.arange(wassr_offsets[0], wassr_offsets[-1]+0.1, 0.1)

        self.ui.ConsoleTextBrowser.append("Fitting WASSR file ...")
        wassr_data_fitted_params = fit_volume_parallel_gaussian(
            wassr_data,
            wassr_offsets,
            p0=[0.5, -1, 0, 0.5]
        )
        wassr_data_fitted_params_img = nib.Nifti1Image(wassr_data_fitted_params, wassr_img.affine, wassr_img.header)
        nib.save(wassr_data_fitted_params_img, self.ui.FilenameLineEdit.text().strip()+'_fitted_params.nii.gz')
        self.ui.ConsoleTextBrowser.append(f"Fitted parameters saved to: {self.ui.FilenameLineEdit.text().strip()}_fitted_params.nii.gz")
        
        wassr_data_fitted = np.zeros((
            wassr_data.shape[0],
            wassr_data.shape[1],
            wassr_data.shape[2],
            len(wassr_offsets_interp)
        ))
        for i in range(wassr_data.shape[0]):
            for j in range(wassr_data.shape[1]):
                for k in range(wassr_data.shape[2]):
                    popt = wassr_data_fitted_params[i,j,k,:]
                    if np.nansum(popt) != 0:
                        wassr_data_fitted[i,j,k,:]=_1Gaussian(wassr_offsets_interp,
                                                              popt[0],
                                                              popt[1], popt[2], popt[3]
                                                              )
                    else:
                        continue
        wassr_data_fitted_img = nib.Nifti1Image(wassr_data_fitted, wassr_img.affine, wassr_img.header)
        nib.save(wassr_data_fitted_img, self.ui.FilenameLineEdit.text().strip()+'_fitted.nii.gz')
        self.ui.ConsoleTextBrowser.append(f"Fitted z-spectra saved to: {self.ui.FilenameLineEdit.text().strip()}_fitted.nii.gz")

        dB0_data = np.zeros((
            wassr_data.shape[0],
            wassr_data.shape[1],
            wassr_data.shape[2]
        ))

        dB0_data = wassr_data_fitted_params[:, :, :, 2]
        dB0_img = nib.Nifti1Image(dB0_data, wassr_img.affine, wassr_img.header)
        nib.save(dB0_img, self.ui.FilenameLineEdit.text().strip()+'_dB0.nii.gz')
        self.ui.ConsoleTextBrowser.append(f"Fitted z-spectra saved to: {self.ui.FilenameLineEdit.text().strip()}_dB0.nii.gz")
        self.ui.ConsoleTextBrowser.append("\n")

    def _validate_filename(self):
        text = self.ui.FilenameLineEdit.text().strip()

        if not text:
            self.ui.FitFileButton.setEnabled(False)
            return
        # Split into directory + filename
        directory, filename = os.path.split(text)

        # Check filename is not empty
        if not filename:
            self.ui.FitFileButton.setEnabled(False)
            return

        # Optional: reject illegal characters (Windows-safe)
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            self.ui.FitFileButton.setEnabled(False)
            return

        # If directory is provided, check it exists
        if directory and not os.path.exists(directory):
            self.ui.FitFileButton.setEnabled(False)
            return

        self.ui.FitFileButton.setEnabled(True)   

    # ---- Console Display Methods ---- #
    def _build_tree(self, file_paths):
        tree = {}

        for path in file_paths:
            parts = os.path.normpath(path).split(os.sep)
            current_level = tree

            for part in parts[:-1]:
                current_level = current_level.setdefault(part, {})
            
            current_level.setdefault("__files__", []).append(parts[-1])
        
        return tree

    def _tree_to_ascii(self, tree, prefix=""):
        lines = []

        # separate folders and files
        folders = [k for k in tree.keys() if k != "__files__"]
        files = tree.get("__files__", [])

        entries = folders + files

        for i, name in enumerate(entries):
            is_last = (i == len(entries) - 1)

            connector = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "

            if name in folders:
                lines.append(f"{prefix}{connector}{name}/")
                lines.extend(
                    self._tree_to_ascii(tree[name], prefix + next_prefix)
                )
            else:
                lines.append(f"{prefix}{connector}{name}")

        return lines
    
    def _display_file_tree(self, file_paths):
        tree = self._build_tree(file_paths)
        text = "\n".join(self._tree_to_ascii(tree))
        self.ui.ConsoleTextBrowser.append(text)

    # -------------------------
    # Viewer Methods
    # -------------------------
    def _load_wassr_data(self):
        self.f_index = 0
        self.ui.frequencyIndex.setText(f"f_index = {self.f_index}")
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select WASSR File",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.v_wassr_file = file
            self.v_wassr_img = nib.load(file)
            self.v_wassr_data = self.v_wassr_img.get_fdata()
            self.wassr_viewer.set_volume(self.v_wassr_data[:,:,:,self.f_index])
            
            self.ui.frequencySlider.setMaximum(self.v_wassr_data.shape[3]-1)

    def _load_wassr_fit(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select WASSR Fit Parameters",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.v_wassr_params_file = file
            self.v_wassr_params_img = nib.load(file)
            self.v_wassr_params_data = self.v_wassr_params_img.get_fdata()
            self.ui.LoadWASSROffsetsButton.setEnabled(True)
    
    def _load_wassr_offsets(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select WASSR Offsets",
            "",
            "Text Files (*.txt)"
        )
        if file:   # user didn't cancel
            with open(file, 'r') as f:
                self.v_wassr_offsets = [float(line.strip()) for line in f if not('#' in line)]
            self.v_wassr_offsets_interp = np.arange(self.v_wassr_offsets[0], self.v_wassr_offsets[-1]+0.1, 0.1)

            x, y, z = self.wassr_viewer.get_crosshairs()
            self._update_voxel_plot(x, y, z)

    def _load_dB0_data(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select ∆B0 File",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.v_dB0_file = file
            self.v_dB0_img = nib.load(file)
            self.v_dB0_data = self.v_dB0_img.get_fdata()

            # Viewer settings
            self.b0_viewer.set_volume(self.v_dB0_data)

            cmap = pg.colormap.get('viridis')
            self.b0_viewer.axial_view.setColorMap(cmap)
            self.b0_viewer.sagittal_view.setColorMap(cmap)
            self.b0_viewer.coronal_view.setColorMap(cmap)

            vmin = np.min(self.v_dB0_data); vmax = np.max(self.v_dB0_data)
            self.b0_viewer.colorbar.setColorMap(cmap)
            self.b0_viewer.colorbar.setLevels(low=vmin, high=vmax)
            self.b0_viewer.colorbar.setImageItem([
                self.b0_viewer.axial_view.getImageItem(),
                self.b0_viewer.sagittal_view.getImageItem(),
                self.b0_viewer.coronal_view.getImageItem()
            ])

            self.b0_viewer.axial_view.setLevels(vmin, vmax)
            self.b0_viewer.sagittal_view.setLevels(vmin, vmax)
            self.b0_viewer.coronal_view.setLevels(vmin, vmax)

            self.ui.LoadFitButton.setEnabled(True)

    def _on_voxel_select(self, x, y, z):
        sending_widget = self.sender()
        if sending_widget is self.wassr_viewer:
            self.b0_viewer.update_views(x, y, z)
        elif sending_widget is self.b0_viewer:
            self.wassr_viewer.update_views(x, y, z)

        self.ui.lineEdit_x.setText(f"{x}")
        self.ui.lineEdit_y.setText(f"{y}")
        self.ui.lineEdit_z.setText(f"{z}")

        self._update_voxel_plot(x, y, z)

    def _update_voxel_plot(self, x, y, z):
        fit     = self._calculate_fit(
            self.v_wassr_offsets_interp,
            self.v_wassr_params_data[x, y, z]
        )

        self.plotter.update_plot(
            x_raw = self.v_wassr_offsets,
            z_data= self.v_wassr_data[x, y, z, :],
            x_fit = self.v_wassr_offsets_interp,
            fit   = fit,
            components = None,
            residual   = self._calculate_fit(
                self.v_wassr_offsets,
                self.v_wassr_params_data[x, y, z]
            ) - self.v_wassr_data[x, y, z, :]
        )
    
    def _on_slider_change(self, f_index):
        self.f_index = f_index

        x = int(self.ui.lineEdit_x.text())
        y = int(self.ui.lineEdit_y.text())
        z = int(self.ui.lineEdit_z.text())
        
        self.ui.frequencyIndex.setText(f"f_index = {self.f_index}")
        self.wassr_viewer.set_volume(self.v_wassr_data[:,:,:,self.f_index], x, y, z)
        self.wassr_viewer.update_views(x, y, z)

    def _calculate_fit(self, f, popt):
        return _1Gaussian(f, popt[0], popt[1], popt[2], popt[3])

# ---- Launch Application ---- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setStyle('Fusion')

    window = FitWASSRApp()
    window.show()
    sys.exit(app.exec())