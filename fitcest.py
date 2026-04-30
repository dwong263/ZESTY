from PySide6.QtWidgets import (
    QApplication, QMainWindow,
    QFileDialog
)
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QStandardItemModel, QStandardItem

import os, sys
import nibabel as nib
import numpy as np
import pyqtgraph as pg
import csv

from util.viewer import TriPlanarViewer
from util.plotter import ZSpectrumPlotter
from util.fitutils import _multi_lorentzian, create_params_lorentzian, fit_volume_parallel_lorentzian

class FitCESTApp(QMainWindow):
    def __init__(self):
        # Initialize Window
        super(FitCESTApp, self).__init__()
        self.resize(1200, 900)
        self._load_ui()
        self._bind_widgets()
    
    def _load_ui(self):
        loader  = QUiLoader()
        ui_file = QFile("ui/fitcest.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()

        self.setCentralWidget(self.ui.centralWidget())

        # Layout Viewing Tab
        self.plotter = ZSpectrumPlotter()
        self.ui.display_box_fit.layout().addWidget(self.plotter)

        self.cest_viewer = TriPlanarViewer()
        self.ui.display_box_CEST.layout().addWidget(self.cest_viewer)

        self.AACID_viewer = TriPlanarViewer(cbar=True)
        self.ui.display_box_AACID.layout().addWidget(self.AACID_viewer)
   
    def _bind_widgets(self):
        # Load CEST Data Tab
        self.ui.SelectCESTFileButton.clicked.connect(self._select_cest_file)
        self.ui.SelectCESTOffsetsButton.clicked.connect(self._select_cest_offsets)
        self.ui.SelectB0Button.clicked.connect(self._select_b0_file)

        # Load Constraints Tab
        self.ui.SelectCSTFileButton.clicked.connect(self._select_cst_file)
        self.ui.FilenameLineEdit_cst.textChanged.connect(self._validate_filename_cst)
        self.ui.SaveCSTFileButton.clicked.connect(self._save_cst_file)
        self.ui.SelectGuessFileButton.clicked.connect(self._select_guess_file)
        self.ui.FilenameLineEdit_guess.textChanged.connect(self._validate_filename_guess)
        self.ui.SaveGuessFileButton.clicked.connect(self._save_guess_file)

        # Run Fit Tab
        self.ui.FilenameLineEdit_fit.textChanged.connect(self._validate_filename_fit)
        self.ui.RunFitButton.clicked.connect(self._fit_file)

        # Layout Viewing Tab
        self.ui.LoadCESTButton.clicked.connect(self._load_cest_data)
        self.ui.LoadAACIDButton.clicked.connect(self._load_aacid_data)
        self.ui.LoadFitButton.clicked.connect(self._load_cest_fit)
        self.ui.LoadOffsetsButton.clicked.connect(self._load_cest_offsets)

        self.ui.frequencySlider.valueChanged.connect(self._on_slider_change)
        self.cest_viewer.voxelSelected.connect(self._on_voxel_select)
        self.AACID_viewer.voxelSelected.connect(self._on_voxel_select)


    def _display_file_tree(self, file_paths):
        tree = self._build_tree(file_paths)
        text = "\n".join(self._tree_to_ascii(tree))
        self.ui.ConsoleTextBrowser.append(text)

    # ---------------------------
    # Load CEST Data Tab Methods
    # ---------------------------
    def _select_cest_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select CEST File",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.f_cest_file = file
            self.f_cest_img = nib.load(self.f_cest_file)
            self.f_cest_data = self.f_cest_img.get_fdata()
            self.ui.ConsoleTextBrowser.append("The following CEST file was loaded.")
            self._display_file_tree([file])
            self.ui.ConsoleTextBrowser.append("\n")

    def _select_cest_offsets(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select CEST Offsets",
            "",
            "Text Files (*.txt)"
        )
        if file:    # user didn't cancel
            with open(file, 'r') as f:
                self.f_cest_offsets = [float(line.strip()) for line in f if not ('#' in line)]
            self.ui.ConsoleTextBrowser.append("The following offset file was loaded.")
            self._display_file_tree([file])
            self.ui.ConsoleTextBrowser.append(f"Offsets: {str(self.f_cest_offsets)}")
            self.ui.ConsoleTextBrowser.append("\n")
    
    def _select_b0_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select ΔB0 File",
            "",
            "NifTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.f_b0_file = file
            self.f_b0_img = nib.load(self.f_b0_file)
            self.f_b0_data = self.f_b0_img.get_fdata()
            self.ui.ConsoleTextBrowser.append("The following ∆B0 file was loaded.")
            self._display_file_tree([file])
            self.ui.ConsoleTextBrowser.append("\n")
    
    # -----------------------------
    # Load Constraints Tab Methods
    # -----------------------------
    def _select_cst_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Constraint File",
            "",
            "Tab-Separated Files (*.tsv)"
        )
        if file:    # user didn't cancel
            self.ui.CSTFileLabel.setText(f"{file}")
            self.ui.ConstraintsTableView.setModel(self._load_tsv_to_model(file))
            self.ui.ConstraintsTableView.resizeColumnsToContents()
    
    def _select_guess_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Guess File",
            "",
            "Tab-Separated Files (*.tsv)"
        )
        if file:    # user didn't cancel
            self.ui.GuessFileLabel.setText(f"{file}")
            self.ui.GuessTableView.setModel(self._load_tsv_to_model(file))
            self.ui.GuessTableView.resizeColumnsToContents()

    def _validate_filename_cst(self):
        text = self.ui.FilenameLineEdit_cst.text().strip()

        if not text:
            self.ui.SaveCSTFileButton.setEnabled(False)
            return
        # Split into directory + filename
        directory, filename = os.path.split(text)

        # Check filename is not empty
        if not filename:
            self.ui.SaveCSTFileButton.setEnabled(False)
            return

        # Optional: reject illegal characters (Windows-safe)
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            self.ui.SaveCSTFileButton.setEnabled(False)
            return

        # If directory is provided, check it exists
        if directory and not os.path.exists(directory):
            self.ui.SaveCSTFileButton.setEnabled(False)
            return

        self.ui.SaveCSTFileButton.setEnabled(True)

    def _validate_filename_guess(self):
        text = self.ui.FilenameLineEdit_cst.text().strip()

        if not text:
            self.ui.SaveGuessFileButton.setEnabled(False)
            return
        # Split into directory + filename
        directory, filename = os.path.split(text)

        # Check filename is not empty
        if not filename:
            self.ui.SaveGuessFileButton.setEnabled(False)
            return

        # Optional: reject illegal characters (Windows-safe)
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            self.ui.SaveGuessFileButton.setEnabled(False)
            return

        # If directory is provided, check it exists
        if directory and not os.path.exists(directory):
            self.ui.SaveGuessFileButton.setEnabled(False)
            return

        self.ui.SaveGuessFileButton.setEnabled(True)
    
    def _save_cst_file(self):
        filename = self.ui.FilenameLineEdit_cst.text().strip()
        self._save_model_to_tsv(self.ui.ConstraintsTableView.model(), filename)        

    def _save_guess_file(self):
        filename = self.ui.FilenameLineEdit_guess.text().strip()
        self._save_model_to_tsv(self.ui.GuessTableView.model(), filename)

    def _load_tsv_to_model(self, filepath: str) -> QStandardItemModel:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = [row for row in reader if any(cell.strip() for cell in row)]  # skip blank lines

        if not rows:
            return QStandardItemModel()

        # Strip trailing empty columns by finding the rightmost non-empty column
        max_cols = max(
            next((i for i in range(len(row) - 1, -1, -1) if row[i].strip()), -1) + 1
            for row in rows
        )

        headers = rows[0][:max_cols]
        model = QStandardItemModel(len(rows) - 1, len(headers))
        model.setHorizontalHeaderLabels(headers)

        for row_idx, row in enumerate(rows[1:]):
            for col_idx in range(max_cols):
                value = row[col_idx] if col_idx < len(row) else ""
                model.setItem(row_idx, col_idx, QStandardItem(value))

        return model
    
    def _model_to_bounds(self, model: QStandardItemModel) -> list[list[float]]:
        bounds = []
        for row in range(model.rowCount()):
            min_val = float(model.item(row, 2).text())  # "Minimnum" column
            max_val = float(model.item(row, 3).text())  # "Maximum" column
            bounds.append([min_val, max_val])
        return bounds

    def _model_to_p0(self, model: QStandardItemModel) -> list[float] | None:
        p0 = []
        try:
            for row in range(model.rowCount()):
                p0.append(float(model.item(row, 2).text()))  # "Guess" column
        except (ValueError, AttributeError) as e:
            print(f"Invalid value in table at row {row}: {e}")
            return None
        return p0

    def _save_model_to_tsv(self, model: QStandardItemModel, filepath: str) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")

            # Write header row
            headers = [model.horizontalHeaderItem(col).text() for col in range(model.columnCount())]
            writer.writerow(headers)

            # Write data rows
            for row in range(model.rowCount()):
                writer.writerow([model.item(row, col).text() for col in range(model.columnCount())])
    
    # ------------
    # Run Fit Tab
    # ------------
    def _validate_filename_fit(self):
        text = self.ui.FilenameLineEdit_fit.text().strip()

        if not text:
            self.ui.RunFitButton.setEnabled(False)
            return
        # Split into directory + filename
        directory, filename = os.path.split(text)

        # Check filename is not empty
        if not filename:
            self.ui.RunFitButton.setEnabled(False)
            return

        # Optional: reject illegal characters (Windows-safe)
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            self.ui.RunFitButton.setEnabled(False)
            return

        # If directory is provided, check it exists
        if directory and not os.path.exists(directory):
            self.ui.RunFitButton.setEnabled(False)
            return

        self.ui.RunFitButton.setEnabled(True)
    
    def _fit_file(self):

        self.ui.FitConsoleTextBrowser.append(f"Getting relevant images ...")

        cest_img = self.f_cest_img
        cest_data = self.f_cest_data
        cest_offsets = np.array(self.f_cest_offsets)
        
        step = 0.05
        cest_offsets_interp = np.arange(cest_offsets[0], cest_offsets[-1]+step, step)

        dB0_data = self.f_b0_data

        # Generate Corrected Offsets (i.e. apply B0 correction)
        corrected_offsets = np.zeros((
                        dB0_data.shape[0],
                        dB0_data.shape[1],
                        dB0_data.shape[2],
                        len(cest_offsets)
                    ))
        for i in range(dB0_data.shape[0]):
            for j in range(dB0_data.shape[1]):
                for k in range(dB0_data.shape[2]):
                    if dB0_data[i,j,k] != 0:
                        corrected_offsets[i,j,k,:] = cest_offsets - dB0_data[i,j,k]
                    else:
                        continue
        
        self.ui.FitConsoleTextBrowser.append(f"Getting constraints and guess from tables ...")
        p0 = self._model_to_p0(self.ui.GuessTableView.model())
        bounds = self._model_to_bounds(self.ui.ConstraintsTableView.model())

        n_peaks = int((len(p0)-1)/3)
        self.ui.FitConsoleTextBrowser.append(f"Number of peaks detected: {n_peaks}")

        self.ui.FitConsoleTextBrowser.append(f"Starting fit (INTERFACE WILL FREEZE!)")
        # Fit Lorentzian Model
        cest_data_fitted_params = fit_volume_parallel_lorentzian(
                cest_data, 
                corrected_offsets,
                p0,
                bounds
            )

        # Save Fitted Parameters
        cest_data_fitted_params_img = nib.Nifti1Image(cest_data_fitted_params, cest_img.affine, cest_img.header)
        nib.save(cest_data_fitted_params_img, self.ui.FilenameLineEdit_fit.text().strip()+'_fitted_params.nii.gz')
        self.ui.FitConsoleTextBrowser.append(f"Fitted parameters saved to: {self.ui.FilenameLineEdit_fit.text().strip()}_fitted_params.nii.gz")

        cest_data_fitted = np.zeros((
            cest_data.shape[0],
            cest_data.shape[1],
            cest_data.shape[2],
            len(cest_offsets_interp)
        ))
        for i in range(cest_data.shape[0]):
            for j in range(cest_data.shape[1]):
                for k in range(cest_data.shape[2]):
                    popt = cest_data_fitted_params[i,j,k,:]
                    if np.nansum(popt) != 0:
                        cest_data_fitted[i,j,k,:] = _multi_lorentzian(create_params_lorentzian(popt, bounds), cest_offsets_interp)
                    else:
                        continue
        cest_data_fitted_img = nib.Nifti1Image(cest_data_fitted, cest_img.affine, cest_img.header)
        nib.save(cest_data_fitted_img, self.ui.FilenameLineEdit_fit.text().strip()+'_fitted.nii.gz')
        self.ui.FitConsoleTextBrowser.append(f"Fitted z-spectra saved to: {self.ui.FilenameLineEdit_fit.text().strip()}_fitted.nii.gz")

        AACID_data = np.zeros((
                        cest_data.shape[0],
                        cest_data.shape[1],
                        cest_data.shape[2],
                ))
        for i in range(cest_data.shape[0]):
            for j in range(cest_data.shape[1]):
                for k in range(cest_data.shape[2]):
                    popt = cest_data_fitted_params[i,j,k,:]
                    if np.nansum(popt) != 0:
                        Mz_3_50 = _multi_lorentzian(create_params_lorentzian(popt, bounds), 3.50)
                        Mz_6_00 = _multi_lorentzian(create_params_lorentzian(popt, bounds), 6.00)
                        Mz_2_75 = _multi_lorentzian(create_params_lorentzian(popt, bounds), 2.75)
                        AACID_data[i, j, k] = (Mz_3_50 * (Mz_6_00 - Mz_2_75)) / (Mz_2_75 * (Mz_6_00 - Mz_3_50))
                    else:
                        continue
        AACID_img = nib.Nifti1Image(AACID_data, cest_img.affine, cest_img.header)
        nib.save(AACID_img, f"{self.ui.FilenameLineEdit_fit.text().strip()}_AACID_{n_peaks}L.nii.gz")
        self.ui.FitConsoleTextBrowser.append(f"AACID map saved to: {self.ui.FilenameLineEdit_fit.text().strip()}_AACID_{n_peaks}.nii.gz")

    # ---------------
    # Viewer Methods
    # ---------------
    def _load_cest_data(self):
        self.f_index = 0
        self.ui.frequencyIndex.setText(f"f_index = {self.f_index}")
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select CEST File",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.v_cest_file = file
            self.v_cest_img = nib.load(file)
            self.v_cest_data = self.v_cest_img.get_fdata()
            self.cest_viewer.set_volume(self.v_cest_data[:,:,:,self.f_index])
            
            self.ui.frequencySlider.setMaximum(self.v_cest_data.shape[3]-1)

    def _load_cest_fit(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select CEST Fit Parameters",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.v_cest_params_file = file
            self.v_cest_params_img = nib.load(file)
            self.v_cest_params_data = self.v_cest_params_img.get_fdata()
            self.ui.LoadOffsetsButton.setEnabled(True)

    def _load_cest_offsets(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select CEST Offsets",
            "",
            "Text Files (*.txt)"
        )
        if file:   # user didn't cancel
            with open(file, 'r') as f:
                self.v_cest_offsets = [float(line.strip()) for line in f if not('#' in line)]
            step = 0.05
            self.v_cest_offsets_interp = np.arange(self.v_cest_offsets[0], self.v_cest_offsets[-1]+step, step)

            x, y, z = self.cest_viewer.get_crosshairs()
            self._update_voxel_plot(x, y, z)

    def _load_aacid_data(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select ∆B0 File",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.v_aacid_file = file
            self.v_aacid_img = nib.load(file)
            self.v_aacid_data = self.v_aacid_img.get_fdata()

            # Viewer settings
            self.AACID_viewer.set_volume(self.v_aacid_data)

            cmap = pg.colormap.get('viridis')
            self.AACID_viewer.axial_view.setColorMap(cmap)
            self.AACID_viewer.sagittal_view.setColorMap(cmap)
            self.AACID_viewer.coronal_view.setColorMap(cmap)

            vmin = np.min(self.v_aacid_data); vmax = np.max(self.v_aacid_data)
            self.AACID_viewer.colorbar.setColorMap(cmap)
            self.AACID_viewer.colorbar.setLevels(low=vmin, high=vmax)
            self.AACID_viewer.colorbar.setImageItem([
                self.AACID_viewer.axial_view.getImageItem(),
                self.AACID_viewer.sagittal_view.getImageItem(),
                self.AACID_viewer.coronal_view.getImageItem()
            ])

            self.AACID_viewer.axial_view.setLevels(vmin, vmax)
            self.AACID_viewer.sagittal_view.setLevels(vmin, vmax)
            self.AACID_viewer.coronal_view.setLevels(vmin, vmax)

            self.ui.LoadFitButton.setEnabled(True)

    def _on_voxel_select(self, x, y, z):
        sending_widget = self.sender()
        if sending_widget is self.cest_viewer:
            self.AACID_viewer.update_views(x, y, z)
        elif sending_widget is self.AACID_viewer:
            self.cest_viewer.update_views(x, y, z)

        self.ui.lineEdit_x.setText(f"{x}")
        self.ui.lineEdit_y.setText(f"{y}")
        self.ui.lineEdit_z.setText(f"{z}")

        self._update_voxel_plot(x, y, z)

    def _update_voxel_plot(self, x, y, z):
        fit     = self._calculate_fit(
            self.v_cest_offsets_interp,
            self.v_cest_params_data[x, y, z]
        )

        self.plotter.update_plot(
            x_raw = self.v_cest_offsets,
            z_data= self.v_cest_data[x, y, z, :],
            x_fit = self.v_cest_offsets_interp,
            fit   = fit,
            components = None,
            residual   = self._calculate_fit(
                self.v_cest_offsets,
                self.v_cest_params_data[x, y, z]
            ) - self.v_cest_data[x, y, z, :]
        )
    
    def _on_slider_change(self, f_index):
        self.f_index = f_index

        x = int(self.ui.lineEdit_x.text())
        y = int(self.ui.lineEdit_y.text())
        z = int(self.ui.lineEdit_z.text())
        
        self.ui.frequencyIndex.setText(f"f_index = {self.f_index}")
        self.cest_viewer.set_volume(self.v_cest_data[:,:,:,self.f_index], x, y, z)
        self.cest_viewer.update_views(x, y, z)

    def _calculate_fit(self, f, popt):
        bounds  = np.array([[None, None]] * len(popt))
        return _multi_lorentzian(create_params_lorentzian(popt, bounds), f)

    # -------------------------
    # Console Display Methods
    # -------------------------
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

# ---- Launch Application ---- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setStyle('Fusion')

    window = FitCESTApp()
    window.show()
    sys.exit(app.exec())