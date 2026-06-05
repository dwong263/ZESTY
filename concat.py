from PySide6.QtWidgets import (
    QApplication, QMainWindow,
    QFileDialog
)
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

import os, sys
import nibabel as nib
import numpy as np

from natsort import natsorted
from util.croputils import crop_to_common_overlap
from util.cest_moco import run_cest_moco

class ConcatApp(QMainWindow):
    def __init__(self):
        # Initialize Window
        super(ConcatApp, self).__init__()
        self.resize(800, 600)
        self._load_ui()
        self._bind_widgets()
    
    def _load_ui(self):
        loader  = QUiLoader()
        ui_file = QFile("ui/concat.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()
        self.setCentralWidget(self.ui.centralWidget())
    
    def _bind_widgets(self):
        self.ui.SelectFilesButton.clicked.connect(self._select_files)
        self.ui.SelectOffsetsButton.clicked.connect(self._select_offsets)
        self.ui.SelectMaskButton.clicked.connect(self._select_mask)
        self.ui.SaveFileButton.clicked.connect(self._save_file)
        self.ui.FilenameLineEdit.textChanged.connect(self._validate_filename)

    # ---- CEST Files Selector ---- #
    def _select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CEST Data Files",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if files:   # user didn't cancel
            self.selected_files = files
            self.selected_files = natsorted(self.selected_files)
            self.ui.ConsoleTextBrowser.append("The following CEST images were loaded.")
            self._display_file_tree(self.selected_files)
            self.ui.ConsoleTextBrowser.append("\n")
    
    # ---- CEST Offsets Selector ---- #
    def _select_offsets(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select CEST Offsets",
            "",
            "Text Files (*.txt)"
        )
        if file:   # user didn't cancel
            with open(file, 'r') as f:
                self.offsets = [float(line.strip()) for line in f if not('#' in line)]
                self.offsets.insert(0, -300.0)
            self.ui.ConsoleTextBrowser.append("The following offset file was loaded.")
            self._display_file_tree([file])
            self.ui.ConsoleTextBrowser.append(f"Offsets: {str(self.offsets)}")
            self.ui.ConsoleTextBrowser.append("\n")

    # ---- Mask File Selector ---- #
    def _select_mask(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mask",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        if file:    # user didn't cancel
            self.mask_file = file
            self.ui.ConsoleTextBrowser.append("The following mask file was loaded.")
            self._display_file_tree([file])
            self.ui.ConsoleTextBrowser.append("\n")

    # --- Save File Methods ---- #
    def _validate_filename(self):
        text = self.ui.FilenameLineEdit.text().strip()

        if not text:
            self.ui.SaveFileButton.setEnabled(False)
            return
        # Split into directory + filename
        directory, filename = os.path.split(text)

        # Check filename is not empty
        if not filename:
            self.ui.SaveFileButton.setEnabled(False)
            return

        # Optional: reject illegal characters (Windows-safe)
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            self.ui.SaveFileButton.setEnabled(False)
            return

        # If directory is provided, check it exists
        if directory and not os.path.exists(directory):
            self.ui.SaveFileButton.setEnabled(False)
            return

        self.ui.SaveFileButton.setEnabled(True)   

    def _save_file(self):

        filename = self.ui.FilenameLineEdit.text().strip().replace(".nii.gz", "").replace(".nii", "")

        self.ui.ConsoleTextBrowser.append("Loading data ...")
        cest_imgs = [nib.load(file) for file in self.selected_files]
        cest_data = [img.get_fdata() for img in cest_imgs]


        self.ui.ConsoleTextBrowser.append("Concatenating data ...")
        stacked_cest_data = np.stack(cest_data, axis=3)
        stacked_cest_img  = nib.Nifti1Image(stacked_cest_data, cest_imgs[0].affine, cest_imgs[0].header)
        
        nib.save(stacked_cest_img, f"{filename}.nii.gz")
        self.ui.ConsoleTextBrowser.append(f" | File saved to: {filename}.nii.gz")
        self.ui.ConsoleTextBrowser.append("\n")

        if "Yes (uses mcflirt, requires FSL)" in self.ui.MoCoComboBox.currentText():
            self.ui.ConsoleTextBrowser.append("Motion correction of data (uses FSL mcflirt) ...")
            cmd = f"mcflirt \
                -in '{filename}' \
                -out '{filename}_mcf' \
                -meanvol \
                -stages 4 \
                -report"
            os.system(cmd)

            stacked_cest_img = nib.load(f"{filename}_mcf.nii.gz")
            stacked_cest_data = stacked_cest_img.get_fdata()
            unstacked_cest_data = np.unstack(stacked_cest_data, axis=3)
            unstacked_cest_imgs = [
                nib.Nifti1Image(data, cest_imgs[0].affine, cest_imgs[0].header) for data in unstacked_cest_data
            ]
            cest_imgs = unstacked_cest_imgs

            filename = filename + '_mcf'
        elif "Yes (uses utils/cest_moco.py)" in self.ui.MoCoComboBox.currentText():
            self.ui.ConsoleTextBrowser.append("Motion correction of data (uses cest_moco.py) ...")
            run_cest_moco(
                input_nii  = f"{filename}.nii.gz",
                output_nii = f"{filename}_moco.nii.gz",
                reference_volume_index=0,
                offsets_ppm= np.array(self.offsets),
            )

            stacked_cest_img = nib.load(f"{filename}_moco.nii.gz")
            stacked_cest_data = stacked_cest_img.get_fdata()
            unstacked_cest_data = np.unstack(stacked_cest_data, axis=3)
            unstacked_cest_imgs = [
                nib.Nifti1Image(data, cest_imgs[0].affine, cest_imgs[0].header) for data in unstacked_cest_data
            ]
            cest_imgs = unstacked_cest_imgs

            filename = filename + '_moco'

        self.ui.ConsoleTextBrowser.append("Masking data ...")
        self.ui.ConsoleTextBrowser.append(f" | Mask loaded from: {self.mask_file}")
        mask_img  = nib.load(self.mask_file)

        cest_data_masked = []
        cest_masks = []
        for img in cest_imgs:
            cest_img, cest_mask = crop_to_common_overlap(img, mask_img, resample=True)
            cest_data_masked.append(cest_img.get_fdata() * cest_mask.get_fdata())
            cest_masks.append(cest_mask.get_fdata())

        stacked_cest_data_masked = np.stack(cest_data_masked, axis=3)
        stacked_cest_img_masked  = nib.Nifti1Image(stacked_cest_data_masked, cest_imgs[0].affine, cest_imgs[0].header)

        nib.save(stacked_cest_img_masked, f"{filename}_masked.nii.gz")
        self.ui.ConsoleTextBrowser.append(f" | File saved to: {filename}_masked.nii.gz")
        self.ui.ConsoleTextBrowser.append("\n")


        self.ui.ConsoleTextBrowser.append("Normalizing masked data ...")
        
        cest_data_normalized = []
        for i, data in enumerate(cest_data_masked):
            data_normalized = data / cest_data_masked[0]        # Normalize by M0 image
            data_normalized = data_normalized * cest_masks[i]   # Reapply mask
            data_normalized = np.nan_to_num(data_normalized, nan=0.0, posinf=0.0, neginf=0.0)   # Handle NaNs
            cest_data_normalized.append(data_normalized)
        
        stacked_cest_data_normalized = np.stack(cest_data_normalized, axis=3)[..., 1:]
        stacked_cest_img_normalized  = nib.Nifti1Image(stacked_cest_data_normalized, cest_imgs[0].affine, cest_imgs[0].header)

        nib.save(stacked_cest_img_normalized, f"{filename}_masked_normalized.nii.gz")
        self.ui.ConsoleTextBrowser.append(f" | File saved to: {filename}_masked_normalized.nii.gz")
        self.ui.ConsoleTextBrowser.append("\n")

    # ---- Display Methods ---- #
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

    window = ConcatApp()
    window.show()
    sys.exit(app.exec())