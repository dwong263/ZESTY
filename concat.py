from PySide6.QtWidgets import (
    QApplication, QMainWindow,
    QFileDialog
)
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

import os, sys
import nibabel as nib
import numpy as np

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
            self.selected_files.sort()
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
        self.ui.ConsoleTextBrowser.append("Normalizing data ...")
        cest_imgs = [nib.load(file) for file in self.selected_files]
        cest_data = [img.get_fdata() for img in cest_imgs]

        mask_img  = nib.load(self.mask_file)
        mask_data = mask_img.get_fdata()

        cest_data_normalized = []
        for data in cest_data:
            # Normalize by M0 image (first image)
            data_normalized = (data * mask_data) / cest_data[0]
            # Handle division by zero and NaN values
            data_normalized = np.nan_to_num(data_normalized, nan=0.0, posinf=0.0, neginf=0.0)
            cest_data_normalized.append(data_normalized)

        self.ui.ConsoleTextBrowser.append("Stacking data ...")
        # Stack data and discard -300 ppm M0 image
        stacked_cest_data_normalied = np.stack(cest_data_normalized, axis=3)[..., 1:] 
        stacked_cest_img_normalized = nib.Nifti1Image(stacked_cest_data_normalied, cest_imgs[0].affine, cest_imgs[0].header)

        nib.save(stacked_cest_img_normalized, self.ui.FilenameLineEdit.text().strip())
        self.ui.ConsoleTextBrowser.append(f"File saved to: {self.ui.FilenameLineEdit.text().strip()}")
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