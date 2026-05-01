import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal
import pyqtgraph as pg


class TriPlanarViewer(QWidget):
    voxelSelected = Signal(int, int, int)

    def __init__(self, spacing=(1.0, 1.0, 1.0), cbar=False, parent=None):
        super().__init__(parent)

        self.volume = None
        self.x = self.y = self.z = 0
        self.cbar = cbar
        self.spacing = spacing

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create views
        self.axial_view = pg.ImageView()
        self.sagittal_view = pg.ImageView()
        self.coronal_view = pg.ImageView()

        # Remove unecessary menus
        self._simplify_imageview(self.axial_view)
        self._simplify_imageview(self.sagittal_view)
        self._simplify_imageview(self.coronal_view)

        # Set radiological orientation for each view
        self._set_radiological_orientation()

        layout.addWidget(self.axial_view)
        layout.addWidget(self.sagittal_view)
        layout.addWidget(self.coronal_view)

        # Crosshair lines
        self._init_crosshairs()

        # Colorbar
        if self.cbar:
            colorbar_layout = pg.GraphicsLayoutWidget()
            self.colorbar = pg.ColorBarItem(
                values=(0,1),
                orientation="horizontal"
            )
            colorbar_layout.addItem(self.colorbar)
            colorbar_layout.setMaximumHeight(50)
            colorbar_layout.setMaximumWidth(400)
            layout.addWidget(colorbar_layout)

        # Mouse events
        self.axial_view.getImageItem().mouseClickEvent = self._axial_click
        self.sagittal_view.getImageItem().mouseClickEvent = self._sagittal_click
        self.coronal_view.getImageItem().mouseClickEvent = self._coronal_click

    # -------------------------
    # Orientation & aspect ratio
    # -------------------------
    def _set_radiological_orientation(self):
        # Axial and coronal: flip X so patient-right appears on viewer-left
        # (standard radiological convention)
        self.axial_view.getView().invertX(True)
        self.coronal_view.getView().invertX(True)
 
        # All views: origin at bottom-left so S (superior) is at the top
        self.axial_view.getView().invertY(False)
        self.sagittal_view.getView().invertY(False)
        self.coronal_view.getView().invertY(False)
 
    def _apply_spacing_transforms(self):
        sx, sy, sz = self.spacing
        # Set aspect ratio from voxel spacing ratios so each view displays
        # pixels at their correct proportions without scaling to mm units.
        # Axial:    horizontal=X, vertical=Y
        # Sagittal: horizontal=Y, vertical=Z
        # Coronal:  horizontal=X, vertical=Z
        self.axial_view.getView().setAspectLocked(True, ratio=sx / sy)
        self.sagittal_view.getView().setAspectLocked(True, ratio=sy / sz)
        self.coronal_view.getView().setAspectLocked(True, ratio=sx / sz)

    # -------------------------
    # Crosshairs
    # -------------------------
    def _init_crosshairs(self):
        def make_crosshair(view):
            vline = pg.InfiniteLine(angle=90, movable=False, pen='r')
            hline = pg.InfiniteLine(angle=0, movable=False, pen='r')
            view.addItem(vline)
            view.addItem(hline)
            return vline, hline

        self.ax_v, self.ax_h = make_crosshair(self.axial_view.getView())
        self.sg_v, self.sg_h = make_crosshair(self.sagittal_view.getView())
        self.cr_v, self.cr_h = make_crosshair(self.coronal_view.getView())

    def _update_crosshairs(self):
        # Positions are in voxel indices

        # Axial: (x, y)
        self.ax_v.setPos(self.x)
        self.ax_h.setPos(self.y)

        # Sagittal: (y, z)
        self.sg_v.setPos(self.y)
        self.sg_h.setPos(self.z)

        # Coronal: (x, z)
        self.cr_v.setPos(self.x)
        self.cr_h.setPos(self.z)
    
    def get_crosshairs(self):
        return self.x, self.y, self.z

    # -------------------------
    # Load data
    # -------------------------
    def set_volume(self, volume, spacing=(1.0, 1.0, 1.0), x=None, y=None, z=None):
        self.volume = volume
        self.spacing = tuple(float(s) for s in spacing)

        if x is None:
            self.x = volume.shape[0] // 2
        else:
            self.x = x

        if y is None:
            self.y = volume.shape[1] // 2
        else:
            self.y = y
        
        if z is None:
            self.z = volume.shape[2] // 2
        else:
            self.z = z

        self._apply_spacing_transforms()
        self.update_views()

    # -------------------------
    # Update display
    # -------------------------
    def _simplify_imageview(self, view):
        view.ui.roiBtn.hide()
        view.ui.menuBtn.hide()
        view.ui.histogram.hide()

    def update_views(self, x=None, y=None, z=None):
        if self.volume is None:
            return

        if x is None:
            x = self.x
        else:
            self.x = x

        if y is None:
            y = self.y
        else:
            self.y = y

        if z is None:
            z = self.z
        else:
            self.z = z

        axial = self.volume[:, :, z]
        sagittal = self.volume[x, :, :]
        coronal = self.volume[:, y, :]

        self.axial_view.setImage(axial, autoLevels=False)
        self.sagittal_view.setImage(sagittal, autoLevels=False)
        self.coronal_view.setImage(coronal, autoLevels=False)

        self._update_crosshairs()

    # -------------------------
    # Mouse handling
    # -------------------------
    def _map_to_image(self, view, pos):
        vb = view.getView()
        mouse_point = vb.mapSceneToView(pos)
        return int(mouse_point.x()), int(mouse_point.y())

    def _axial_click(self, event):
        x, y = self._map_to_image(self.axial_view, event.scenePos())

        self.x = np.clip(x, 0, self.volume.shape[0]-1)
        self.y = np.clip(y, 0, self.volume.shape[1]-1)

        self.update_views()
        self.voxelSelected.emit(self.x, self.y, self.z)

    def _sagittal_click(self, event):
        y, z = self._map_to_image(self.sagittal_view, event.scenePos())

        self.y = np.clip(y, 0, self.volume.shape[1]-1)
        self.z = np.clip(z, 0, self.volume.shape[2]-1)

        self.update_views()
        self.voxelSelected.emit(self.x, self.y, self.z)

    def _coronal_click(self, event):
        x, z = self._map_to_image(self.coronal_view, event.scenePos())

        self.x = np.clip(x, 0, self.volume.shape[0]-1)
        self.z = np.clip(z, 0, self.volume.shape[2]-1)

        self.update_views()
        self.voxelSelected.emit(self.x, self.y, self.z)