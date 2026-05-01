import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg


class ZSpectrumPlotter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Main plot (Z-spectrum + fits)
        self.main_plot = pg.PlotWidget(title="Z-Spectrum")
        self.main_plot.addLegend()
        self.main_plot.setLabel('left', 'Signal')
        self.main_plot.setLabel('bottom', 'Frequency Offset (ppm)')
        self.main_plot.invertX(True)
        self.main_plot.setYRange(0, 1)

        # Residual plot
        self.residual_plot = pg.PlotWidget(title="Residual")
        self.residual_plot.setLabel('bottom', 'Frequency Offset (ppm)')
        self.residual_plot.invertX(True)

        layout.addWidget(self.main_plot)
        layout.addWidget(self.residual_plot)
        self.residual_plot.setMaximumHeight(200)

        # Store plot items
        self.data_curve = None
        self.fit_curve = None
        self.component_curves = []
        self.residual_curve = None

    def update_plot(self, x_raw, z_data, x_fit=None, fit=None, components=None, residual=None):
        self.main_plot.clear()
        self.main_plot.addLegend()

        # Data
        self.main_plot.plot(
            x_raw, z_data,
            pen=None,
            symbol='o',
            name="Data"
        )

        # Fit
        if x_fit is not None and fit is not None:
            self.main_plot.plot(
                x_fit, fit,
                pen=pg.mkPen('r', width=2),
                name="Fit"
            )

        # Components (list of tuples)
        if x_fit is not None and components is not None:
            for i, (comp, name) in enumerate(components):
                self.main_plot.plot(
                    x_fit, comp,
                    pen=pg.mkPen(color=pg.intColor(i), style=pg.QtCore.Qt.DashLine),
                    name=name
                )

        # Residual
        self.residual_plot.clear()
        if residual is not None:
            self.residual_plot.plot(
                x_raw, residual,
                pen=pg.mkPen('y')
            )