#!/usr/bin/env python
import torch
import numpy
import sys

from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QDialog
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QButtonGroup
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QSpinBox, QFrame, QRadioButton
from PySide6.QtGui import QColor, QPixmap, QVector3D
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PIL import ImageQt

from posegen import SimpleDiffusionPoseGen, SDPG_IMGSIZE


class SDPoseGen3DViewer(Qt3DExtras.Qt3DWindow):
  '''
  Viewer for drawing 3D pose
  '''
  def __init__(self):
    super().__init__()
    self.defaultFrameGraph().setClearColor(QColor(72,72,72))
    self.root_entity = Qt3DCore.QEntity()
    self.material = Qt3DExtras.QPhongMaterial(self.root_entity)
    self.material.setDiffuse(QColor.fromRgbF(0.8, 0.7, 0.6))
    self.material.setAmbient(QColor.fromRgbF(0.3, 0.2, 0.4))
    self.setRootEntity(self.root_entity)
    self._pose_entities = []
    self.set_up_camera()

  def set_up_camera(self):
    self.camera().lens().setPerspectiveProjection(24, 1.0, 0.1, 1000)
    self.camera().setPosition(QVector3D(0, 0, 5))
    self.camera().setViewCenter(QVector3D(0, 0, 0))
    self.camera().setUpVector(QVector3D(0, 1, 0))
    self.cam_ctrl = Qt3DExtras.QOrbitCameraController(self.root_entity)
    self.cam_ctrl.setLinearSpeed(50)
    self.cam_ctrl.setLookSpeed(180)
    self.cam_ctrl.setCamera(self.camera())

  def draw_pose(self, segments):
    self._pose_entities.clear()
    visited_markers = set()
    u = QVector3D(0,1,0)

    def add_entity(mesh, xform, shader=self.material):
      entity = Qt3DCore.QEntity(self.root_entity)
      entity.addComponent(mesh)
      entity.addComponent(xform)
      entity.addComponent(shader)
      self._pose_entities.append((entity, xform, mesh))

    def add_link(x, y, radius=0.01):
      mesh = Qt3DExtras.QCylinderMesh()
      mesh.setRadius(radius)
      v = y - x
      mesh.setLength(v.length())
      xform = Qt3DCore.QTransform()
      v = v.normalized()
      theta = numpy.arccos(QVector3D.dotProduct(u,v))
      v = QVector3D.crossProduct(u,v)
      q = Qt3DCore.QTransform.fromAxisAndAngle(v, numpy.degrees(theta)) # Not radians!?
      xform.setRotation(q)
      xform.setTranslation(0.5 * (x + y))
      add_entity(mesh, xform)

    def add_marker(i, x, radius=0.0125):
      mesh = Qt3DExtras.QSphereMesh()
      mesh.setRadius(radius)
      xform = Qt3DCore.QTransform()
      xform.setTranslation(x)
      add_entity(mesh, xform)
      visited_markers.add(i)

    for seg in segments:
      i, j = seg.keys()
      x = QVector3D(*seg[i])
      y = QVector3D(*seg[j])
      add_link(x, y)
      if i not in visited_markers:
        add_marker(i, x)
      if j not in visited_markers:
        add_marker(j, y)


class SDPoseGenWidget(QWidget):
  '''
  Main window for SD Pose Gen 
  '''
  def __init__(self):
    super().__init__()

    # Pop-up dialog for SD model settings
    sddg = QDialog(self)
    sddg.setWindowTitle("Stable Diffusion Model Settings")
    sdl0 = QVBoxLayout()
    sdl1 = QHBoxLayout()
    sd15 = QRadioButton("Stable Diffusion v1.5"); sdxl = QRadioButton("Stable Diffusion XL")
    sd15.setChecked(True)
    sdg1 = QButtonGroup()
    sdg1.addButton(sd15); sdg1.addButton(sdxl)
    sdl1.addWidget(sd15); sdl1.addWidget(sdxl)

    sdl2 = QHBoxLayout()
    sdl2.addWidget(QLabel('Hyper-SD Steps:'))
    sdh2 = QRadioButton('2'); sdh4 = QRadioButton('4'); sdh8 = QRadioButton('8')
    sdh8.setChecked(True)
    sdg2 = QButtonGroup()
    sdg2.addButton(sdh2); sdg2.addButton(sdh4); sdg2.addButton(sdh8)
    sdl2.addWidget(sdh2); sdl2.addWidget(sdh4); sdl2.addWidget(sdh8)

    sdl3 = QHBoxLayout()
    sdl3.addWidget(QLabel('Device:'))
    sdd1 = QRadioButton('CPU'); sdd2 = QRadioButton('CUDA'); sdd3 = QRadioButton('MPS')
    sdd1.setChecked(True)
    sdg3 = QButtonGroup()
    sdg3.addButton(sdd1); sdg3.addButton(sdd2); sdg3.addButton(sdd3)
    sdl3.addWidget(sdd1); sdl3.addWidget(sdd2); sdl3.addWidget(sdd3)
    if torch.cuda.is_available():
      sdd2.setChecked(True)
    elif torch.backends.mps.is_available():
      sdd3.setChecked(True)
    
    sdbn = QPushButton('OK')
    sdbn.clicked.connect(lambda x: sddg.done(QDialog.Accepted))

    sdl0.addLayout(sdl1); sdl0.addLayout(sdl2); sdl0.addLayout(sdl3); sdl0.addWidget(sdbn)
    sddg.setLayout(sdl0)
    if sddg.exec():
      base = 'SDXL' if sdxl.isChecked() else 'SD15'
      steps = 2 if sdh2.isChecked() else (4 if sdh4.isChecked() else 8)
      device = 'cuda' if sdd2.isChecked() else ('mps' if sdd3.isChecked() else 'cpu')
      self._pose_gen = SimpleDiffusionPoseGen(steps, base, device)

    # Main window
    vbox = QVBoxLayout()
    # Viewer widgets
    hboxa = QHBoxLayout()
    self.pixmap = QLabel()
    self.pixmap.setFrameStyle(QFrame.Box | QFrame.Raised)
    self.pixmap.setFixedSize(SDPG_IMGSIZE, SDPG_IMGSIZE)
    hboxa.addWidget(self.pixmap)
    self.view = SDPoseGen3DViewer()
    container = QWidget.createWindowContainer(self.view)
    container.setFixedSize(SDPG_IMGSIZE, SDPG_IMGSIZE)
    hboxa.addWidget(container)
    vbox.addLayout(hboxa)
    # Control widgets
    hboxb = QHBoxLayout()
    hboxb.addWidget(QLabel('Prompt'))
    self.prompt = QLineEdit('A basketball player making a 3-pointer jump shot')
    hboxb.addWidget(self.prompt)
    hboxb.addWidget(QLabel('Seed'))
    self.seed = QSpinBox()
    self.seed.setRange(1, 9999)
    self.seed.setValue(4093)
    hboxb.addWidget(self.seed)
    button = QPushButton('Generate')
    button.clicked.connect(self.on_generate)
    hboxb.addWidget(button)
    vbox.addLayout(hboxb)
    self.setLayout(vbox)

  def on_generate(self):
    image, pose_segments = self._pose_gen(self.prompt.text(), self.seed.value())
    self.pixmap.setPixmap(QPixmap.fromImage(ImageQt.ImageQt(image)))
    self.view.draw_pose(pose_segments)


if __name__ == "__main__":
  app = QApplication(sys.argv)
  win = QMainWindow()
  win.setCentralWidget(SDPoseGenWidget())
  win.setWindowTitle('Simple Diffusion Pose Gen')
  win.show()
  sys.exit(app.exec())