# PyInstaller: приложение «Arx Nova» (окно + фоновый сервер в одном файле).
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

hidden = (collect_submodules("uvicorn") + collect_submodules("arxnova")
          + collect_submodules("zeroconf") + ["soxr", "soundfile"])
datas = collect_data_files("_soundfile_data") + collect_data_files("onnxruntime")
binaries = collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("_soundfile_data")

a = Analysis(["entry.py"], pathex=["../src"], hiddenimports=hidden, datas=datas, binaries=binaries,
             excludes=["torch", "transformers", "matplotlib", "pytest"])
pyz = PYZ(a.pure)
icon = "icon.icns" if sys.platform == "darwin" else ("icon.ico" if sys.platform == "win32" else None)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Arx Nova" if sys.platform != "linux" else "arxnova",
          console=False if sys.platform != "linux" else True, icon=icon)
coll = COLLECT(exe, a.binaries, a.datas, name="Arx Nova" if sys.platform != "linux" else "arxnova")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="Arx Nova.app", icon=icon, bundle_identifier="com.arxnova.server",
                 info_plist={"CFBundleShortVersionString": "0.1.0", "NSHighResolutionCapable": True,
                             "LSMinimumSystemVersion": "12.0"})
