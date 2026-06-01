#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-"$ROOT/.venv/bin/python"}"

export DISPLAY="${DISPLAY:-:2}"
export LD_LIBRARY_PATH="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$ROOT/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$ROOT/.venv/lib/python3.12/site-packages/nvidia/cusparse/lib:$ROOT/.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:$ROOT/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"
"$PYTHON" - "$@" <<'PY'
import os
import runpy
import sys
import time
from pathlib import Path

import pyglet.display.xlib as xlib

if not hasattr(xlib.XlibScreen, "is_primary"):
    xlib.XlibScreen.is_primary = property(lambda self: False)

if os.environ.get("GENESIS_VIEWER_NO_TK_DIALOG", "1") != "0":
    import genesis as gs
    from genesis.ext.pyrender.viewer import Viewer
    from genesis.vis.viewer_plugins.plugins.default_controls import DefaultControlsPlugin

    def _next_viewer_output_path(ext):
        configured_filename = os.environ.get("GENESIS_VIEWER_SAVE_FILE")
        if configured_filename:
            filename = Path(configured_filename).expanduser()
        else:
            save_dir = Path(os.environ.get("GENESIS_VIEWER_SAVE_DIR", "recordings")).expanduser()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"genesis_viewer_{timestamp}.{ext}"

        filename.parent.mkdir(parents=True, exist_ok=True)
        return str(filename)

    def _get_save_filename_without_dialog(self, file_exts):
        return _next_viewer_output_path(file_exts[0])

    def _toggle_record_video_without_dialog(self):
        if self.viewer.viewer_flags["record"]:
            self.viewer._video_recorder.close()
            filename = getattr(self.viewer, "_video_recorder_output_filename", None)
            self.viewer._video_recorder = None
            self.viewer._video_recorder_output_filename = None
            self.viewer.viewer_flags["record"] = False
            self.viewer.set_caption(self.viewer.viewer_flags["window_title"])
            if filename is not None:
                self.viewer.set_message_text(f"Saved video: {filename}")
                gs.logger.info(f"Saved viewer recording to {filename}")
            return

        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

        filename = _next_viewer_output_path("mp4")
        self.viewer._video_recorder_output_filename = filename
        self.viewer._video_recorder = FFMPEG_VideoWriter(
            filename=filename,
            fps=self.viewer.viewer_flags["refresh_rate"],
            size=self.viewer.viewport_size,
        )
        self.viewer.viewer_flags["record"] = True
        self.viewer.set_caption(f"{self.viewer.viewer_flags['window_title']} (RECORDING)")
        self.viewer.set_message_text(f"Recording: {filename}")
        gs.logger.info(f"Started viewer recording to {filename}")

    Viewer._get_save_filename = _get_save_filename_without_dialog
    DefaultControlsPlugin._toggle_record_video = _toggle_record_video_without_dialog

sys.argv = ["examples/IPC_Solver/ipc_robot_cloth_teleop.py", *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
PY
