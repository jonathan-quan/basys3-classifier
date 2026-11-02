"""The two FPGA designs the recording view can show. Per-layer cycle counts are data
independent; the classifier's come from its FSM (and are re-checked by telemetry), the
detector's from its RTL simulation log (fpga_det/build/simulation_core.log)."""
import json
from dataclasses import dataclass, field

from . import PROJECT
from .telemetry import LAYER_COUT, LAYER_HW, LAYER_NAMES, TOTAL_CYCLES, expected_stamps


@dataclass
class Profile:
    key: str
    title: str
    subtitle: str
    layer_names: tuple
    layer_cout: tuple
    layer_hw: tuple
    layer_ends: list
    total_cycles: int
    input_side: int
    camera_aspect: tuple
    delay: float
    live_timing: tuple            # intake, sweep, outro (s)
    replay_timing: tuple
    head_name: str
    head_detail: str
    placement: str
    threshold: float = 0.0
    names: tuple = ("Rubik's cube", 'Pen', 'Book', 'Mouse')

    @property
    def compute_ms(self):
        return self.total_cycles / 100_000


def classifier():
    return Profile(
        key='classifier', title='BASYS 3  ·  CNN RUNNING IN FPGA HARDWARE',
        subtitle='Webcam → USB UART → Artix-7 FPGA. The laptop only captures and displays; no neural network runs on it.',
        layer_names=LAYER_NAMES, layer_cout=LAYER_COUT, layer_hw=LAYER_HW, layer_ends=expected_stamps(),
        total_cycles=TOTAL_CYCLES, input_side=96, camera_aspect=(4, 3), delay=0.55,
        live_timing=(0.06, 0.22, 0.04), replay_timing=(0.4, 4.0, 0.4),
        head_name='Pool → FC → softmax', head_detail='48 → 4 logits, argmax, score', placement='placement')


def detector():
    summary = json.loads((PROJECT / 'detection' / 'runs' / 'webcam_ft_v1' / 'summary.json').read_text())
    return Profile(
        key='detector', title='BASYS 3  ·  OBJECT DETECTOR RUNNING IN FPGA HARDWARE',
        subtitle='Webcam → USB UART → Artix-7 FPGA: class + box from 4.78M multiply-adds per frame. No neural network runs on the laptop.',
        layer_names=('conv 3x3 /2', 'dw 3x3 /2', 'pw 1x1', 'dw 3x3 /2', 'pw 1x1', 'dw 3x3 /2', 'pw 1x1', 'dw 3x3', 'pw 1x1', 'dw 3x3', 'pw 1x1'),
        layer_cout=(16, 16, 32, 32, 64, 64, 96, 96, 96, 96, 96), layer_hw=(64, 32, 32, 16, 16, 8, 8, 8, 8, 8, 8),
        layer_ends=[7405619, 8077412, 10338501, 10674470, 12853735, 13021864, 14625737, 14877930, 17268235, 17520428, 19910733],
        total_cycles=20118456, input_side=128, camera_aspect=(16, 9), delay=1.1,
        live_timing=(0.10, 0.50, 0.08), replay_timing=(0.4, 6.0, 0.4),
        head_name='Detection head', head_detail='pool, class + presence, heat map, box',
        placement='placement_detector_viz' if (PROJECT / 'fpga_viz' / 'build' / 'placement_detector_viz' / 'placed_cells.csv').exists()
        else 'placement_detector', threshold=float(summary['score_threshold_selected_on_validation']))
