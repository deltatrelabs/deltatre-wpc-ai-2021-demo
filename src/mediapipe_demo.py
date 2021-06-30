import click
import multiprocessing
from pathlib import Path
from mediapipe_utils import mediapipe_multiproc

@click.command()
@click.option('--input_file',
              type=Path,
              required=True,
              help='Video file to process')
@click.option('--output_folder',
              type=Path,
              required=True,
              default='./outputs',
              help='Output path')
@click.option('--model_type',
              type=click.STRING,
              required=True,
              help='MediaPipe model to use for inference')
@click.option('--processor_count',
              type=click.INT,
              required=False,
              default=1,
              help='Number of CPUs to use in multi-processing mode')
@click.option('--delete_temp_files',
              type=click.BOOL,
              required=False,
              default=True,
              help='If true, delete temporary partial files')
@click.option('--min_detection_confidence',
              type=click.FLOAT,
              required=False,
              default=0.5,
              help='Set confidence threshold for detection models')
@click.option('--min_tracking_confidence',
              type=click.FLOAT,
              required=False,
              default=0.5,
              help='Set confidence threshold for tracking models')
def mediapipe_demo(input_file: Path, output_folder: Path, model_type: str, processor_count: int, delete_temp_files: bool, min_detection_confidence: float, min_tracking_confidence: float):
    
    if processor_count == -1:
        processor_count = multiprocessing.cpu_count()

    mediapipe_multiproc(input_file, output_folder, processor_count, model_type, delete_temp_files, min_detection_confidence, min_tracking_confidence)

if __name__ == '__main__':
    mediapipe_demo()
