import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Assuming diffsynth is installed and accessible
try:
    from diffsynth.controlnets.processors import Annotator
except ImportError:
    print("Error: diffsynth module not found. Please install it or add it to your PYTHONPATH.")
    sys.exit(1)

def process_video(input_path, output_dir, fold, processor_id):
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the Annotator
    processor = Annotator(processor_id)

    # Open the video file
    video = cv2.VideoCapture(str(input_path))
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {input_path}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(total_frames), desc="Processing frames"):
        out_file = output_dir / f"{i:06d}.png"
        ret, frame = video.read()
        if i % fold[1] == fold[0]:
            if not ret:
                break
            if os.path.exists(out_file):
                continue
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Process the image
            pose_image = processor(image)

            # Save the processed image
            pose_image.save(out_file)

    video.release()
    print(f"Processed {total_frames} frames. Output saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process video and extract pose images.")
    # parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--fold", nargs=2, type=int, default=(0,1))
    parser.add_argument("--processor_id", default='openpose')
    args = parser.parse_args()
    from speedy import load_by_ext
    inputs = load_by_ext("datasets/tiktokdance/metadata_test.json")
    inputs = [x['path'] for x in inputs]
    print(inputs)
    for file in inputs:
        file = os.path.join('datasets/tiktokdance', file)
        print(file)
        input_path = Path(file)
        if not input_path.exists():
            print(f"Error: Input file '{input_path}' does not exist.")
            sys.exit(1)

        output_dir = input_path.with_suffix('') / "pose"
        
        try:
            process_video(input_path, output_dir, args.fold, args.processor_id)
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
