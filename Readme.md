

# Face Landmark Model Comparison Tool

This project is a GUI-based utility designed to visually compare the inference results of **TensorFlow** (`.pb`) and **ONNX** face landmark detection models side-by-side.

It allows developers to validate the accuracy of model conversions (e.g., TF to ONNX) by overlaying predicted landmarks against a reference template on input images.

## Features

  * **User-Friendly GUI:** Built with `tkinter`, allowing easy selection of models and image datasets without modifying code.
  * **Dual Model Support:** Run inference on both TensorFlow Frozen Graphs (`.pb`) and ONNX models (`.onnx`).
  * **Visual Comparison:** Generates side-by-side Matplotlib plots comparing the output of both models.
  * **Template Overlay:** Overlays a reference landmark template (lime green dots) to help assess the alignment accuracy of the predictions.
  * **Batch Processing:** Processes all images within a selected directory.


## Directory Structure

```text
.
├── models/               # Place your .pb and .onnx models here
├── sample_input/         # Directory containing test images (.jpg, .png)
├── sample_output/        # Directory for saving results (if configured)
├── landmarks_demo.py     # Main application script
├── requirements.txt      # Python dependencies
└── image_17_landamrks.png # Sample reference image
```


## Prerequisites

Ensure you have Python installed (3.7+ recommended). You will need the following libraries:

  * **TensorFlow:** For running the `.pb` model.
  * **ONNX & ONNX Runtime:** For running the `.onnx` model.
  * **OpenCV:** For image processing.
  * **Matplotlib:** For plotting the results.
  * **Pillow:** For GUI image handling.
  * **NumPy:** For array manipulations.

### Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

*(Note: If you have a GPU, you may want to install `onnxruntime-gpu` instead of `onnxruntime`)*

## Usage

1.  **Run the script:**

    ```bash
    python landmarks_demo.py
    ```

2.  **Select Files via GUI:**

      * **Select TF Model:** Choose your TensorFlow frozen graph file (`.pb`).
      * **Select ONNX Model:** Choose your converted ONNX model file (`.onnx`).
      * **Select Folder:** Choose a local directory containing the test images (`.jpg`, `.png`, etc.).

3.  **Run Comparison:**

      * Click the **Run** button.
      * The tool will iterate through the images in the folder.
      * A Matplotlib window will pop up for every image, showing:
          * **Left:** TensorFlow predictions (Red 'X') vs Template.
          * **Right:** ONNX predictions (Blue dots) vs Template.

## Configuration & Constraints [Important]

The script is currently configured for a specific model architecture. If you are using your own custom models, you may need to adjust the following hardcoded values in the script:

1.  **Input Size:** The script resizes images to **64x64** before inference.
      * *Location:* `cv2.resize(input_data, (64, 64))` inside `get_landmarks_on_an_image_path`.
2.  **Tensor Names:** The TensorFlow inference function expects specific tensor names.
      * *Input:* `"images:0"`
      * *Output:* `"landmarks:0"`
      * *Location:* `get_landmarks_on_an_image_path` function.
3.  **Landmark Count:** The ONNX reshape function expects **98 landmarks** (196 coordinates).
      * *Location:* `.reshape(98,2)` inside `get_landmarks_on_an_image_path`.

