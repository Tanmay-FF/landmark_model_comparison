import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os 

import tensorflow as tf

import onnxruntime as ort
import onnx

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)
])

print("Template Size:" ,TEMPLATE.shape)
# Assuming the pre-built function that runs the model is available
# def run_model(image_path: str, model: ort.InferenceSession) -> dict:
#     # This is your function that returns landmarks for an image
#     pass

# Assuming you have a function to plot landmarks as follows:
# def plot_landmarks(landmark_dict: dict):
#     # Function that takes a dictionary and plots the landmarks on the images
#     pass

def load_pb_model(model_path):
    # Load the protobuf file from disk
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import the graph into a new Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph

def run_tf_inference(model_path, input_data, input_tensor_name, output_tensor_name):\

    # Load the frozen graph
    graph = load_pb_model(model_path)

    # Start a session and run inference
    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)

        # Run inference
        predictions = sess.run(output_tensor, feed_dict={input_tensor: input_data})

    return predictions

def create_onnx_session(model_path = 'model_landmarks.onnx'):
    onnx_model = onnx.load(model_path)
    onnx_options = ort.SessionOptions()
    return ort.InferenceSession(onnx_model.SerializeToString(),onnx_options)

def get_landmarks_on_an_image_path(image_path, 
                                   model_path, 
                                   just_use_key_landmarks = False):

    landmarks_dict = dict()

    # Example input data (adjust based on your model's expected input shape)
    # input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    input_data = cv2.imread(image_path)
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    input_data_resize = cv2.resize(input_data, (64, 64))
    input_data_resize = np.expand_dims(input_data_resize, 0)

    if '.pb' in model_path:
        model_type = 'TensorFlow'
        input_tensor_name = "images:0"
        output_tensor_name = "landmarks:0"
        landmarks = run_tf_inference(model_path, input_data_resize, input_tensor_name, output_tensor_name)
        landmarks = landmarks[0]

    if '.onnx' in model_path:
        model_type = 'ONNX'
        onnx_session = create_onnx_session(model_path)
        inputs= {onnx_session.get_inputs()[0].name:input_data_resize}
        ort_outputs= onnx_session.run(None, inputs)
        landmarks = np.array(ort_outputs).reshape(98,2)

    if just_use_key_landmarks:
        landmarks = [coordinates for i, coordinates in enumerate(landmarks) if i in {64, 68, 85}]

    landmarks_dict[image_path] = landmarks
    
    return landmarks_dict, model_type

def plot_landmarks_on_an_image(image_path, landmarks1, landmarks2, model_type1, model_type2):

    color1 = 'red'
    color2 = 'blue'
    template_color = 'lime'

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- LEFT IMAGE (TF model) ---
    axes[0].imshow(image_rgb)

    # TF landmarks
    for (y, x) in landmarks1:
        axes[0].scatter(x * w, y * h, c=color1, s=40, marker='x')

    # TEMPLATE overlay
    for i, (ty, tx) in enumerate(TEMPLATE):
        yy = ty * h
        xx = tx * w
        axes[0].scatter(xx, yy, c=template_color, s=25)
        axes[0].annotate(str(i), (xx, yy), color='yellow', fontsize=7, ha='center', va='center')

    axes[0].set_title(f"{model_type1} + TEMPLATE")
    axes[0].axis("off")

    # --- RIGHT IMAGE (ONNX model) ---
    axes[1].imshow(image_rgb)

    # ONNX landmarks
    for i, (y, x) in enumerate(landmarks2):
        axes[1].scatter(x * w, y * h, c=color2, s=40)
        axes[1].annotate(str(i), (x * w, y * h), color='yellow', fontsize=8, ha='center', va='center')

    # TEMPLATE overlay
    for i, (tx, ty) in enumerate(TEMPLATE):
        yy = ty * h
        xx = tx * w
        axes[1].scatter(xx, yy, c=template_color, s=25)
        axes[1].annotate(str(i), (xx, yy), color='yellow', fontsize=7, ha='center', va='center')

    axes[1].set_title(f"{model_type2} + TEMPLATE")
    axes[1].axis("off")

    plt.subplots_adjust(wspace=0)
    plt.show()


def select_onnx_model():
    file_path = filedialog.askopenfilename(filetypes=[("ONNX Model", "*.onnx")])
    if file_path:
        onnx_path_var.set(file_path)

def select_tensorflow_model():
    file_path = filedialog.askopenfilename(filetypes=[("TensorFlow Model", "*.pb")])
    if file_path:
        tf_path_var.set(file_path)

def select_image_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_path_var.set(folder_path)

def run_or_remove_landmarks():
    tf_path = tf_path_var.get()
    onnx_path = onnx_path_var.get()
    folder_path = folder_path_var.get()

    print("tf_path: ", tf_path)
    print("onnx_path: ", onnx_path)
    print("folder_path: ", folder_path)

    if not tf_path or not folder_path or not onnx_path:
        messagebox.showerror("Error", "Please select TF model, ONNX model, and an image folder.")
        return

    # Get all images in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        messagebox.showerror("Error", "No images found in the selected folder.")
        return

    # Dictionary to store image paths and their corresponding landmarks
    tf_landmark_dict = {}
    onnx_landmark_dict = {}
    image_paths = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        tf_img_dict, tf_model_type = get_landmarks_on_an_image_path(image_path, tf_path)  # Call your function here
        tf_landmark_dict[image_path] = tf_img_dict[image_path]

        onnx_img_dict, onnx_model_type = get_landmarks_on_an_image_path(image_path, onnx_path)  # Call your function here
        onnx_landmark_dict[image_path] = onnx_img_dict[image_path]

        image_paths.append(image_path)

    # Plot landmarks (or remove them if checkbox is unchecked)
    for image_path in image_paths:
        tf_landmarks = tf_landmark_dict[image_path]
        onnx_landmarks = onnx_landmark_dict[image_path]
        plot_landmarks_on_an_image(image_path, tf_landmarks, onnx_landmarks, tf_model_type, onnx_model_type)  # Call your function to update the images with or without landmarks

# GUI setup
root = tk.Tk()
root.title("Landmarks Comparison")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# TF Model selection
tf_path_var = tk.StringVar()
tf_button = ttk.Button(frame, text="Select TF Model", command=select_tensorflow_model)
tf_button.grid(row=0, column=0, padx=5, pady=5)
tf_label = ttk.Label(frame, textvariable=tf_path_var, width=50)
tf_label.grid(row=0, column=1, padx=5, pady=5)

# ONNX Model selection
onnx_path_var = tk.StringVar()
onnx_button = ttk.Button(frame, text="Select ONNX Model", command=select_onnx_model)
onnx_button.grid(row=1, column=0, padx=5, pady=5)  # Move ONNX button to row 1
onnx_label = ttk.Label(frame, textvariable=onnx_path_var, width=50)
onnx_label.grid(row=1, column=1, padx=5, pady=5)

# Image folder selection
folder_path_var = tk.StringVar()
folder_button = ttk.Button(frame, text="Select Folder", command=select_image_folder)
folder_button.grid(row=2, column=0, padx=5, pady=5)  # Move folder button to row 2

folder_label = ttk.Label(frame, textvariable=folder_path_var, width=50)
folder_label.grid(row=2, column=1, padx=5, pady=5)

# Button to execute the process
run_button = ttk.Button(frame, text="Run", command=run_or_remove_landmarks)
run_button.grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()
