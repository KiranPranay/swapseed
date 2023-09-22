import os
import matplotlib.pyplot as plt
import gdown
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from faceswap import swap_n_show, swap_n_show_same_img, swap_face_single,fine_face_swap

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Download 'inswapper_128.onnx' file using gdown
model_url = 'https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu'
model_output_path = 'inswapper/inswapper_128.onnx'
if not os.path.exists(model_output_path):
    gdown.download(model_url, model_output_path, quiet=False)

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Load images
img1_fn = 'images/Anushka.jpg'
img2_fn = 'images/keerthi.jpg'

# Swap faces between two images
# swap_n_show(img1_fn, img2_fn, app, swapper)

# Swap faces within the same image 
# swap_n_show_same_img(img1_fn, app, swapper)

# Add face to an image
swap_face_single(img1_fn, img2_fn, app, swapper, enhance=True, enhancer='REAL-ESRGAN 2x',device="cpu")

# Fine face swapper
fine_face_swap(img1_fn, img2_fn, app, swapper, enhance=True, enhancer='REAL-ESRGAN 2x',device="cpu")