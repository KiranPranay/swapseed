# Face Swapping with InsightFace and ONNX

This project demonstrates face swapping using the InsightFace library and ONNX model. It allows you to swap faces between two images or even swap faces within the same image.

You can use the [available image Enhancers](#available-enhancers) to bring your output to the next level.

<p align="center">
<img src="images/result.png" width="700px" alt="Face Swap Result">
</p>
<p align="center">
<img src="images/swapseed.png" width="155" style="border-radius: 1em" alt="Face Swap Result">
</p>

## Installation

1. Clone the repository:

```bash
git clone https://github.com/KiranPranay/swapseed
cd swapseed
```

2. Install the required dependencies:

```pip
pip install -r requirements.txt
```

3. Execution

```python
 python main.py
```

## Usage

### There are three main functions available for face swapping:

- swap_n_show(img1_fn, img2_fn, app, swapper, plot_before=True, plot_after=True): This function swaps faces between two input images.

- swap_n_show_same_img(img1_fn, app, swapper, plot_before=True, plot_after=True): This function swaps faces within the same image.

- swap_face_single(img1_fn, img2_fn, app, swapper): This function adds face from the source image to the target image and saves in output/ folder.

- fine_face_swap(img1_fn, img2_fn, app, swapper): This function has ability to finely select faces from image with multiple faces.

You can use these functions in your Python scripts or Jupyter notebooks.

## Example

```python
import cv2
import matplotlib.pyplot as plt
from faceswap import swap_n_show, swap_n_show_same_img, swap_face_single

# Load images
img1_fn = 'images/bramhi.jpg'
img2_fn = 'images/modi.jpg'

# Swap faces between two images
swap_n_show(img1_fn, img2_fn, app, swapper, enhance=True, enhancer='REAL-ESRGAN 2x')

# Swap faces within the same image
swap_n_show_same_img(img1_fn, app, swapper, enhance=True, enhancer='REAL-ESRGAN 2x')

# Add face to an image
swap_face_single(img1_fn, img2_fn, app, swapper, enhance=True, enhancer='REAL-ESRGAN 2x')

# Swap faces in images with multiple faces
fine_face_swap(img1_fn, img2_fn, app, swapper, enhance=True, enhancer='REAL-ESRGAN 2x')
```

## Available Enhancers

- GFPGAN
- REAL-ESRGAN 2x
- REAL-ESRGAN 4x
- REAL-ESRGAN 8x

## GPU Support

- cuda
  **_(set 'device=cuda' to run with gpu)_**

## Acknowledgments

This project uses the InsightFace library and ONNX model for face analysis and swapping. Thanks to the developers of these libraries for their contributions.

- [Insightface](https://github.com/deepinsight)
- [Real-ESRGAN (ai-forever)](https://github.com/ai-forever/Real-ESRGAN)

## License

[MIT License](https://github.com/KiranPranay/faceswap/blob/main/LICENSE)

## Disclaimmer

**This project is for educational purposes only. The face swapping techniques demonstrated here are intended to showcase the capabilities of the InsightFace library and ONNX model for educational and research purposes. The project should not be used for any malicious or illegal activities.**

---

<b> If you like my content or find anything useful, give it a :star: or support me by buying me a coffee :coffee::grinning: </b>

<a href='https://ko-fi.com/R6R57A2ZT' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi3.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
