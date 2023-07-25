# Face Swapping with InsightFace and ONNX

This project demonstrates face swapping using the InsightFace library and ONNX model. It allows you to swap faces between two images or even swap faces within the same image.

<p align="center">
<img src="images/result.png" width="700px" alt="Face Swap Result">
</p>

## Installation

1. Clone the repository:

```bash
git clone https://github.com/KiranPranay/faceswap
cd faceswap
```

2. Install the required dependencies:

```pip
pip install requirements.txt
```

## Usage

### There are two main functions available for face swapping:

- swap_n_show(img1_fn, img2_fn, app, swapper, plot_before=True, plot_after=True): This function swaps faces between two input images.

- swap_n_show_same_img(img1_fn, app, swapper, plot_before=True, plot_after=True): This function swaps faces within the same image.

You can use these functions in your Python scripts or Jupyter notebooks.

## Example

```python
import cv2
import matplotlib.pyplot as plt
from faceswap import swap_n_show, swap_n_show_same_img

# Load images
img1_fn = 'images/bramhi.jpg'
img2_fn = 'images/modi.jpg'

# Swap faces between two images
swap_n_show(img1_fn, img2_fn, app, swapper)

# Swap faces within the same image
swap_n_show_same_img(img1_fn, app, swapper)
```

## Acknowledgments

This project uses the InsightFace library and ONNX model for face analysis and swapping. Thanks to the developers of these libraries for their contributions.

## License

[MIT License](https://github.com/KiranPranay/faceswap/blob/main/LICENSE)
