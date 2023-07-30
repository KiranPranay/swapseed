import os
import cv2
import matplotlib.pyplot as plt

from face_enhancer import load_face_enhancer_model

def validate_image(img):
    if not os.path.exists(img):
        raise ValueError(f'Image {img} does not exist')
    # check if img is a valid image file 
    if not os.path.isfile(img):
        raise ValueError(f'Image {img} is not a valid image file')
    # validate it to be jpg jpeg, png formats 
    if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise ValueError(f'Image {img} is not a valid image file')

def cpu_warning(device):
    if device == "cpu":
        print("Using CPU for face enhancer. If you have a GPU, you can set device='cuda' to speed up the process. You can also set enhance=False to skip the enhancement.")

def swap_n_show(img1_fn, img2_fn, app, swapper,
                plot_before=False, plot_after=True, enhance=False, enhancer='REAL-ESRGAN 2x',device="cpu"):
    
    validate_image(img1_fn)
    validate_image(img2_fn)
    
    img1 = cv2.imread(img1_fn)
    img2 = cv2.imread(img2_fn)
    
    if plot_before:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1[:,:,::-1])
        axs[0].axis('off')
        axs[1].imshow(img2[:,:,::-1])
        axs[1].axis('off')
        plt.show()
    
    # Do the swap
    face1 = app.get(img1)[0]
    face2 = app.get(img2)[0]
    
    img1_ = img1.copy()
    img2_ = img2.copy()
    if plot_after:
        img1_ = swapper.get(img1_, face1, face2, paste_back=True)
        img2_ = swapper.get(img2_, face2, face1, paste_back=True)
        if enhance:
            cpu_warning(device)
            model, model_runner = load_face_enhancer_model(enhancer,device)
            img1_ = model_runner(img1_, model)
            img2_ = model_runner(img2_, model)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1_[:,:,::-1])
        axs[0].axis('off')
        axs[1].imshow(img2_[:,:,::-1])
        axs[1].axis('off')
        plt.show()
    return img1_, img2_

def swap_n_show_same_img(img1_fn,
                         app, swapper,
                         plot_before=False,
                         plot_after=True, enhance=False, enhancer='REAL-ESRGAN 2x',device="cpu"):
    
    validate_image(img1_fn)
    img1 = cv2.imread(img1_fn)
    
    if plot_before:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(img1[:,:,::-1])
        ax.axis('off')
        plt.show()
    
    # Do the swap
    faces = app.get(img1)
    face1, face2 = faces[0], faces[1]
    
    img1_ = img1.copy()
    if plot_after:
        img1_ = swapper.get(img1_, face1, face2, paste_back=True)
        img1_ = swapper.get(img1_, face2, face1, paste_back=True)
        if enhance:
            cpu_warning(device)
            model, model_runner = load_face_enhancer_model(enhancer,device)
            img1_ = model_runner(img1_, model)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(img1_[:,:,::-1])
        ax.axis('off')
        plt.show()
    return img1_

def swap_face_single(img1_fn, img2_fn, app, swapper,
             plot_before=False, plot_after=True, enhance=False, enhancer='REAL-ESRGAN 2x',device="cpu"):
    
    validate_image(img1_fn)
    validate_image(img2_fn)
    
    img1 = cv2.imread(img1_fn)
    img2 = cv2.imread(img2_fn)
    
    if plot_before:
        axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1[:,:,::-1])
        axs[0].axis('off')
        axs[1].imshow(img2[:,:,::-1])
        axs[1].axis('off')
        plt.show()
    
    # Do the swap
    face1 = app.get(img1)[0]
    face2 = app.get(img2)[0]
    
    img1_ = img1.copy()
    if plot_after:
        img1_ = swapper.get(img1_, face1, face2, paste_back=True)
        if enhance:
            cpu_warning(device)
            model, model_runner = load_face_enhancer_model(enhancer,device)
            img1_ = model_runner(img1_, model)
        # Save the image
        output_fn = os.path.join('outputs', os.path.basename(img1_fn))
        cv2.imwrite(output_fn, img1_)
        print(f'Image saved to {output_fn}')
    return img1_
