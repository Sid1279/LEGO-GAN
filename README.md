# LEGO-GAN
### A deep convolutional generative adversarial neural network to generate realistic images of LEGO minifigures

- `LEGO_DCGAN/minifig_images` contains images of LEGO minifigures to train the model. Performed web scraping via [Rebrickable](https://rebrickable.com/).

- `LEGO_DCGAN/minifig_output` contains generated images for each trial during training. 

- `LEGO_DCGAN/generated`: has several generated images by running `LEGO_DCGAN/generated/inference.py`. Give it a try yourself ;)

### Below are some images generated by the DCGAN generator:
1. 128x128 (took about 10-12 hours of training on a CPU)
<img src="https://github.com/Sid1279/LEGO-GAN/blob/main/LEGO_DCGAN/generated/128x128/generated_10.png" width="49%" height="auto">
<img src="https://github.com/Sid1279/LEGO-GAN/blob/main/LEGO_DCGAN/generated/128x128/generated_10.png" width="49%" height="auto">

2. 64x64 (took about 7-8 hours of training on a CPU)
<img src="https://github.com/Sid1279/LEGO-GAN/blob/main/LEGO_DCGAN/generated/64x64/generated_49.png" width="49%" height="auto">
<img src="https://github.com/Sid1279/LEGO-GAN/blob/main/LEGO_DCGAN/generated/64x64/generated_85.png" width="49%" height="auto">

Learn more about implementing your own DCGAN in PyTorch [here](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html!
