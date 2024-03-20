##for WGAN
ml_hw6_WGAN.py


##for WGAN_GP
ml_hw6_WGAN_GP.py


##for stylegan2
# Install
pip install stylegan2_pytorch

# train
stylegan2_pytorch --data ./faces --name styleGAN --image-size 64 --num-train-steps 50000

# save images
stylegan2_pytorch ¡Vgenerate --name styleGAN --image-size 64 --num-generate 1000 --num-image-tiles 1

(use stylegan2_output.py can help to generate the document of the images we want)