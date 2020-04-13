# ITk Strips Powerboard S/N Identification From Images

## Dependencies

All code is written using Python 3 and depends on the following packages:
- [Pillow](https://pillow.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/)

All dependencies can be installed using `pip`:
```shell
pip install pillow matplotlib tensorflow
```

## Example
The following example generates a series of training and test images containing labels that are etched on a Powerboard shield-box. They contain the LBL logo and a 7-digit decimal number. In the generate images, the serial number is generated randomly. The output is saved as `outDir/SERIALN.png`, where `SERIALN` is replaced by the S/N contained inside the image. The image file name is also used as the truth label for the neural network.

```shell
# Generate images for training
mkdir trainimages
./genlabel.py trainimages
# Generate images for testing
mkdir testimages
./genlabel.py testimages
# Train and test the model
./testmodel.py trainimages testimages
```