from skimage import io
import imageio
print(imageio.__version__)
import skimage
print(skimage.__version__)
method = "imageio"
method = "skimage"
file="/Users/pm286/workspace/pyamiimage/test/resources/biosynth1_cropped/arrows_removed.png" 
# file="/Users/pm286/pmr/pmr.jpg"
if method == "skimage":
	img = imageio.imread(file, as_gray=False)
else:
	img = io.imread(file)
print(f"{method} {file} {img}")

