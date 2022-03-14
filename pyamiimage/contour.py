from skimage.measure import find_contours
from skimage import io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

image = io.imread('contour_finding_test.png')
# image = io.imread('FlowchartDiagram.png')
image = rgb2gray(image)
out = find_contours(image)
print(len(out))

# Find contours at a constant value of 0.8
# contours = find_contours(image, 0.8)
contours = find_contours(image, )

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
# io.imshow(image)
# io.show()
