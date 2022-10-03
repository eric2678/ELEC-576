# import the modules
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

type = "relu"
# get the path/directory
folder_dir = os.getcwd() + "\\figure\\" + type
print(folder_dir)
count = 1
for images in os.listdir(folder_dir):
    # check if the image ends with png
    if images.endswith(".png"):
        plt.subplot(4, 7, count)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=False,
            labelleft=False,
            labelbottom=False)  # labels along the bottom edge are off
        img = mpimg.imread(folder_dir + "\\" + images)
        plt.imshow(img)
        count += 1
plt.suptitle(type)
plt.show()
