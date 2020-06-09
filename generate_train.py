import os

image_files = []
path='/content/drive/My Drive/pneumonia/train_images/'
os.chdir(path)
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append(path + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")