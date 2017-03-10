from PIL import Image


def normalizeImage():
    image = Image.open("5.png")
    image = image.convert('LA')     # "LA" (grayscale + alpha).
    image.thumbnail((8,8),Image.ANTIALIAS)
    image.save("5out.png")


normalizeImage()