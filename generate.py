import tensorflow as tf
import numpy as np
from PIL import Image
import tqdm

# SIZE FOR CROPPED INPUT
# WIDTH  = 400
# HEIGHT = 300


# WIDTH  = 980
# HEIGHT = 551

FILENAME = "test_image.png"

input_img = Image.open(f"dataset/{FILENAME}")
input_rgb = input_img.convert("RGB")

WIDTH, HEIGHT = input_img.size

model = tf.keras.models.load_model("new_model.h5")

img = Image.new("RGB", (WIDTH, HEIGHT))

inputs = []

for x in tqdm.tqdm(range(WIDTH)):
	for y in range(HEIGHT):
		r,g,b = np.array(input_rgb.getpixel((x,y)), dtype=float) / 255
		inputs.append([r,g,b])

outputs = model.predict(inputs)

print(f"len is {len(outputs)}")
print(f"len should be {HEIGHT*WIDTH}")

for x in tqdm.tqdm(range(WIDTH)):
	for y in range(HEIGHT):

		if outputs[x*HEIGHT+y] > 0.5:
			out = (255,255,255)
		else:
			out = (0,0,0)

		img.putpixel((x,y), out)


img.save(f"out_{FILENAME}")