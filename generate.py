import tensorflow as tf
import numpy as np
from PIL import Image
import tqdm

# SIZE FOR CROPPED INPUT
# WIDTH  = 400
# HEIGHT = 300


WIDTH  = 1280
HEIGHT = 720

input_img = Image.open("panneau.jpg")
input_rgb = input_img.convert("RGB")

model = tf.keras.models.load_model("model.h5")

img = Image.new("RGB", (WIDTH, HEIGHT))


inputs = []

for x in tqdm.tqdm(range(WIDTH)):
	for y in range(HEIGHT):
		r,g,b = np.array(input_rgb.getpixel((x,y)), dtype=float) / 255
		
		inputs.append([r,g,b])

		# if model.predict([[r,g,b]])[0][0] > 0.5:
		# 	out = (255,255,255)
		# else:
		# 	out = (0,0,0)
		
		# img.putpixel((x,y), out)

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


img.save("output_full.png")