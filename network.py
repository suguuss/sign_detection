import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import tqdm

def get_data():
	WIDTH  = 1280
	HEIGHT = 720

	img_source	= Image.open("panneau.jpg")
	img_target 	= Image.open("panneau_target.png")

	source_rgb = img_source.convert("RGB")
	target_rgb = img_target.convert("RGB")

	inputs = []
	outputs = []

	for x in tqdm.tqdm(range(WIDTH)):
		for y in range(HEIGHT):
			r,g,b = target_rgb.getpixel((x,y))
			target = 0.01
			if r == g == b == 255:
				target = 0.99

			r,g,b = source_rgb.getpixel((x,y))

			inputs.append([r/255,g/255,b/255])
			outputs.append(target)

	return np.array(inputs, dtype=float), np.array(outputs, dtype=float)


if '__main__' == __name__:
	inputs, outputs = get_data()

	print(f"{np.count_nonzero(outputs == 0.99) / len(outputs) * 100} %")

	model = keras.Sequential(
		[
			layers.Dense(3, input_shape=[3]),
			layers.Dense(3, activation="sigmoid", name="hidden"),
			layers.Dense(1, activation="sigmoid", name="ouput")
		]
	)

	model.compile(
		optimizer=keras.optimizers.Adam(0.01),
		loss='mean_squared_error'
	)

	model.summary()

	model.fit(
		inputs,
		outputs,
		batch_size=5000,
		epochs=500
	)

	r,g,b = inputs[420000]
	print(r,g,b)
	print(model.predict([[r,g,b]]))
	print(model.predict([[0,0,0]]))

	model.save("model.h5")