from flask import Flask, render_template
import tensorflow
import keras
import cv2
app = Flask(__name__)


@app.route("/")
def hello_world():
    model = keras.models.load_model("hcr_related\hcr3.h5", compile=False)
    img = cv2.imread("hcr_related\s.png", cv2.IMREAD_GRAYSCALE)
    img = img.reshape(784)
    prediction = model.predict(img)
    print(prediction)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
