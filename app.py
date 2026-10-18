import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)
model = keras.models.load_model("mango_model.keras")

class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("leaf")
        if file and file.filename:
            img = Image.open(file.stream).convert("RGB")

            # make a small copy to show on the page
            buf = io.BytesIO()
            img.resize((300, 300)).save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            # predict
            arr = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
            probs = tf.nn.softmax(model.predict(arr, verbose=0)[0]).numpy()
            top = np.argsort(probs)[::-1][:3]

            result = {
                "image": img_b64,
                "label": class_names[top[0]],
                "confidence": round(float(probs[top[0]]) * 100, 1),
                "unsure": bool(probs[top[0]] < 0.6),
                "others": [(class_names[i], round(float(probs[i]) * 100, 1)) for i in top],
            }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)