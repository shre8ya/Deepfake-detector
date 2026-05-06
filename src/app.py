from flask import Flask, render_template, request
import os
from detector import analyze_video

app = Flask(__name__, template_folder="../templates")

UPLOAD_FOLDER = "../uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    output = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            output = analyze_video(filepath)

    return render_template("index.html", result=output)


if __name__ == "__main__":
    app.run(debug=True)