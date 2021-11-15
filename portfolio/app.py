from flask import Flask,render_template
from flask import request


## create instance
app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("home.html")

@app.route("/<string:slug>")
def blog_details(slug):
    return render_template("home.html", slug=slug)

@app.route("/search")
def search():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)