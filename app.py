from flask import Flask, render_template

app = Flask(__name__, static_url_path='')


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():



if __name__ == '__main__':
    app.run(port=3000, debug=True)
