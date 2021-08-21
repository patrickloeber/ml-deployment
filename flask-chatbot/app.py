from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response, bot_name

app = Flask(__name__)
CORS(app)

# https://bbbootstrap.com/snippets/bootstrap-chat-box-custom-scrollbar-template-40453784
# https://github.com/hitchcliff/front-end-chatjs

# pip install Flask torch torchvision nltk
# style.css increase width, height, font size and icon2 size


messages = []

USER = "Patrick"

@app.get('/')
def index_get():
    return render_template("base.html", messages=reversed(messages))


@app.post('/')
def index_post():
    if request.method == "POST":
        text = request.form['text']
        # TODO: error checking for text

        new_message = {"name": USER, "msg": text}
        messages.append(new_message)

        response = get_response(text)
        new_message = {"name": bot_name, "msg": response}
        messages.append(new_message)

        return index_get()


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # POST request
    if request.method == 'POST':
        # print(request.data)
        print(request.get_json())  # parse as JSON

        text = request.get_json().get("message")
        # TODO: error checking for text

        response = get_response(text)
        message = {'answer': response}
        return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)
