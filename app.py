import os
from flask import Flask, render_template, request, redirect
from inference import get_prediction

# instantiate Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        predicted_class = get_prediction(image_bytes=img_bytes)
        return render_template('result.html', predicted_class = predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))