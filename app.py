import os
import commons.JsonEncoder as JsonEncoder
from flask import Flask, render_template, request, redirect
from inference import get_prediction
from commons import format_class_name

# instantiate Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        prob_dict = get_prediction(image_bytes=img_bytes)
        return render_template('result.html', prob_dict, cls = JsonEncoder)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))