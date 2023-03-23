import sys
import os
from joblib import load

from reader import DSReader

from flask import Flask, request, render_template


sys.path.append('src')

clf = load(os.path.abspath('models/MultinomialNB_finalized_model.sav'))

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        source_email_text = request.form['email_text']
        email_text = DSReader.str_cleaning(source_email_text)
        predict = clf.predict([email_text])
        label = "SPAM" if predict == [1] else "NOT SPAM"
        return {
            "email_text": source_email_text,
            "classification_label": label
            }
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
