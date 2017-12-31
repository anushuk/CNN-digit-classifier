from flask import Flask
import os

from digit_recognizer import digit

app=Flask(__name__)


@app.route('/predicthdr/',methods=['GET','POST'])
def predicthdr():
    try:

        imgData = request.get_data()
        out = digit(imgData)
        li=str(out)
        return li
    except Exception as e:
        return (str(e))


if __name__ == '__main__':
    app.run(debug=True)
