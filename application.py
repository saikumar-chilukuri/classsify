from flask import Flask
from flask import request
from fastai.vision import *
import urllib.request

application = Flask(__name__)


@application.route('/')
def detect_document():
    img = open_image(urllib.request.urlopen(request.args.get('imageUrl')))
    path = Path('./')
    learn = load_learner(path)
    pred_class, pred_idx, outputs = learn.predict(img)
    return str(pred_class)


if __name__ == '__main__':
    application.run()
