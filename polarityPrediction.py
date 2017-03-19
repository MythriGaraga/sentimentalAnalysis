import uuid
import re
import numpy as np
from flask import *
from sklearn.feature_extraction.text import CountVectorizer
from wtforms import Form, TextAreaField, validators
from sklearn.linear_model import SGDClassifier
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://localhost/Mythri'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)
Bootstrap(app)

class db_entry(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.String(36), unique=True, primary_key=True)
    review = db.Column(db.Text())
    sentiment = db.Column(db.Integer())

    def __init__(self, review, y):
        self.id = uuid.uuid4()
        self.review = review
        self.sentiment = y

db.create_all()
db.session.commit()



#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t')
        review = re.sub('<[a-zA-Z]+>', '', review)
        review = re.sub('n\'t', ' not', review)
        review = re.sub('\.+', ' ', review)
        review = re.sub('[^a-zA-Z\d\.]', ' ', review)
        review = re.sub(' +', ' ', review)
        reviews.append(review.lower())
        labels.append(int(rating))
    f.close()
    return reviews,labels

#read the reviews and their polarities from a given file
def loadTestReview(testReview):
    reviews=[]
    testReview=testReview.strip()
    testReview = re.sub('<[a-zA-Z]+>', '', testReview)
    testReview = re.sub('n\'t', ' not', testReview)
    testReview = re.sub('\.+', ' ', testReview)
    testReview = re.sub('[^a-zA-Z\d\.]', ' ', testReview)
    testReview = re.sub(' +', ' ', testReview)
    reviews.append(testReview.lower())

    return reviews

rev_train,labels_train=loadData('reviews_train.txt')

Vectorizer =  CountVectorizer(ngram_range=(1,2), max_df=0.41, min_df=0, strip_accents='unicode')
Vectorizer.fit(rev_train)
rev_train = Vectorizer.transform(rev_train)

classes = np.array([0, 1])
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
clf.partial_fit(rev_train, labels_train,classes=classes)

def classify(document):
    label  = {0:'negative', 1:'positive'}
    review = loadTestReview(document)
    X = Vectorizer.transform(review)
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document,y):
    X = Vectorizer.transform([document])
    clf.partial_fit(X,[y])


class ReviewForm(Form):
    moviereview = TextAreaField('',
    [validators.DataRequired(),
    validators.length(min=15)])


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST':
        review = request.form['moviereview']
        print review
        y, proba = classify(review)
        print y,proba
        return render_template('results.html',
                                content=review,
                                prediction=y,
                               probability=round(proba * 100, 2))
    return render_template('reviewform.html', form=form)

id = 0
@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    print y
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    paste = db_entry(review, y)
    db.session.add(paste)
    db.session.commit()

    return render_template('thanks.html')


if __name__ == '__main__':
    app.run()
