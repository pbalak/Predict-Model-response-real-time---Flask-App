## Model was trained and saved to disk as part of a save_Model .py class (pima Diabetes)
#SO, as part of this class, we are loading the model and testing one record by passing it as
#input from postman as JSON


### Note that model prodict was not working when calling the inputs from postman. Hence had to change the backend from
# tensor flow to theano in \.keras\keras.jon file in your local.


## Postman name - localhost_pycharm under Machinelearning

#from flask import Flask, request
from keras.models import model_from_json
import pandas as pd
#app = Flask(__name__)

import flask
from flask import request

app = flask.Flask(__name__)


@app.route('/')
## This will run in the following path -  http://127.0.0.1:5000/
def index():
    return 'The method is %s' %request.method


@app.route('/checkmethod', methods=['GET','POST'])

def check():
    if request.method == 'POST':
        return 'You are using POST'
    else:
        return 'You are probably using GET'


@app.route('/predicttest', methods=['POST'])
def predicttest():
    # grabs the data tagged as 'name'
    name = request.get_json()['name']

    # sending a hello back to the requester
    return "Hello " + name

json_file = open('C:/Users/Sheetal/Desktop/Big Data and Data Analytics/Python_Analytics/deploying_pyhton_pickle_and_flask/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/Sheetal/Desktop/Big Data and Data Analytics/Python_Analytics/deploying_pyhton_pickle_and_flask/model.h5")
print("Loaded model from disk")


@app.route('/predict', methods=['POST'])
def predict():
    # grabs the data tagged as 'name'
    A = request.get_json()['A']
    B = request.get_json()['B']
    C = request.get_json()['C']
    D = request.get_json()['D']
    E = request.get_json()['E']
    F = request.get_json()['F']
    G = request.get_json()['G']
    H = request.get_json()['H']

	#Sample Input -
	#{
	#"A":"1",
	#"B":"85",
	#"C":"66",
	#"D":"29",
	#"E":"0",
	#"F":"26.6",
	#"G":"0.351",
	#"H":"31"
	#}
	
    print("Printing the Inputs")
    df = pd.DataFrame([[A, B, C, D, E, F, G, H]], columns=list('ABCDEFGH'))
    array = df.values
    print(A)
    print(B)

    response = {}
    response['predictions'] = loaded_model.predict([array]).tolist()



    #prediction = loaded_model.predict(array)
    #print(prediction)
    # returning the response object as json

    print (response)
    #return prediction

    return flask.jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
