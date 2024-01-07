from joblib import load
from numpy import array



#Test url : http://127.0.0.1:8080/project1/predict?model=KNN&preg=12&plas=0.1&pres=0.1&skin=12&test=0.1&mass=0.1&pedi=1&age=22

def predModel(data):
    user_input = array([[
        float(data['preg']),
        float(data['plas']),
        float(data['pres']),
        float(data['skin']),
        float(data['test']),
        float(data['mass']),
        float(data['pedi']),
        float(data['age'])
    ]])
    model = data['model']
    if model == "KNN":
        filename = '../models/knn_model.joblib'
    elif model == "SVM":
        filename = '../models/svm_model.joblib'
    elif model == "DT":
        filename = '../models/DTmodel.joblib'
    else:
        return {"Failed":"Model Not Found !!"}

    loaded_model = load(open(filename, 'rb'))
    res = loaded_model.predict(user_input)
    return {'result' : int(res[0]), 'status':200}