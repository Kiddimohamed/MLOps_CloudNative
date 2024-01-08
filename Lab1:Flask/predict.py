
import pandas as pd
import pickle

import numpy as np


model_XGBC_smoten = pickle.load(open('model/knn_over.pkl', 'rb'))

def minmax(X, Xmax, Xmin):
    X = float(X)
    return ((X - Xmin) / (Xmax - Xmin))
def pred(Datadup, enterydata):
    # Retrieve form data

    meshmax = Datadup['MeshSize'].max()
    meshmin = Datadup['MeshSize'].min()

    numberofvariablesmax = Datadup['NumberofVariables'].max()
    numberofvariablesmin = Datadup['NumberofVariables'].min()
    numberofnodesmax = Datadup['NumberofNodes'].max()
    numberofnodesmin = Datadup['NumberofNodes'].min()
    numberofcoresmax = Datadup['NumberofCores'].max()
    numberofcoresmin = Datadup['NumberofCores'].min()
    dimmax = Datadup['Dim'].max()
    dimmin = Datadup['Dim'].min()

    features = enterydata
    pdfeatures = pd.DataFrame([list(features.values())], columns=features.keys())
    print(pdfeatures)
    pdfeatures['MeshSize'] = minmax(pdfeatures['MeshSize'].iloc[0], meshmax, meshmin)
    pdfeatures['NumberofVariables'] = minmax(pdfeatures['NumberofVariables'].iloc[0], numberofvariablesmax,
                                             numberofvariablesmin)
    pdfeatures['VariableType'] = minmax(pdfeatures['VariableType'].iloc[0], 5, 0)
    pdfeatures['NumberofNodes'] = minmax(pdfeatures['NumberofNodes'].iloc[0], numberofnodesmax, numberofnodesmin)
    pdfeatures['NumberofCores'] = minmax(pdfeatures['NumberofCores'].iloc[0], numberofcoresmax, numberofcoresmin)
    pdfeatures['Dim'] = minmax(pdfeatures['Dim'].iloc[0], dimmax, dimmin)
    for i in pdfeatures.values:
        print(i)

    features = pdfeatures.values.tolist()
    features = [list(map(int, row)) for row in features]
    final_features = np.array(features).reshape(1, 7)
    # Perform classification with your machine learning model
    # Replace this with your actual model code
    prediction = model_XGBC_smoten.predict(final_features)
    # Pass the prediction to the template
    result =""
    if prediction[0] == 0:
        result = 'Blocking\_Neighbor\_alltoallv'
    elif prediction[0] == 1:
        result= 'Blocking\_alltoallv'
    elif prediction[0] == 2:
        result = 'Blocking\_p2p'
    elif prediction[0] == 3:
        result = 'Non\_Blocking\_Neighbor\_alltoallv'
    elif prediction[0] == 4:
        result = 'Non\_Blocking\_alltoallv'
    elif prediction[0] == 5:
        result = 'Non\_Blocking\_p2p'
    elif prediction[0] == 6:
        result="Persistant\_p2p"
    else:
        result = "Error in the form"
    return result