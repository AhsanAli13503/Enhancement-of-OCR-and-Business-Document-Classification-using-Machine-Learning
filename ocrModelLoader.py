import numpy as np
import cv2
import tensorflow as tf 

# Model reconstruction from JSON file
def loadModel(json1,h5):
    with open(json1, 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(h5)
    return model

def loadAllModels():
   
    
    alphabat=[]
    numerical=[]
    abc= loadModel("Model\\numalpha.json","Model\\numalpha.h5")
    
    zero= loadModel("Model\\0Model.json","Model\\0Model.h5")
    numerical.append(zero)
    one= loadModel("Model\\1Model.json","Model\\1Model.h5")
    numerical.append(one)
    two= loadModel("Model\\2model.json","Model\\2model.h5")
    numerical.append(two)
    three=loadModel("Model\\3model.json","Model\\3model.h5")
    numerical.append(three)
    four=loadModel("Model\\4model.json","Model\\4model.h5")
    numerical.append(four)
    five=loadModel("Model\\5model.json","Model\\5model.h5")
    numerical.append(five)
    six= loadModel("Model\\6model.json","Model\\6model.h5")
    numerical.append(six)
    seven=loadModel("Model\\7model.json","Model\\7model.h5")
    numerical.append(seven)
    eight=loadModel("Model\\8model.json","Model\\8model.h5")
    numerical.append(eight)
    nine=loadModel("Model\\9model.json","Model\\9model.h5")
    numerical.append(nine)
    a=loadModel("Model\\amodel.json","Model\\amodel.h5")
    alphabat.append(a)
    b= loadModel("Model\\bmodel.json","Model\\bmodel.h5")
    alphabat.append(b)
    c=loadModel("Model\\cmodel.json","Model\\cmodel.h5")
    alphabat.append(c)
    d=loadModel("Model\\dmodel.json","Model\\dmodel.h5")
    alphabat.append(d)
    e= loadModel("Model\\emodel.json","Model\\emodel.h5")
    f= loadModel("Model\\fmodel.json","Model\\fmodel.h5")
    g= loadModel("Model\\gmodel.json","Model\\gmodel.h5")
    h= loadModel("Model\\hmodel.json","Model\\hmodel.h5")
    i=loadModel("Model\\Ymodel.json","Model\\Ymodel.h5")
    j=loadModel("Model\\jmodel.json","Model\\jmodel.h5")
    k=loadModel("Model\\kmodel.json","Model\\kmodel.h5")
    l= loadModel("Model\\lmodel.json","Model\\lmodel.h5")
    m=loadModel("Model\\mmodel.json","Model\\mmodel.h5")
    n=loadModel("Model\\nmodel.json","Model\\nmodel.h5")
    o=loadModel("Model\\omodel.json","Model\\omodel.h5")
    p= loadModel("Model\\pmodel.json","Model\\pmodel.h5")
    q= loadModel("Model\\qmodel.json","Model\\qmodel.h5")
    r= loadModel("Model\\rmodel.json","Model\\rmodel.h5")
    s=loadModel("Model\\smodel.json","Model\\smodel.h5")
    t=loadModel("Model\\tmodel.json","Model\\tmodel.h5")
    u= loadModel("Model\\umodel.json","Model\\umodel.h5")
    v=loadModel("Model\\vmodel.json","Model\\vmodel.h5")
    w=loadModel("Model\\wmodel.json","Model\\wmodel.h5")
    x=loadModel("Model\\xmodel.json","Model\\xmodel.h5")
    y=loadModel("Model\\ymodel.json","Model\\ymodel.h5")
    z=loadModel("Model\\zmodel.json","Model\\zmodel.h5")
    alphabat.append(e)
    alphabat.append(f)
    alphabat.append(g)
    alphabat.append(h)
    alphabat.append(i)
    alphabat.append(j)
    alphabat.append(k)
    alphabat.append(l)
    alphabat.append(m)
    alphabat.append(n)
    alphabat.append(o)
    alphabat.append(p)
    alphabat.append(q)
    alphabat.append(r)
    alphabat.append(s)
    alphabat.append(t)
    alphabat.append(u)
    alphabat.append(v)
    alphabat.append(w)
    alphabat.append(x)
    alphabat.append(y)
    alphabat.append(z)
    return abc,alphabat,numerical

abc,alphabat,number=loadAllModels()