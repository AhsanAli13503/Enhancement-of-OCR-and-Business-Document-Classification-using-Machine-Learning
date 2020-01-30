dataset =[]

def conv_list_toText(data):
    text=""
    for a in data:
        text = text + " " + a
    return text

def getFeatureText(text):
    abc=""
    with open('newdataset.txt') as fp:
        abc = fp.readlines()

    for i in range(len(abc)):
        words=abc[i].split()
        for word in words:
            dataset.append(word.lower())
    findings = text.split()
    featuretext = ""
    for wro in findings:
        wro = wro.lower()
        if wro in dataset:
            featuretext = featuretext+ " " + wro

    return featuretext
