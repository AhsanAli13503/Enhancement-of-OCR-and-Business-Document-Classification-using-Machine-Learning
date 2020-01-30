
from flask import Flask,render_template, request,jsonify,send_file,make_response
from werkzeug.utils import secure_filename
app = Flask(__name__)
import cv2
from Main import main_fun
from classificationmodel import load_classfication_pred_res
from creatingword import conv_list_toText,getFeatureText
import barcodescanner
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/return-barfiles/')
def return_files_tut():
	try:
		return send_file('/Enhancement Of OCR and Document classification/static/abc.jpg', attachment_filename='abc.jpg')
	except Exception as e:
		return str(e)
@app.route('/return-files/')
def return_files_tut1():
	try:
		return send_file('/Enhancement Of OCR and Document classification/static/document.docx', attachment_filename='document.docx')
	except Exception as e:
		return str(e)

@app.route('/uploadajax', methods=['POST'])
def process():
   f = request.files['file']
   f.save(secure_filename(f.filename))
   files = f.filename
   img = cv2.imread(files,0)
   data ,typeOfimg,tableimg= main_fun(img)
   text = conv_list_toText(data)
   featuretext= getFeatureText(text)
   prediction=load_classfication_pred_res(featuretext)
   return (prediction)   

@app.route('/textMining', methods=['POST'])
def process1():
   text= request.data
   text=str(text)
   
   featuretext= getFeatureText(text)
   print(featuretext)
   prediction=load_classfication_pred_res(featuretext)
   print(prediction)
   return (prediction)   
@app.route('/barCode', methods=['POST'])
def process2():
   f = request.files['file']
   f.save(secure_filename(f.filename))
   files = f.filename
   img = cv2.imread(files)
   data=barcodescanner.barcodeScanner(img)
   
   if len(data)>0:
        print(data)
        return data
   else:
        print(data) 
        return "No Bar Code Detected"
   
    
if __name__ == "__main__":
    app.run(debug=True)