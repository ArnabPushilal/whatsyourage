
import json
from flask import Blueprint, render_template, request, flash,redirect,url_for, request,session
from PIL import Image
from model import VGGnet
from flask import current_app
from inference import preprocess_image,test_model
import torch
import os
from facedetect import face
from werkzeug.utils import secure_filename
from app import db
from database import Images,add_image

views=Blueprint('views',__name__)

IMAGE_DIR = 'temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

#Handle File too large error by redirecting
@views.errorhandler(413)
def error413(e):
    flash('File too large')
    return redirect(url_for('views.home'))

#Allowed filenames
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@views.route('/', methods=['GET', 'POST'])
def home():

    """
    Function for home page. Checks for 
    empty filename, age range. Additionally,
    does the model predcitionss.
    
    """

    if request.method == 'POST':
        first_name = request.form.get('firstName')
        true_age = request.form.get('age')
        #Avoids error when form is empty
        try: 
         true_age=int(true_age) 
        except:
         flash('Invalid Age')
         redirect(request.url)


        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('You should upload an image!')
            return redirect(request.url)
        if allowed_file(file.filename) == False:

            flash('Please only upload jpg/pngs!')
            return redirect(request.url)

        
        if file and allowed_file(file.filename):

         #Convert image to tensor
        
         im_cv=face(file.stream)
       
         #No faces detected
         if im_cv is None:
             flash('No faces detected in the image')
             return redirect(request.url)

         im=preprocess_image(im_cv)
        
         if im.shape[1]!=3:
             
             flash('The image must have 3 channels only!')
             return redirect(request.url)
        
         elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
         if true_age <= 0 or true_age>= 120 or type(true_age)!= int :
          
            flash('Age must be within 0 and 120 and must be an integer', category='error')
            return redirect(request.url)
         
         #Model prediction
         model=VGGnet()  
         model.load_state_dict(torch.load('new_model.pt',map_location=torch.device('cpu')))
         pred_age=test_model(im,model)

         
         filename = secure_filename(file.filename)

         #Convert to bytes for db storage
         numpy_image=im.numpy()
         ims=numpy_image.tobytes()

         #Save Img in temp
         imsave = Image.open(file.stream)
         imsave.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))

         #Dump as messages to use in the predict page
         messages = json.dumps({"True_Age":true_age,"Pred_Age":pred_age,\
             "filename": filename})
         session['messages'] = messages

         #Permission given to store data to db
         if request.form.get('filterName') == 'on':
            add_image({'name':first_name, 
                            'img_filename':filename, \
                            'img_data': ims ,
                            'true_age':true_age,
                            'pred_age':pred_age})
            flash('Data for "{}" saved!'.format(first_name))
        
            
         return redirect(url_for('views.predict',messages=messages))
         
       
    return render_template("form.html")

@views.route("/example", methods = ["GET"])
def ex():
    """
    Renders view for example images
    
    """

    return render_template("example.html")

@views.route("/predict", methods = ["GET"])
def predict():

    """

    Renders view for predictions along with
    given image
    
    """
    
    messages = request.args['messages']
    messages = session['messages'] 

    messages = json.loads(messages)

    true_age=messages['True_Age']
    pred_age=messages['Pred_Age']
    filen=messages['filename']



    return render_template("predict.html",true_age=true_age\
        ,pred_age=pred_age,filen=filen)


