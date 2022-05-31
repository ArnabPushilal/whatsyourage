
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path


db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
 app = Flask(__name__)
 app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
 app.config['SECRET_KEY']='rannasds'
 app.config['UPLOAD_FOLDER']='.\static\images'
 app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
 app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
 db.init_app(app)
 from database import Images
 from views import views
 app.register_blueprint(views,url_prefix='/')
 create_database(app)

 return app
 
def create_database(app):
    if not path.exists(DB_NAME):
        db.create_all(app=app)
        print('Created Database')


