
from app import create_app
import os
#from waitress import serve
from flask import Flask

app=create_app()


if __name__=="__main__":
    port=os.environ.get("PORT",5000)


    app.run(debug=True,host="0.0.0.0",port=port)
   
    
   
   