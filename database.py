from app import db
from PIL import Image

class Images(db.Model):

    """
    Class for Images of table of
    first_name,true_age,filename,data
    created in the form + predicted age
    from the model
    
    """

    __tablename__ = 'images'
    

    id = db.Column(db.Integer, primary_key=True)
    #email = db.Column(db.String(150),unique=True)
    first_name=db.Column(db.String(150))
    true_age=db.Column(db.Float)
    predicted_age=db.Column(db.Float)
    img_filename = db.Column(db.String())
    img_data = db.Column(db.LargeBinary)

    def __repr__(self):
        return '<image id={},name={}>'.format(self.id, self.name)

def get_image(the_id):

    return Image.query.get_or_404(the_id)

def add_image(image_dict):
    new_image = Images(first_name=image_dict['name'], 
                        img_filename=image_dict['img_filename'], 
                        img_data=image_dict['img_data'],
                        true_age=image_dict['true_age'],
                        predicted_age=image_dict['pred_age'])

    db.session.add(new_image)
    db.session.commit()
