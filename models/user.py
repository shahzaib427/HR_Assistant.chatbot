from .db import db

class User(db.Model):
    __tablename__ = 'users'   # ⚠️ MUST match 'users.id'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)