import os
import csv
import json
import src.Part_1 as Part_1

from flask import Blueprint, request, render_template


user_bp = Blueprint('movie', __name__)



@user_bp.route('/movie', methods=['GET'])
def get_movie():
    
    movie_title = request.args.get('movie_title')

    Part_1.make_reviews(movie_title)

    score=list()
    info=dict()
    score=Part_1.get_score(movie_title)
    info=Part_1.get_info(movie_title)


    if movie_title == None:
      return "No movie title given", 400
    elif score != None:
      return render_template('index.html', score=score, info=info)
    else:
      return f"moive '{movie_title}' doesn't exist", 404


