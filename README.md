# FaceRecognizer
1) Initialize the environment cv using the necessary commands, or create your own virtual env using venv command
		Make sure the following dependencies are installed in the virtual environment if u have created one
			i) OpenCV	
			ii)OpenCV-contrib
			iii)SQLite
			iv)numpy
			v) pillow
  
2)Run new_database.py to initalize the database
3)Check if DataBase.db is created
4)Next run face_rec.py and enter the name of the face you want to be recognized next show the face to the camera wait for the app to scan it once its done, it will auto-close
5)Now execute trainer.py and show the face to camera once again
6)Finally run detection_fin.py and it opens a "face-recognizer"  Make sure the face is in the same position when it was recorded first, if everything went well your face
will be detected and the name will show up.
