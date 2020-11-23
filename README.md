# deepface-mail-security
FB's DeepFace model and OpenCV module help realtime face recognition and also sends an email if the face is not a recognized image

## Steps
1. Create a directory named 'pics' and save all the users, you want the model to train.
2. I used python 3.5 , so if you want to create an environment with python 3.5, (assuming you have anaconda and path set),
    > conda create -n envname python=3.5
3. Activate the environment with
    > activate envname
4. Install all the dependencies from requiremnets.txt
    > pip install -r requirements.txt
5. Enter sender's mail-id and password and also reciever's mail-id in the "deepface-mail.py".
6. Download "VGGFace2_DeepFace_weights_val-0.9034.h5" from google and place it in your directory.
7. Run deepface-mail.py
    > python deepface.py
________________________________________________________________________________________________________________________________________________________________________________

## Feel free to edit errors and help better perform the project for further versions of python, tensorflow, keras
