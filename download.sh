scp carnd@54.165.133.205:~/steeringangle8/model.h5 ~/Desktop/steeringangle8/model.h5

scp carnd@54.165.133.205:~/steeringangle8/model.json ~/Desktop/steeringangle8/model.json

scp carnd@54.165.133.205:~/steeringangle8/preprocess_dict.p ~/Desktop/steeringangle8/preprocess_dict.p

source activate carnd-term1

python drive.py model.json