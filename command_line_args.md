# this is the file containing example command line args
this will be used to show the flexibility in the way that argparse was implemented 

python pipeline/argument_parser.py --mode full --epochs 20 
- run full pipeline from data processing to model building 

python pipeline/argument_parser.py --mode build_model --epochs 20 --loaded_model
- building model, running fnn for 20 epochs, loading in existing model 

python pipeline/argument_parser.py --mode build_model --epochs 5 --target_mode classification
- create classification model 

python pipeline/argument_parser.py --mode full --epochs 5 --target_mode regression 
- create regression model after doing pipeline to create regression dataset 

the regression and classificatoin pipelines have been changed to allow for more flexibility. merge the branch back and continue working

