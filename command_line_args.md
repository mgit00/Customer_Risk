# this is the file containing example command line args
this will be used to show the flexibility in the way that argparse was implemented 

python pipeline/argument_parser.py --mode full --epochs 20 
- run full pipeline from data processing to model building 

python pipeline/argument_parser.py --mode build_model --epochs 20 --loaded_model
- building model, running fnn for 20 epochs, loading in existing model 

python pipeline/argument_parser.py --mode build_model --epochs 5 --loaded_model