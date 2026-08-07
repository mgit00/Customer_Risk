# this is the file containing example command line args
this will be used to show the flexibility in the way that argparse was implemented 

python pipeline_redo/argument_parser.py --run_pipeline --model_type slp --output_mode classification --load_model --epochs 10 
- this runs the entire pipeline for the slp model, with classification as the target feature 
- the model that was previously stored will be trained as well 

python pipeline_redo/argument_parser.py --run_pipeline --model_type autoencoder+kmeans+fnn --output_mode regression 
- entire pipeline for autoencoder+kmeans+fnn with regression mode 

python pipeline_redo/argument_parser.py --build_code --model_type autoencoder+kmeans+fnn --output_mode regression 
- just builds the model, not the kmeans exploration or data processing 

python pipeline_redo/argument_parser.py --build_code --model_type autoencoder+kmeans+fnn --output_mode regression --load_model --epochs 10 
- loads in the saved model from before 