# this is the file containing examples command line args
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


# all models to train 
note the models may have been trained more the number of epochs here 

## regression 
### slp -- DONE 
python pipeline_redo/argument_parser.py --build_code --model_type slp --output_mode regression --epochs 20

python pipeline_redo/argument_parser.py --build_code --model_type slp --output_mode regression --epochs 20 --load_model

### fnn --DONE
python pipeline_redo/argument_parser.py --build_code --model_type fnn --output_mode regression --epochs 50

python pipeline_redo/argument_parser.py --build_code --model_type fnn --output_mode regression --epochs 20 --load_model

### kmeans+fnn --DONE
python pipeline_redo/argument_parser.py --build_code --model_type kmeans+fnn --output_mode regression --epochs 30

python pipeline_redo/argument_parser.py --build_code --model_type kmeans+fnn --output_mode regression --epochs 20 --load_model

### autoencoder+kmeans+fnn --DONE
python pipeline_redo/argument_parser.py --build_code --model_type autoencoder+kmeans+fnn  --output_mode regression --epochs 30
- autoencoder and fnn both trained for 30 epochs 

python pipeline_redo/argument_parser.py --build_code --model_type autoencoder+kmeans+fnn  --output_mode regression --epochs 20 --load_model



## classification 
### slp --DONE
python pipeline_redo/argument_parser.py --build_code --model_type slp --output_mode classification --epochs 20

python pipeline_redo/argument_parser.py --build_code --model_type slp --output_mode classification --epochs 50 --load_model

### fnn --DONE
python pipeline_redo/argument_parser.py --build_code --model_type fnn --output_mode classification --epochs 30

python pipeline_redo/argument_parser.py --build_code --model_type fnn --output_mode classification --epochs 20 --load_model

### kmeans+fnn --DONE
python pipeline_redo/argument_parser.py --build_code --model_type kmeans+fnn --output_mode classification --epochs 20

python pipeline_redo/argument_parser.py --build_code --model_type kmeans+fnn --output_mode classification --epochs 10 --load_model

### autoencoder+kmeans+fnn  --- ODD JUMP DOWN IN LOSSES when reinitiated
-- this is due to the autoencoder not saving properly (the autoencoder's layers are frozen and not saved in state_dict, so you can see the effect that the autoencoder has on the first leg of the losses ) 

python pipeline_redo/argument_parser.py --build_code --model_type autoencoder+kmeans+fnn  --output_mode classification --epochs 20
- autoencoder and fnn both trained for 20 epochs, then 30 more for fnn below 

python pipeline_redo/argument_parser.py --build_code --model_type autoencoder+kmeans+fnn  --output_mode classification --epochs 10 --load_model