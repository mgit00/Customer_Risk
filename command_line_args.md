# this is the file containing example command line args
this will be used to show the flexibility in the way that argparse was implemented 

python pipeline/argument_parser.py --mode full --epochs 20 
- run full pipeline from data processing to model building 

python pipeline/argument_parser.py --mode build_model --epochs 20 --loaded_model
- building model, running fnn for 20 epochs, loading in existing model 

python pipeline/argument_parser.py --mode full --epochs 5 --target_mode classification
- create classification model 

python pipeline/argument_parser.py --mode full --epochs 5 --target_mode regression 
- create regression model after doing pipeline to create regression dataset 

python pipeline/argument_parser.py --mode build_model --epochs 10 --saved_model model_info/saved_model_kmeans_fnn.pt 
- create classification model for kmeans_fnn 
    python pipeline/argument_parser.py --mode build_model --epochs 10 --saved_model model_info/saved_model_kmeans_fnn.pt --loaded model model_info/saved_model_kmeans_fnn.pt 
    - load in model from above  

- save the kmeans model along with the model_state 
- if saving, just need centroid distances, then argmax euclidean 
- this can be done with just saving 
- building 
    - join in kmeans fit predict 
    - save centroid locations 
    - add distances to centroids to data 
    - pass through the nn model 
- loading in 
    - add distances to centroids to data 
    - pass through the nn model 

figure out how to integrate in the kmeans model, in model_helper??