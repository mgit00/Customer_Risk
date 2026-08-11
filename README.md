# Bank Credit Risk 
This project uses credit card application and bank record data from https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction. 

with this data, i will create an end-to-end pipeline which classifies the risk level of approving a credit card application for a customer. 

This can be integrated into a much larger application which can ultimatly decide whether or not to apporve a credit card application. This larger system will take into account the national economy, local neightborhood of the applicant, and the company's overall need to expand into a larger customer base (which requires more approvals). 

I previously made an entire working pipleine, but realized that it could have been made more simple by creating modularity within the files. 
I faced trouble with adding new models and changing data processing methods, so creating an entirely new pipeline seemed best. I pushed the most recent branch from the old pipeline repo and began working again. So this pipeline is the improved version I made. 


# FastAPI Usage 
fastapi will be used to interact with the pipeline. this is best done with a BaseModel class- which replaces the commandline argparse. as a result, there will be 2 ways to run the pipeline: pipeline/argument_parser or pipeline/basemodel_api. both will be called by python commands. 