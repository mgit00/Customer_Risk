# python base image 

FROM python:3.10 

# have to change working directory, not a flat layout 
WORKDIR /code 
# image directory to be creates 

# add requirements 
    # note .yml is best used by conda, switch to pip with .txt file 
COPY ./app/requirements_pip.txt /code/requirements.txt
# copy from current directory to within the image 

# do installatiion of libraries 
    # --no-cache-dir since no need to cache inside 
    # pip install will be run once in the container, so keeping cache is taking uncessary space 
    # prevents bloated image 
RUN pip install --no-cache-dir -r /code/requirements.txt 

# have to add python path, since commands should be run base directory of code 
    # note file structure creaed by below command: code/api_framework/main.py
ENV PYTHONPATH "${PYTHONPATH}:/code/"


# add python code to image 
COPY ./app/ /code/
# COPY app/ . #  . means /code -- or 

EXPOSE 8000 

CMD ["python", "-m", "api_framework.main"] # FIXME call from src.main now 