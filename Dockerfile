FROM python:3.8

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /source

RUN mkdir data
RUN mkdir models
RUN mkdir fe_data

COPY requirements.txt /source/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


RUN pip install jupyter
# Copy the current directory contents into the container at /source
COPY . /source/

# Expose the port Jupyter runs on
EXPOSE 8888

# Command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
