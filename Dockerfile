FROM ubuntu:latest

ENV SUMO_USER anonimoose

# RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common
# RUN add-apt-repository ppa:sumo/stable && apt update && apt install -y sumo sumo-tools sumo-doc

RUN apt update && apt install -y sumo sumo-tools sumo-doc python3 python3-dev python3-pip

ENV SUMO_HOME usr/share/sumo

# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


RUN pip install -r $SUMO_HOME/tools/requirements.txt


COPY marlon_tdl ./home/$SUMO_USER/projects/marlon_tdl
RUN pip install -r ./home/$SUMO_USER/projects/marlon_tdl/requirements.txt

RUN adduser $SUMO_USER --disabled-password

WORKDIR /home/$SUMO_USER/projects/marlon_tdl
# CMD ["sumo-gui","-c","./nets/bo/run.sumocfg"]
CMD ["python3","./do_whatever_testing_here.py"]
