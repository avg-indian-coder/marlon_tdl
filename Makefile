DOCKER_USERNAME ?= docker_anonimoose
APPLICATION_NAME ?= marlon_brando
LINUX_USERNAME ?= root
DOCKER_DISPLAY = ${DISPLAY}  





build:
	docker build --tag ${DOCKER_USERNAME}/${APPLICATION_NAME} .

run:
	docker run -it \
	-e DISPLAY=${DOCKER_DISPLAY} \
	-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
	--user=${LINUX_USERNAME} \
    ${DOCKER_USERNAME}/${APPLICATION_NAME} \

# run:
# 	docker run -it --rm \
# 	--env="DISPLAY" \
#     --volume="/etc/group:/etc/group:ro" \
#     --volume="/etc/passwd:/etc/passwd:ro" \
#     --volume="/etc/shadow:/etc/shadow:ro" \
#     --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     --user=${LINUX_USERNAME} \
#     ${DOCKER_USERNAME}/${APPLICATION_NAME} \

check:
	docker run -it \
	--user=${LINUX_USERNAME} \
    ${DOCKER_USERNAME}/${APPLICATION_NAME} \
    bash