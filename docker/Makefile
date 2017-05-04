.PHONY: default push
default:
	sudo nvidia-docker build -t gunpowder .
	-sudo nvidia-docker rmi funkey/gunpowder:latest
	sudo nvidia-docker tag `sudo nvidia-docker build -t gunpowder . | grep 'Successfully built' | sed 's/Successfully built //'` funkey/gunpowder:latest

push: default
	sudo nvidia-docker push funkey/gunpowder:latest
