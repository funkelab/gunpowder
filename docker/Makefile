.PHONY: default push
default:
	docker build -t gunpowder .
	-docker rmi -f funkey/gunpowder:latest
	docker tag `docker build -t gunpowder . | grep 'Successfully built' | sed 's/Successfully built //'` funkey/gunpowder:latest

push: default
	docker push funkey/gunpowder:latest
