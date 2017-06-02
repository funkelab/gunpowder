.PHONY: default push
default:
	docker build --pull -t caffe ./caffe
	-docker rmi -f funkey/caffe:latest
	docker tag `docker build -t caffe ./caffe | grep 'Successfully built' | sed 's/Successfully built //'` funkey/caffe:latest
	docker build --no-cache -t gunpowder ./caffe/gunpowder
	-docker rmi -f funkey/gunpowder:latest
	docker tag `docker build -t gunpowder ./caffe/gunpowder | grep 'Successfully built' | sed 's/Successfully built //'` funkey/gunpowder:v0.2

push: default
	docker push funkey/caffe:latest
	docker push funkey/gunpowder:v0.2

test:
	docker run --rm gunpowder:v0.2
