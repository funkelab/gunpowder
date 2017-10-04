TAG="funkey/gunpowder:v0.3-pre5"

.PHONY: default push
default:
	-docker rmi -f funkey/caffe:latest
	docker build --pull -t funkey/caffe:latest ./caffe
	-docker rmi -f $(TAG)
	-rm -rf caffe/gunpowder/gunpowder
	cp -r ../gunpowder caffe/gunpowder
	cp ../requirements.txt caffe/gunpowder
	docker build -t $(TAG) ./caffe/gunpowder

push: default
	docker push funkey/caffe:latest
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
