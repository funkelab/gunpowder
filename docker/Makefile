TAG="funkey/gunpowder:v0.3.2"

.PHONY: default push
default:
	-docker rmi -f $(TAG)
	-rm -rf gunpowder
	cp -r ../gunpowder .
	cp ../requirements.txt .
	docker build -t $(TAG) .

push: default
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
