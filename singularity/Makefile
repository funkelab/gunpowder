TAG="funkey/gunpowder:v0.3"

.PHONY: default push
default:
	-rm gunpowder.simg
	-rm -rf gunpowder
	cp -r ../gunpowder .
	cp ../requirements.txt .
	sudo singularity build gunpowder.simg Singularity

push: default
	#docker push $(TAG)

test: default
	#docker run --rm $(TAG)
