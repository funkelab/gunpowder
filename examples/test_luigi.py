import luigi
from cremi import train

class TrainTask(luigi.Task):

    n = luigi.IntParameter()

    def run(self):
        train()

if __name__ == "__main__":

    luigi.build(
            [TrainTask(i) for i in range(2)],
            workers=10
    )
