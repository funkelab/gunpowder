import luigi
from producer_pool import run

class TestTask(luigi.Task):

    n = luigi.IntParameter()

    def run(self):
        run()

if __name__ == "__main__":

    luigi.build(
            [TestTask(i) for i in range(2)],
            workers=10
    )
