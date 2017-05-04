import gunpowder
import gunpowder.caffe

if __name__ == "__main__":
    solver_parameters = gunpowder.caffe.SolverParameters()
    train = gunpowder.caffe.Train(solver_parameters)
    print("Successfully set up gunpowder with caffe support")
