import matplotlib.pyplot as plt

class Visualize:
    def PlotLoss(iteration, loss):
        plt.title('Loss against Training Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.xticks(iteration)
        plt.plot(iteration, loss, lw=2.5)
        plt.show()

Visualize.PlotLoss([1, 2, 3, 4], [0.7, 0.4, 0.2, 0.01])