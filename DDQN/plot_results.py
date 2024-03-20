import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class Plot:
    def __init__(self, path: str):
        self.path = path
        self.file = os.path.join(path, "logs.csv")
        self.plt = plt
        # self.plt.grid()
        self.df = pd.read_csv(self.file)

    def plot(self):
        # plotting rewards
        reward_fig, reward_ax = self.plt.subplots()
        reward_ax.plot(self.df["episode"], self.df["rewards"])
        reward_ax.set_title("Rewards")
        reward_ax.set_xlabel("Episode no.")
        reward_ax.set_ylabel("Reward")
        # reward_ax.legend(["reward"], loc="upper right")
        reward_ax.grid()

        # plotting avg and max waiting times
        wait_fig, wait_ax = self.plt.subplots()
        wait_ax.plot(self.df["episode"], self.df["avg_acc_waiting_time"], ls="-", color="blue")
        # wait_ax.plot(self.df["episode"], self.df["max_acc_waiting_time"], ls="--", color="red")
        wait_ax.set_title("Avg. Waiting Times")
        wait_ax.set_xlabel("Episode no.")
        wait_ax.set_ylabel("Avg. waiting time")
        # wait_ax.legend(["avg. waiting time", "max. waiting time"], loc="upper right")
        wait_ax.grid()

        # plot max waiting times
        wait2_fig, wait2_ax = self.plt.subplots()
        # wait_ax.plot(self.df["episode"], self.df["avg_acc_waiting_time"], ls="-", color="blue")
        wait2_ax.plot(self.df["episode"], self.df["max_acc_waiting_time"], ls="-", color="red")
        wait2_ax.set_title("Max. Waiting Times")
        wait2_ax.set_xlabel("Episode no.")
        wait2_ax.set_ylabel("Max.waiting time")
        # wait_ax.legend(["avg. waiting time", "max. waiting time"], loc="upper right")
        wait2_ax.grid()

        # save to a csv
        pp = PdfPages(os.path.join(self.path, "graphs.pdf"))
        pp.savefig(reward_fig)
        pp.savefig(wait_fig)
        pp.savefig(wait2_fig)
        pp.close()


if __name__ == "__main__":
    plot = Plot("./DDQN/runs/3x3/run_6")
    plot.plot()

        
