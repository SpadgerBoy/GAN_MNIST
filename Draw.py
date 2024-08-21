import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Draw:
    def plot(samples):
        fig = plt.figure(figsize=(3, 3))
        gs = gridspec.GridSpec(3, 3)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        return fig


