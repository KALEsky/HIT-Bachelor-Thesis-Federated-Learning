# Data directory

The training code expects to be launched from `code/ver8_shapley_fgsm_adver`
and uses `../data/` as the dataset root.

By default, `torchvision.datasets.MNIST` can download MNIST into this directory
when the runtime has network access. The public repository does not include the
dataset files or the old CIFAR-10 archive.
