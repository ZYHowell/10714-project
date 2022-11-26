import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        # img_new[i][j][k] = img[i][-j][k]
        if flip_img:
            return np.flip(img, axis=1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        out = np.zeros_like(img)
        height, width, c = img.shape
        h = height - abs(shift_x)
        w = width - abs(shift_y)
        if h > 0 and w > 0:
            out_x_start = max(-shift_x, 0)
            out_y_start = max(-shift_y, 0)
            src_x_start = max(shift_x, 0)
            src_y_start = max(shift_y, 0)
            out[out_x_start:h + out_x_start, out_y_start:w +
                out_y_start, ] = img[src_x_start:h + src_x_start,
                                     src_y_start:w + src_y_start, ]
        return out
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )
        self.cur_idx = 0

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            dataset_l = len(self.dataset)
            shuffled = np.arange(dataset_l)
            np.random.shuffle(shuffled)
            self.ordering = np.array_split(
                shuffled, range(self.batch_size, dataset_l, self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.cur_idx == len(self.ordering):
            raise StopIteration
        outs = []
        for idx in self.ordering[self.cur_idx]:
            outs.append(self.dataset[idx])
        self.cur_idx += 1

        unzip_outs = [[] for _ in range(len(outs[0]))]
        for out in outs:
            for o, l in zip(out, unzip_outs):
                l.append(np.reshape(o, (1, ) + o.shape))
        final_outs = tuple(
            Tensor(np.concatenate(outs))
            for outs in unzip_outs)
        return final_outs
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        def test_magic_num(num, expected):
            assert num == expected, f"magic number err: expected {num}, got {magic}"

        # Read image raw data
        with gzip.open(image_filename, "rb") as file:
            magic, n, nrow, ncol = struct.unpack(">IIII", file.read(16))
            test_magic_num(magic, 2051)
            tot_bytes = n * nrow * ncol
            imgs = np.frombuffer(file.read(tot_bytes),
                                 dtype=np.uint8).reshape(n, nrow * ncol)
        # Read label raw data
        with gzip.open(label_filename, "rb") as file:
            magic, n = struct.unpack(">II", file.read(8))
            test_magic_num(magic, 2049)
            lbls = np.frombuffer(file.read(n), dtype=np.uint8)
        # Min-Max Normalization for the data
        imgs = imgs.astype(np.float32)
        imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
        self.imgs = imgs
        self.lbls = lbls
        self.nrow = nrow
        self.ncol = ncol
        self.img_shape = (self.nrow, self.ncol, 1)
        self.size = len(self.imgs)
        self.transforms = transforms or []
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        def transform_single_img(img):
            img = np.reshape(img, self.img_shape)
            for t in self.transforms:
                img = t(img)
            return img
        img = self.imgs[index]
        if isinstance(index, slice):
            start = index.start or 0
            step = index.step or 1
            length = int(np.ceil((index.stop - start) / step))
            imgs = np.reshape(img, (length, ) + self.img_shape)
            new_img = []
            for img in imgs:
                new_img.append(
                    transform_single_img(img).reshape((1, ) + self.img_shape))
            return np.concatenate(new_img, axis=0), np.array(self.lbls[index])
        else:
            img = transform_single_img(img)
            return img, np.array(self.lbls[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.size
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        meta_path = os.path.join(base_folder, "batches.meta")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="bytes")
        npb = meta[b"num_cases_per_batch"]
        assert meta[b"num_vis"] == 3 * 32 * 32
        def load_file(filename):
            with open(filename, "rb") as f:
                raw_data = pickle.load(f, encoding="bytes")
                x = raw_data[b"data"] / 255
                y = raw_data[b"labels"]
                x = x.reshape(npb, 3, 32, 32).astype("float")
                y = np.array(y)
            return x, y
        xs = []
        ys = []
        if train:
            for i in range(1, 6):
                filename = os.path.join(base_folder, f"data_batch_{i}")
                x, y = load_file(filename)
                xs.append(x)
                ys.append(y)
            self.X = np.concatenate(xs)
            self.y = np.concatenate(ys)
        else:
            filename = os.path.join(base_folder, "test_batch")
            x, y = load_file(filename)
            self.X = x
            self.y = y

        self.img_shape = (3, 32, 32)
        self.p = p
        self.transforms = transforms or tuple()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        def transform_single_img(img):
            img = np.reshape(img, self.img_shape)
            for t in self.transforms:
                img = t(img)
            return img
        img = self.X[index]
        if isinstance(index, slice):
            start = index.start or 0
            step = index.step or 1
            length = int(np.ceil((index.stop - start) / step))
            imgs = np.reshape(img, (length, ) + self.img_shape)
            new_img = []
            for img in imgs:
                new_img.append(
                    transform_single_img(img).reshape((1, ) + self.img_shape))
            return np.concatenate(new_img, axis=0), np.array(self.y[index])
        else:
            img = transform_single_img(img)
            return img, np.array(self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                for word in line.strip().split():
                    ids.append(self.dictionary.add_word(word))
                ids.append(self.dictionary.add_word("<eos>"))
                if max_lines is not None and i > max_lines:
                    break
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    data = np.array(data)
    assert len(data.shape) == 1
    # trim off the remainder
    remainder = data.shape[0] % batch_size
    if remainder:
        data = data[:-remainder]
    return data.reshape((batch_size, data.shape[0] // batch_size)).transpose()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    bs = batches.shape[1]
    chunk = batches[i:i + bptt]
    tgt = (batches[i + 1:i + 1 + bptt]).reshape((bptt * bs,))
    chunk = Tensor.make_const(nd.array(chunk, dtype, device))
    tgt = Tensor.make_const(nd.array(tgt, dtype, device))
    return chunk, tgt
    ### END YOUR SOLUTION