import torch
from typing import List, Callable, Tuple, Any
from spacy.tokens import Doc
import spacy
from thinc.types import Floats2d, Floats3d, Ints2d, Ints3d
from thinc.types import List2d, Array3d, ArgsKwargs
from thinc.api import Model, PyTorchWrapper, chain, list2ragged
from thinc.api import registry as thinc_registry
import torch.nn.functional as F
from thinc.util import is_torch_array, convert_recursive, torch2xp
from thinc.util import is_xp_array, xp2torch

"""
http://aclanthology.lst.uni-saarland.de/W16-4816.pdf
https://aclanthology.org/2020.emnlp-main.371.pdf
https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
https://github.com/danifg/BottomUp-Hierarchical-PtrNet/blob/master/neuronlp2/models/parsing.py
https://arxiv.org/pdf/2103.06874.pdf
https://github.com/huggingface/transformers/blob/master/src/transformers/models/canine/tokenization_canine.py
https://github.com/explosion/spaCy/blob/master/spacy/tokens/doc.pyx
https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/canine/modeling_canine.py#L66
"""


@thinc_registry.layers("spacy.doc2bytes.v1")
def docs2utf8() -> Model[List[Doc], List[Ints2d]]:
    """
    Layer to convert a list of docs into variable length utf8 arrays.
    """
    return Model("doc2bytes", forward_docs2utf8)


def _doc2utf8(doc: Doc, xp) -> Ints2d:
    """
    Convet a single doc into a number of words per number of bytes matrix.
    The integer 0 is reserved for padding and 256 for START and 257 for END.
    """
    byte_strings = [token.orth_.encode('utf8') for token in doc]
    
    byte_strings = list(map(list, byte_strings))
    nr_char = max(len(bs) for bs in byte_strings)
    output = xp.zeros((len(byte_strings), nr_char), dtype='int')
    # Reserving 0 for padding
    output -= 1
    for i, bs in enumerate(byte_strings):
        output[i, :len(bs)] = xp.asarray(bs)
    output += 1
    return output


def forward_docs2utf8(
    model: Model[List[Doc], List[Ints2d]],
    docs: List[Doc],
    is_train: bool
) -> List[Ints2d]:
    outputs = []
    for doc in docs:
        outputs.append(_doc2utf8(doc, model.ops.xp))
    backprop: Callable[[List[Ints2d]], List] = lambda d_features: []
    return outputs, backprop


@thinc_registry.layers("spacy.with_padutf8arrays.v1")
def with_padutf8arrays(
    layer: Model[Array3d, Array3d]
) -> Model[List2d, List2d]:
    """
    Utility layer that pads and unpads sequences.
    It helps to parallelize the character-level model
    over the num_docs x num_tokens x num_chars array
    across all docs.
    """
    return Model(
        f"with_padutf8arrays({layer.name})",
        forward_padutf8arrays,
        init=init_padutf8arrays,
        layers=[layer],
        dims={name: layer.maybe_get_dim(name) for name in layer.dim_names}
    )


def init_padutf8arrays(model: Model = None, X=None, Y=None):
    """
    Doesn't really do anything other than calling "initialize"
    on the PytorchWrapper() model.
    """
    layer = model.layers[0]
    layer.initialize()
    return model


def _seqs2padded(
    model,
    arrs: List2d
) -> Tuple[Array3d, List[Tuple[int, int]]]:
    """
    The 'arrs' can vary in both dimensions and
    the output has the max-size over both dimensions.
    """
    shapes = [arr.shape for arr in arrs]
    dim0 = len(shapes)
    dim1 = max(shape[0] for shape in shapes)
    dim2 = max(shape[1] for shape in shapes)
    output = model.ops.xp.zeros((dim0, dim1, dim2), dtype='int')
    for i, arr in enumerate(arrs):
        output[i, :shapes[i][0], :shapes[i][1]] = arr
    return output, shapes


def _padded2seqs(
    tensor: Array3d,
    shapes: List[Tuple[int, int]]
) -> List2d:
    """
    Slices an 3d tensor into 2d matrices
    specified by the shapes.
    """
    output = []
    for i, sh in enumerate(shapes):
        mat = tensor[i, :sh[0], :]
        output.append(mat)
    return output


def forward_padutf8arrays(
    model,
    arrs: List2d,
    is_train: bool
) -> Tuple[List2d, Callable]:
    """
    Take a list of 2d ints and generate one big padded version
    and also return the original shapes.
    """
    layer = model.layers[0]
    Xf, shapes = _seqs2padded(model, arrs)
    Yf, get_dX = layer(Xf, is_train)
    def backprop(dY: List2d) -> List2d:
        dYf, dYshapes = _seqs2padded(model, dY)
        dXf = get_dX(dYf)
        return []
        # return _padded2seqs(dXf, dYshapes)

    return _padded2seqs(Yf, shapes), backprop


def convert_inputs(model: Model, X, is_train: bool):
    def reverse_conversion(dXtorch):
        dX = convert_recursive(is_torch_array, torch2xp, dXtorch)
        return dX.args
    converted = xp2torch(X, requires_grad=False)
    return ArgsKwargs(args=(converted,), kwargs={}), reverse_conversion


class PytorchCharConv(torch.nn.Module):
    """
    A single-layer conv with max-pool.
    It has vocabulary of 257: 256 for utf8 + 1 for padding (0)
    """
    def __init__(self, width, num_chars=257, char_dim=64,
                 filter_size=[1, 2, 3]):
        super(PytorchCharConv, self).__init__()
        self.char_embed = torch.nn.Embedding(num_chars, char_dim)
        self.filter_size = filter_size

        if len(self.filter_size) > 1:
            self.conv1d = torch.nn.ModuleList(
                [torch.nn.Conv1d(
                    char_dim,
                    width,
                    size,
                    padding=size - 1
                )
            for size in self.filter_size
                 ]
            )
            self.bottleneck = torch.nn.Linear(
                len(filter_size) * width,
                width
            )
        else:
            self.conv1d = torch.nn.Conv1d(
                char_dim,
                width,
                filter_size,
                padding=2
            )
            self.bottleneck = torch.nn.Identity()
        self.normalize_conv = torch.nn.LayerNorm(width * len(filter_size))
        self.normalize_out = torch.nn.LayerNorm(width)
        self.dropout = torch.nn.Dropout2d(p=0.0)

    # batch x num_tokens x num_chars --> batch x num_tokens x dim
    def forward(self, input_ids: Ints3d) -> Floats3d:
        x = self.char_embed(input_ids)
        size = x.size()
        x = x.view(size[0] * size[1], size[2], size[3]).transpose(1, 2)
        if len(self.filter_size) == 1:
            x = self.conv1d(x)
            x = F.relu(x)
            x, _ = x.max(dim=2)
        else:
            x_conv = [F.silu(conv1d(x)) for conv1d in self.conv1d]
            x_max  = [x.max(dim=2)[0] for x in x_conv]
            x = torch.concat(x_max, dim=1)
        x = x.view(size[0], size[1], -1)
        x = self.normalize_conv(x)
        x = self.bottleneck(x)
        x = self.normalize_out(x)
        x = self.dropout(x)
        return x


@spacy.registry.architectures("spacy.UTF8Conv.v1")
def build_UTF8Conv(
    width: int,
) -> Model[List[Doc], List[Floats2d]]:
    """
    Takes a list of documents, turns them into utf8 bytes,
    and computes byte-level convolution for each token separately.
    It makes use of some custom utilities to be able to
    parallelize across all tokens in all documents.

    The `docs2utf8` utility takes a list of documents and returns
    a list of Ints2d matrices. Each matrix is number of tokens,
    by number of max-characters per document, so both dimensions
    vary.

    The 'with_padutf8arrays' utility creates a single Inst3d
    array packing all the Ints2d arrays into a
    num_docs x num_tokens x max_chars array. It provides this tensor
    to the PytorchCharConv and shapes it back after.

    """
    model = chain(
        docs2utf8(),
        with_padutf8arrays(
            PyTorchWrapper(
                PytorchCharConv(width),
                convert_inputs=convert_inputs
            )
        )
    )
    return model


if __name__ == "__main__":
    convnet = build_UTF8Conv()
    nlp = spacy.blank('en')
    text1 = 'hello i am a doc'
    text2 = 'i am a lovely teapot man'
    docs = [nlp(text1), nlp(text2)]
    stuff, backprop = convnet(docs, is_train=False)
    for i in stuff:
        print(i.shape)
