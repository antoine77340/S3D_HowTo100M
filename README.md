# PyTorch S3D Text-Video trained HowTo100M
This repo contains a PyTorch S3D Text-Video model trained from scratch on HowTo100M using MIL-NCE [1]
If you use this model, we would appreciate if you could cite [1] and [2] :).

The official Tensorflow hub version of this model can be found here: https://tfhub.dev/deepmind/mil-nce/s3d/1
with a colab on how to use it here: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/text_to_video_retrieval_with_s3d_milnce.ipynb

## Getting the data

You will first need to download the model weights and the word dictionary.

```sh
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
```


## How To use it ?

The following code explain how to instantiate S3D Text-Video with the pretrained weights and run inference
on some examples.

```python
import torch as th
from s3dg import S3D

# Instantiate the model
net = S3D('s3d_dict.npy', 512)

# Load the model weights
net.load_state_dict(th.load('s3d_howto100m.pth'))

# Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1] 
video = th.rand(2, 3, 32, 224, 224)

# Evaluation mode
net = net.eval()
 
# Video inference
video_output = net(video)

# Text inference
text_output = net.text_module(['open door', 'cut tomato'])
```
NB: The video network is fully convolutional (with global average pooling in time and space at the end). However, we recommend using T=32 frames (same as during training), T=16 frames also works ok. For H and W we have been using values from 200 to 256.

*video_output* is a dictionary containing two keys:
- *video_embedding*: This is the video embedding (size 512) from the joint text-video space. It should be used to compute similarity scores with text inputs using the text embedding.
- *mixed_5c*: This is the global averaged pooled feature from S3D of dimension 1024. This should be use for classification on downstream tasks.

*text_output* is also a dictionary with a single key:
- *text_embedding*: It is the text embedding (size 512) from the joint text-video space. To compute the similarity score between text and video, you would compute the dot product between *text_embedding* and *video_embedding*.

## Computing all the pairwise video-text similarities:

The similarity scores can be computed with a dot product between the *text_embedding* and the *video_embedding*.

```python
video_embedding = video_output['video_embedding']
text_embedding = text_output['text_embedding']
# We compute all the pairwise similarity scores between video and text.
similarity_matrix = th.matmul(text_embedding, video_embedding.t())
```


## References 

[1] A. Miech, J.-B. Alayrac, L. Smaira, I. Laptev, J. Sivic and A. Zisserman,
End-to-End Learning of Visual Representations from Uncurated Instructional Videos.
https://arxiv.org/abs/1912.06430

[2] A. Miech, D. Zhukov, J.-B. Alayrac, M. Tapaswi, I. Laptev and J. Sivic, 
HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips.
https://arxiv.org/abs/1906.03327


Bibtex:

```bibtex
@inproceedings{miech19howto100m,
   title={How{T}o100{M}: {L}earning a {T}ext-{V}ideo {E}mbedding by {W}atching {H}undred {M}illion {N}arrated {V}ideo {C}lips},
   author={Miech, Antoine and Zhukov, Dimitri and Alayrac, Jean-Baptiste and Tapaswi, Makarand and Laptev, Ivan and Sivic, Josef},
   booktitle={ICCV},
   year={2019},
}

@inproceedings{miech19endtoend,
   title={{E}nd-to-{E}nd {L}earning of {V}isual {R}epresentations from {U}ncurated {I}nstructional {V}ideos},
   author={Miech, Antoine and Alayrac, Jean-Baptiste and Smaira, Lucas and Laptev, Ivan and Sivic, Josef and Zisserman, Andrew},
   booktitle={CVPR},
   year={2020},
}
```
# Acknowledgements
We would like to thank Yana Hasson for the help provided in the non trivial porting of the original Tensorflow weights to PyTorch.
