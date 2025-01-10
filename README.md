# Scalable SAE Circuits in Gemma 2 9B

TLDR: 
We propose a novel approach to:

- Scaling SAE Circuits to Large Models: We find circuits in Gemma 9B by placing residual SAEs at intervals throughout the model, rather than at every layer and type.
- Developing a Better Circuit-Finding Algorithm: Our method uses a binary masking optimization over SAE latents, which proves significantly more effective than existing methods like Attribution Patching or Integrated Gradients.


Main Masking notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NainaniJatinZ/ScalableSAECircuits/blob/main/ScalableSAECircuits_Colab.ipynb)

Lesswrong post link: [Scaling Sparse Feature Circuit Finding to Gemma 9B](https://www.lesswrong.com/posts/PkeB4TLxgaNnSmddg/scaling-sparse-feature-circuit-finding-to-gemma-9b)


## Directions to run code

1. Download the data json files by downloading them or cloning the repo
```
git clone https://github.com/NainaniJatinZ/ScalableSAECircuits.git
```

2. Open the `ScalableSAECircuits_Colab.ipynb` and follow setup instructions to train/evaluate any of the 4 tasks covered in the lesswrong post