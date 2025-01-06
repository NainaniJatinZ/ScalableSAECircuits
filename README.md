# Scalable SAE Circuits in Gemma 2 9B

TLDR: 
We propose a novel approach to:

- Scaling SAE Circuits to Large Models: We find circuits in Gemma 9B by placing residual SAEs at intervals throughout the model, rather than at every layer and type.
- Developing a Better Circuit-Finding Algorithm: Our method uses a binary masking optimization over SAE latents, which proves significantly more effective than existing methods like Attribution Patching or Integrated Gradients.


Lesswrong post link: 
