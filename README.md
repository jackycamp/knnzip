# knnzip

Rust implementation of the paper: [Less is More: Parameter-Free Text Classification with Gzip](https://arxiv.org/pdf/2212.09410.pdf)

## Motivation

In a world where neural nets are thrown at everything, every once in a while, you just gotta take a step back, relax, and go non-parametric.

The homies refer to this as losing weight. ¬‿¬

The paper originally implemented the alg in python. So, although the simplicity shined down from the heavens, it took a while to touch the ground.

Rust ftw.

## How does it work?

To understand the implementation, it's probably helpful to understand Kolmogorov complexity a bit.

Kolmogorov complexity is a way of measuring how complex a piece of information/data is.
Given data x, the kolmogorov complexity is the length of the shortest program that can produce x. 
(Assume the program is written in binary code so the program is a series of 1's and 0's).

So Kolmogorov complexity can be represented as K(x). If K(x) is large then the data is considered
to be very complex.

However, K(x) is incomputable (no known algorithm that can take any x and determine the EXACT shorted program that generates x [see halting problem])

The paper explains that you can __approximate__ K(x) by maximally compressing x (represented by C(x)).

Since you can approximate K(x) then you can approximate the differences or distances between K(x) and K(y).

Which brings us to a computable distance of information, Normalized Compression Distance (NCD).

        C(xy) - min(C(x), C(y))
NCD =    --------------------
            max(C(x), C(y))

Where C(xy) is the compressed length of combining x and y.

The algorithm is awesomely simple.

For a string you want to classify x, and each labeled sample y, you can build an NCD(x,y) distance matrix and then perform K nearest neighbors to classify.


## Evaluation

NOTES:
using the agnews dataset, the SCI/Tech training samples are too loose. A significant number of samples labeled as sci/tech should actually be labeled something else.
this causes the classifier to usually default to sci/tech when taking the top k.
