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

```
        C(xy) - min(C(x), C(y))
NCD =    --------------------
            max(C(x), C(y))
```

Where C(xy) is the compressed length of combining x and y.

The algorithm is awesomely simple.

Let x be a string you want to classify. Let y be a labeled training sample. We compute NCD(x,y) for each y in the training set. 
This gives us a "distance matrix" which we will use to perform K-Nearest-Neighbors on to classify.

Sounds like a for-loop to me ;).

## Performance
> Keep in mind, all of this was done on a 2022 M1 Macbook pro.

TODO:

## Evaluation

NOTES:
using the agnews dataset, the SCI/Tech training samples are too loose. A significant number of samples labeled as sci/tech should actually be labeled something else.
this causes the classifier to usually default to sci/tech when taking the top k.

## Check it out yourself

```
# do some clonin'
git clone https://github.com/jackycamp/knnzip.git
cd knnzip

# do some buildin'
cargo build --release

# do some data downloadin'
mkdir -p data/ag_news
wget -P data/ag_news/train.csv https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
wget -P data/ag_news/test.csv https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv 

# for a single sample
./target/release/knnzip --test-sample "Earth's Forces Are Causing This Massive Plate to Split in Two"

# or for all of the test samples in test.csv
./target/release/knnzip --test-path data/ag_news/test.csv

```
