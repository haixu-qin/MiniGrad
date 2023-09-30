
# MiniGrad
\- a Mini Auto Differentiation / Gradient Descent library for teaching purposes. \
Ref: [micrograd](https://github.com/karpathy/micrograd), and other online webs.

Probably one of the lightest NN library online. \
much faster for small datasets. \
Pros: small space, fast runtime. easy to use for small datasets. \
Cons: not optimized in terms of big datasets.

## Example 
Prereqs: `pip3 install numpy` \
Run: `python3 example.py`

Example Output: [0, 1, 1, 0]

## Math Explanation 
Math Explanation for `.backward()` (for educational purposes). \
It could be found [there](docs/math1.md).

Math Explanation for `.backward_opt()` (for better and faster convergence). \
It could be found [there](docs/math2.md).

## Optimizers.
Optimizer is used by the library by default. \
[RAdam](https://arxiv.org/abs/1908.03265) (Rectified Adam), which introduces a rectification term in addition to the popular standard Adam optimizer (which uses Momentum and RMSprop and a bias-correction mechanism). It performs well for small datasets.


