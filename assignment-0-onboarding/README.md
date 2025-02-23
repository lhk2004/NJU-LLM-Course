[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Nl7eBamx)
# assignment-0-onboarding
This assignment is designed to help you get familiar with the programming environment, submission process and basic pytorch functionalities. 

By completing it, you'll ensure that your development setup is working properly, understand how to submit your work for future assignments, and strengthen your pytorch coding skills.


## Tasks

### Task 1 (100 points)

#### TODO

You are required to implement a python function named `matmul_with_importance` in `src/functional.py` with pytorch.

#### Explanation

* According to its docstring, this function is to apply a special variant of matrix multiplication operation (denoted as `matmul`) of two tensors, where:
    * the input tensor is a 3D tensor with the shape `[batch_size, seq_len, hidden_size]`, which represents `batch_size` of sequences, and each sequence has `seq_len` elements, where each element is a row-vector with dimension `hidden_size`. We denote the input tensor as `A1` with the shape `[b, s, h]`.
    * the weight tensor is a 2D tensor with the shape `[hidden_size, embed_size]`, which represents a projection matrix that projects any row vector from `hidden_size`-dim to `embed_size`-dim. We denote the weight tensor as `W1` with the shape `[h, e]`.
* The naive `matmul` is just to apply `O1[i] = A1[i] @ W1` for each `i`-th sequence in the batch to get output tensor denoted as `O1` with the shape `[b, s, e]`.
* The multi-head variant of `matmul` involves splitting the `h` dimension of `A1` and `W1` into `num_heads` shards equally (*denoted as `nh`, provided as an argument*) and performing `matmul` on each pair of `A1` shard and `W1` shard individually. This transforms the input tensor into a 4D tensor, denoted as `A2` with the shape `[b, s, nh, hd]`, and accordingly transforms the weight tensor into a 3D tensor denoted as `W2` with the shape `[nh, hd, e]`. As a result, the output tensor becomes a 4D tensor as well, denoted as `O2` with the shape `[b, s, nh, e]`.
* Building on the multi-head version of `matmul`, we introduce an importance probability tensor, denoted as `P`, with the shape `[b, s]`. Each element in `P` represents the probability of how important the corresponding element in `A1` is relative to other elements within the same sequence.
* As a result, we aim to apply matmul only to the "important" elements in each sequence. The projected vectors of these important elements, totaling `total_important_seq_len` (*denoted as `t`*), are then gathered into an output tensor, denoted as `O3` with the shape `[t, nh, e]`.
* To precisely define what is considered "important", we provide two optional arguments:
    * `top_p`: A float in the range `[0., 1.]`. Only elements with a probability **equal to or higher** than `top_p` are considered "important". The default value is `1.0`.
    * `top_k`: An integer in the range `[1, ..., seq_len]`. For each sequence in the batch, only the elements with the `top_k` highest probabilities are considered "important". If `top_k` is not provided (default is `None`), it is treated as `top_k = seq_len`.
* Additionally, if the optional gradient of the output tensor (*`grad_output`, denoted as `dO3` with the same shape as `O3`*) is provided, we should also compute the gradients for the input tensor (*`grad_input`, denoted as `dA1` with the same shape as `A1`*) and the weight tensor (*`grad_weight`, denoted as `dW1` with the same shape as `W1`*). If `dO3` is not given, we return `None` for both `dA1` and `dW1`.

#### Summary

In summary, the core of the `matmul_with_importance` function is to compute and return a tuple of three (optional) tensors: either `(O3, dA1, dW1)` or `(O3, None, None)`, given as input two tensors `A1` and `W1` as `matmul` operators, an importance probability tensor `P` with `top_p` and `top_k` to control "importance", an optional gradient tensor `dO3`, and an integer `num_heads` to split the `hidden_size`.

#### Notice

* All given tensors are randomly initialized from the standard normal distribution `N(0, 1)` on the same device (either `cpu` or `cuda`), with the same data type (`float32`, `float16`, or `bfloat16`), and do **NOT** require gradients in any test cases.
* `top_p` and `top_k` are guaranteed to have valid values within their respective ranges in all test cases.
* `hidden_size` is guaranteed to be divisible by `num_heads` in all test cases.
* If `grad_output` is not provided, avoid computing gradients to improve efficiency and save memory.
* If `grad_output` is provided, you can compute gradients using pytorch’s autograd mechanism, but be cautious of potential **side effects**, which may be checked in the test cases.
* There are so many ways to perform the `matmul` operation in pytorch, including `@`, `torch.matmul`, `torch.mm`, `torch.bmm`, and `torch.einsum`, among others. It is recommended that you experiment with different methods to implement the task and explore the differences between them.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to pytorch:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the official English documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**

* [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
* [Pytorch Autograd Mechanism](https://pytorch.org/docs/stable/autograd.html#module-torch.autograd)
* [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)


## Environment

* You should have python 3.10+ installed on your machine.
* (*Optional*) You had better have Nvidia GPU(s) with CUDA12.0+ installed on your machine, otherwise some features may not work properly (*We will do our best to ensure that the difference in hardware does not affect your score.*).
* You are supposed to install all the necessary dependencies with the following command, **which may vary a little among different assignments**.
    ```python
    pip install -r requirements.txt
    ```
* (*Optional*) You are strongly recommended to use a docker image from [Nvidia Pytorch Release](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) like [23.10](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html#rel-23-10) or some newer version as your basic environment in case of denpendency conflicts.


## Submission

* You need to submit your assignment by `git commit` and `git push` this private repository **on the `main` branch** with the specified source files required to fullfill above **before the hard deadline**, otherwise **the delayed assignment will be rejected automatically**.
* Try **NOT to push unnecessary files**, especially some large ones like images, to your repository.
* If you encounter some special problems causing you miss the deadline, please contact the teacher directly (*See [Contact](#contact)*).


## Scoring

* Each assignment will be scored of the points in the range of `0~100` by downloading your code and running the `test_script.sh` script to execute a `test_score.py` file (*invisible to you as empty files*) for some test cases, where the specific files described in [Tasks](#tasks) will be imported locally on our own machine.
* **ALL** the files required to fulfill in [Tasks](#tasks) are under the `src/` directory, which is the **ONLY** directory that will be imported as a **python module**. Therefore, there are several things you should pay attention to:
    * 1. The `__init__.py` is essential for a python module, and we have already initialized all the necessary ones in `src/` for you, so **be careful** when you intend to modify any of them for personal purposes.
    * 2. If you have any other files supposed to be internally imported like `utils.py`, please make sure that they are all under the `src/` directory and **imported relatively** e.g. `from .utils import *`,  `from .common.utils import ...`, etc.
* You will get the maximum score of `100` points if you pass all the tests within the optional time limit.
* You will get the minimum score of `0` points if you fail all the tests within the optional time limit, or run into some exceptions.
* You will get any other score of the points in the middle of `0~100` if you pass part of the tests within the optional time limit, which is the sum of the scores of all the passed test cases as shown in the following table.
    | Test Case | Score | Other Info |
    | --- | --- | --- |
    | Task1 - Case1 | 20 |  |
    | Task1 - Case2 | 20 |  |
    | Task1 - Case3 | 20 |  |
    | Task1 - Case4 | 20 |  |
    | Task1 - Case5 | 20 |  |
    | Total | 100 |  |
* To help you debug:
    * 1. We will give some test cases as toy examples in the visible `test_toy.py` file, and you had better ensure your code works correctly on your own machine before submitting, with the following command (*Feel free to modify the `test_toy.py` file to your specific debugging needs, since we wouldn't run it to score your code*).
        ```python
        pytest test_toy.py
        ```
    * 2. We will pre-test on your intermediate submission **as frequently as possible**, and offer only the score feedback (*See [Feedback](#feedback)*) each time to allow you to improve your code for higher scores before the hard ddl.
* **Note:** The testing methods provided in the `test_toy.py` file are for debugging purposes only, and may differ from the actual tests we use in `test_score.py` to test and score your code. Thus, be careful, particularly when handling edge cases.


## Feedback

* After scoring your assignment, We will give you a score table like the example one shown below in a new file named `score.md`, by pushing it within a new commit to your repository on a temporary branch called `score-feedback` (*This branch is only for you to view your status on each test cases after every scoring, please do NOT use it for any other purposes*).
    | Test Case | Score | Status | Error Message |
    | --- | --- | --- | --- |
    | Task1 - Case1 | 20 | ✅ |  |
    | Task1 - Case2 | 20 | ✅ |  |
    | Task1 - Case3 | 20 | ✅ |  |
    | Task1 - Case4 | 20 | ✅ |  |
    | Task1 - Case5 | 20 | ✅ |  |
    | Total | 100 | 😊 |  |

* The meaning of the status icons are listed below:
    * ✅: passed the case
    * ❌: failed the case due to wrong answers
    * 🕛: failed the case due to timeout if the time limit is set
    * ❓: failed the case due to some exceptions (the error message will be shown at the corresponding `Error Message` cell)
    * 😊: all passed
    * 🥺: failed at least one case



## Contact

* If you have any questions about the assignment, you can contact the teacher or any assistants directly through QQ group with the number `208283743`.
* You can subscribe to the teacher's bilibili account with UID `390606417` and watch the online courses [here](https://space.bilibili.com/390606417/channel/collectiondetail?sid=3771310).