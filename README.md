# llm4cp: Large Language Model for Competitive Programming
> The current code is under heavy development and cannot guarantee correct execution. Please use it only as a reference. If you are interested in this project or have any questions, please feel free to create issues.


+ 2023.04.14: registered an AtCoder account with the username [llm4cp](https://atcoder.jp/users/llm4cp). In the future, every competition that the model participates in will be recorded, along with the version of the model and the competition results. All submitted code will also be published in this repository.
+ 2023.04.26: Currently, 500,000 problem-answer pairs have been trained in the SFT phase, but the performance is very poor. We are currently training PPO, and the label for the reward model is based on real data, so the mean squared error (MSE) is used as the loss function.
+ 2023.04.28: There are issues with the current RLHF process. After training for a few steps, the program produced by the actor can no longer be compiled, and this issue persists regardless of whether real rewards or reward models are used.
