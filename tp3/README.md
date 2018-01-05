# Preliminaries
For all practicals, we will make heavy use of python and many of its
libraries. To have access to the full power of python, virtual environments
and the command line, it is recommended to have access to a Unix system, an
emulated terminal (Windows 10 gives you access to such a terminal), or a
Virtual Machine.

Besides, we advice you to use conda to manage the various requirements that may
change from one practical to the other. Conda can be installed following [this
link](https://conda.io/docs/install/quick.html). Install the latest python 3
version of conda. Once conda is installed, before working on a practical, you
should create a new conda environment. For example for the third practical,
execute `conda create -n practical3`. You can then access your new environment
using `source activate practical3`. Each of the practical provides you with a
`requirements.txt` file. You can install all the necessary requirements by
running `pip install -r requirements.txt` in your virtual environment.

# Practical 3
In this first practical, you are asked to put what you just learnt
about bandits to good use. You are provided with the `main.py` file,
a bandits test bed. Use `python main.py -h` to check how you are
supposed to use this file. You will quickly notice that all but the
`eps` subcommand return error messages. Your job is to fix this behavior
by implementing optimistic, softmax and UCB agents in the `agents/agents.py`
file. 

## How do I complete these files ?
Just fill in the `# TO IMPLEMENT` part of the
code. Remove the expection raising part, and 
complete the three blank methods for each Agent.

In `__init__`, build the buffers your agent requires.
It might be interesting, for instance, to store the
number of time each action has been selected.

In `interact`, prescribe how the agent selects its
actions (interact must return an action, that is
an index in [0, ..., A]).

Finally, in update, implement how the agent updates
its buffers, using the newly observed `a` and `r`.
