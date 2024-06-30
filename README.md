<br>

**Python Bayesian Machine Learning via PyMC**

<br>

* [Environments](#environments)
  * [Remote Development](#remote-development)
  * [Remote Development & Integrated Development Environments](#remote-development--integrated-development-environments)
* [Code Analysis](#code-analysis)
  * [pylint](#pylint)
  * [pytest & pytest coverage](#pytest--pytest-coverage)
  * [flake8](#flake8)
* [References](#references)

<br>
<br>


## Environments

### Remote Development

For this project/template, the remote development environment requires

* [Dockerfile](.devcontainer/Dockerfile)
* [requirements.txt](.devcontainer/requirements.txt)

An image is built via the command

```shell
docker build . --file .devcontainer/Dockerfile -t uncertainty
```

On success, the output of

```shell
docker images
```

should include

<br>

| repository  | tag    | image id | created  | size     |
|:------------|:-------|:---------|:---------|:---------|
| uncertainty | latest | $\ldots$ | $\ldots$ | $\ldots$ |


<br>

Subsequently, run a container, i.e., an instance, of the image `uncertainty` via:

<br>

```shell
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 -w /app 
	--mount type=bind,src="$(pwd)",target=/app uncertainty
```

<br>

Herein, `-p 10000:8888` maps the host port `10000` to container port `8888`.  Note, the container's working environment,
i.e., -w, must be inline with this project's top directory.  Additionally

* --rm: [automatically remove container](https://docs.docker.com/engine/reference/commandline/run/#:~:text=a%20container%20exits-,%2D%2Drm,-Automatically%20remove%20the)
* -i: [interact](https://docs.docker.com/engine/reference/commandline/run/#:~:text=and%20reaps%20processes-,%2D%2Dinteractive,-%2C%20%2Di)
* -t: [tag](https://docs.docker.com/get-started/02_our_app/#:~:text=Finally%2C%20the-,%2Dt,-flag%20tags%20your)
* -p: [publish](https://docs.docker.com/engine/reference/commandline/run/#:~:text=%2D%2Dpublish%20%2C-,%2Dp,-Publish%20a%20container%E2%80%99s)

<br>

Get the name of the container via:

```shell
docker ps --all
```

Never deploy a root container, study the production [Dockerfile](Dockerfile); cf. [/.devcontainer/Dockerfile](.devcontainer/Dockerfile)

<br>

### Remote Development & Integrated Development Environments

An IDE (integrated development environment) is a helpful remote development tool.  The **IntelliJ
IDEA** set up involves connecting to a machine's Docker [daemon](https://www.jetbrains.com/help/idea/docker.html#connect_to_docker), the steps are

<br>

> * **Settings** $\rightarrow$ **Build, Execution, Deployment** $\rightarrow$ **Docker** $\rightarrow$ **WSL:** {select the linux operating system}
> * **View** $\rightarrow$ **Tool Window** $\rightarrow$ **Services** <br>Within the **Containers** section connect to the running instance of interest, or ascertain connection to the running instance of interest.

<br>

**Visual Studio Code** has its container attachment instructions; study [Attach Container](https://code.visualstudio.com/docs/devcontainers/attach-container).


<br>
<br>



## Code Analysis

The GitHub Actions script [main.yml](.github/workflows/main.yml) conducts code analysis within a Cloud GitHub Workspace.  Depending on the script, code analysis may occur `on push` to any repository branch, or `on push` to a specific branch.

The sections herein outline remote code analysis.

<br>

### pylint

The directive

```shell
pylint --generate-rcfile > .pylintrc
```

generates the dotfile `.pylintrc` of the static code analyser [pylint](https://pylint.pycqa.org/en/latest/user_guide/checkers/features.html).  Analyse a directory via the command

```shell
python -m pylint --rcfile .pylintrc {directory}
```

The `.pylintrc` file of this template project has been **amended to adhere to team norms**, including

* Maximum number of characters on a single line.
  > max-line-length=127

* Maximum number of lines in a module.
  > max-module-lines=135


<br>


### pytest & pytest coverage

> [!IMPORTANT]
> Within main.yml, enable pytest & pytest coverage via patterns akin to
> 
> * pytest -o python_files=test_*
> * pytest --cov-report term-missing  --cov src/data/... tests/data/...
> 

Test a program via

```shell
python -m pytest ...
```


<br>


### flake8

For code & complexity analysis.  A directive of the form

```bash
python -m flake8 --count --select=E9,F63,F7,F82 --show-source 
	--statistics src/{directory.name}
```

inspects issues in relation to logic (F7), syntax (Python E9, Flake F7), mathematical formulae symbols (F63), undefined variable names (F82).  Additionally

```shell
python -m flake8 --count --exit-zero --max-complexity=10 --max-line-length=127 
	--statistics src/{directory.name}
```

inspects complexity.


<br>
<br>


## References

JAX
* [JAX](https://jax.readthedocs.io/en/latest/)
* [Building JAX Dependent Products](https://jax.readthedocs.io/en/latest/installation.html)
* [Snippets](https://github.com/google/jax)
* [JAX Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax), [JAX Early Access](https://developer.nvidia.com/jax-container-early-access)
    * What Is In The Container?

<br>

XLA (Accelerated Linear Algebra)
* [OpenXLA](https://openxla.org/xla)
* [Graphics Processing Unit & XLA Flags](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)

<br>

[PyMC](https://www.pymc.io/welcome.html)
* [PyMC, JAX, Graphics Processing Unit](https://www.pymc-labs.com/blog-posts/pymc-stan-benchmark/)
* [Model Checking](https://pymcmc.readthedocs.io/en/latest/modelchecking.html)
* [Sampling](https://www.pymc-labs.com/blog-posts/pymc-stan-benchmark/)
* [Glossary](https://www.pymc.io/projects/docs/en/latest/glossary.html)
    * [No U Turn Sampler](https://www.pymc.io/projects/docs/en/latest/glossary.html#term-No-U-Turn-Sampler)

<br>

[Multiprocessing](https://docs.python.org/3.10/library/multiprocessing.html#)
* [Multiprocessing Context](https://superfastpython.com/multiprocessing-context-in-python/)
* [Multiprocessing & Python](https://superfastpython.com/multiprocessing-in-python/)
* Fork Issues
    * [details](https://docs.python.org/3/library/os.html#os.fork)
    * [discussion](https://discuss.python.org/t/concerns-regarding-deprecation-of-fork-with-alive-threads/33555)

    
<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
