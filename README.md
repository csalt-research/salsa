<p align="center">
  <a href="https://github.com/csalt-research">
    <img src="https://avatars.githubusercontent.com/u/43694569?s=200&v=4" alt="CSALT @ IITB" width="150" height="150">
  </a>
  <h3 align="center">SALSA: Speedy ASR-LLM Synchronous Aggregation</h3>
  <p align="center"> Accepted to Interspeech 2024
    <br/>
    <br/>
  </p>
</p>
  
<a href="#"> <img src="https://img.shields.io/badge/PDF-Arxiv-teal"></a> ![Downloads](https://img.shields.io/github/downloads/csalt-research/salsa/total.svg) ![Contributors](https://img.shields.io/github/contributors/csalt-research/salsa?color=dark-green) ![Forks](https://img.shields.io/github/forks/csalt-research/salsa?style=social) ![Stargazers](https://img.shields.io/github/stars/csalt-research/salsa?style=social) 

## Table Of Contents

* [About The Repository](#about-the-repository)
* [Getting Started](#getting-started)
  * [Prerequisites and Installation](#getting-started)
  * [Running experiments](#running-experiments)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [Authors](#authors)
* [License](#license)
<!-- * [Citation](#citation) -->

## About The Repository

This repository hosts the artifacts pertaining to our paper [**<samp>SALSA: Speedy ASR-LLM Synchronous Aggregation</samp>**](https://arxiv.org/abs/2408.16542) accepted to ***Interspeech 2024***.

The main contribution of our paper is :mag_right: a simple <samp>LLM stitching technique</samp> that uses a <samp>tokenization agnostic algorithm</samp> to combine the generation capability from *`LLAMA`* and speech comprehension from *`Whisper`*.


## Getting Started

Clone the repo:

```bash
git clone [https://github.com/Lightning-AI/lit-gpt](https://github.com/csalt-research/salsa)
cd salsa
```

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Install with all dependencies (including quantization, sentencepiece, tokenizers for Llama models, etc.):

```bash
pip install -r requirements-all.txt
```

Finally, to run the recipes you will also need access to LLAMA weights, which can be obtained here: [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)


You are all set! 🎉

&nbsp;


## Running experiments

Our codebase has a simple, easily customizable wrapper script `run.sh`, that contains the complete experimental setup divided into different stages. To run all stages, simply execute: 

```bash
./run.sh --hf_access_token <huggingface_personal_access_token>
```

(Please see this [blog]([tutorials/finetune_adapter.md](https://huggingface.co/docs/hub/en/security-tokens)) for details on how to generate a personal access token for hugging face)

To run a specific stage, let's say "dataset creation", you can execute:

```bash
./run.sh --hf_access_token <huggingface_personal_access_token> --stage 1 --stop_stage 1
```

Unfortunately, for datasets other than Fleurs, Commonvoice, and Librispeech, one must modify `data_preparation/dump_dataset_v2.py` and then proceed to run other stages as mentioned above.

&nbsp;

## Roadmap

See the [open issues](https://github.com/csalt-research/salsa/issues) for a list of proposed features (and known issues) relevant to this work. For <samp>lit-gpt</samp> related features/issues, checkout their [github repository](https://github.com/Lightning-AI/litgpt).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/csalt-research/salsa/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please open an individual PR for each suggestion.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add appropriate commit message'`). The correct way to write your commit message can be found [here](https://www.conventionalcommits.org/en/v1.0.0/)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## Authors

* **Ashish Mittal** - *Research Scientist, IBM Research & PhD, CSE, IIT Bombay* - [Ashish Mittal](https://www.linkedin.com/in/ashish-mittal-6720663a/)
* **Darshan Prabhu** - *PhD, CSE, IIT Bombay* - [Darshan Prabhu](https://www.linkedin.com/in/darshan-prabhu/)
* **Sunita Sarawagi** - *Associate Professor, CSE, IIT Bombay* - [Sunita Sarawagi](https://www.cse.iitb.ac.in/~sunita/)
* **Preethi Jyothi** - *Associate Professor, CSE, IIT Bombay* - [Preethi Jyothi](https://www.cse.iitb.ac.in/~pjyothi/)


<!--- 
## Citation

If you use this code for your research, please consider citing our work.

```bibtex
@misc{prabhu2023accented,
      title={Accented Speech Recognition With Accent-specific Codebooks}, 
      author={Darshan Prabhu and Preethi Jyothi and Sriram Ganapathy and Vinit Unni},
      year={2023},
      eprint={2310.15970},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
--->

## License

Distributed under the MIT License. See [LICENSE](https://github.com/csalt-research/accented-codebooks-asr/blob/main/LICENSE.md) for more information.
