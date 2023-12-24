## Coptic Machine Translator

A Python research module containing code to generate and analyze neural machine translators, parse Coptic datasets, and code to create a simple Google-translate-like frontend. This repo was used to build https://www.coptictranslator.com.


### Installing the Package
To install the `translation` package, you must first ensure that your virtual environment has all the required dependencies. Run `pip install -e .` from the base directory and install all required modules into your virtual environment.

### Running the website

Refer to the [frontend](https://github.com/Coptic-NMT/coptic-translator-frontend) repo for instrucitons on running the UI locally.

### Running this middleware server with a GCP server Endpoint

1. Create a `.env` file in the `server` folder
2. Put the following fields in the `.env` file. Replace `<field>` with your values.
```
COPTIC_TO_ENGLISH_ENDPOINT_ID="<cop-eng endpoint id>"
ENGLISH_TO_COPTIC_ENDPOINT_ID="<eng-cop endpoint id>"
PROJECT_ID="<project id>"
LOCATION="<location>"
PORT="<port to run the middleware on>"
```
1. Call
```bash
npm start
```

### Dockerizing a model

To deploy a model, you first need to put in a docker container.
1. Place your model artifacts in `models/<model name>` folder. This should include the following files
   1. Model config files
   2. Tokenizer files
   3. `config.properties` - see the documentation [here](https://pytorch.org/serve/configuration.html).
   4. `handler.py` - see the documentation [here](https://pytorch.org/serve/custom_service.html?highlight=handler#custom-handlers)
   5. `pytorch_model.bin` - weights
2. Run the docker build script. Use `-gpu` flag if you plan on serving your model on a GPU. You must include the `-m` flag with 
the name of your model in the `models/` directory
Example:
```bash
./build_docker.sh -gpu -m cop-en-norm-group-greekified
```

To run your docker contiainer use
```bash
docker run -p 7080:7080 us-central1-docker.pkg.dev/coptic-translation/nmt-docker-repo/<model name>:<processor>
```

Run this to push your model to the GCP repo (if you have access)
```bash
docker push us-central1-docker.pkg.dev/coptic-translation/nmt-docker-repo/<model name>:<processor>
```

### License

This project is licensed under the MIT License - see the LICENSE.md file for details.

### MIT License Summary
You can use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.
This software is provided "AS IS", without warranty of any kind.
You must include the original copyright and permission notice in any copy of the software or substantial portion of it.
For the full MIT License text, please see the LICENSE.md file in this repository.












