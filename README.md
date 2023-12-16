## Coptic Machine Translator

A Python research module containing code to generate and analyze neural machine translators, parse Coptic datasets, and code to create a simple Google-translate-like frontend. This repo was used to build https://www.coptictranslator.com.


### Installing the Package
To install the `translation` package, you must first ensure that your virtual environment has all the required dependencies. Run `pip install -e .` from the base directory and install all required modules into your virtual environment.

### Running the Website
The translation API paths are stored in the `NEXT_PUBLIC_ENGLISH_API` and `NEXT_PUBLIC_COPTIC_API` environmental variables. To run the website locally with your own translation APIs, refer to the instructions in the [Running a Backend](#running-a-backend) section.

To run the frontend:
1. Make sure `npm` is installed onto your machine.
2. Navigate to the `website/coptic-translator` directory.
3. Run `npm install`.
4. Run `npm run dev`.

### Running a Backend
The translation models are not included in the GitHub repo. Eventually, we plan to host our model weights on HuggingFace. If you would like to host your own translation models, we recommend the following steps:
1. Load source-target and target-source PyTorch models (from HuggingFace, or train your own using the provided utilities).
2. Create a source-target and target-source REST API using [TorchServe](https://pytorch.org/serve/).
3. Update the destination in `next.config.js` with your backend server address.
4. Store the APIs in the `NEXT_PUBLIC_ENGLISH_API` and `NEXT_PUBLIC_COPTIC_API` environmental variables (for local developemnt, put them in a `.env.local` file)
5. Run the server and the frontend. Now, your frontend should have full functionality.

### License

This project is licensed under the MIT License - see the LICENSE.md file for details.

### MIT License Summary
You can use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.
This software is provided "AS IS", without warranty of any kind.
You must include the original copyright and permission notice in any copy of the software or substantial portion of it.
For the full MIT License text, please see the LICENSE.md file in this repository.












