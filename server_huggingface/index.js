const express = require("express");
const path = require("path");
const fs = require("fs");
const dotenv = require("dotenv");
const app = express();

const { greekify, degreekify } = require("./helper.js");

// Check if the .env file exists
const envFilePath = path.resolve(__dirname, ".env");
if (fs.existsSync(envFilePath)) {
  // Load variables from the .env file
  dotenv.config({ path: envFilePath });
}

const port = parseInt(process.env.PORT) || 8080;
const copticToEnglishEndpoint = process.env.COPTIC_TO_ENGLISH_ENDPOINT;
const englishToCopticEndpoint = process.env.ENGLISH_TO_COPTIC_ENDPOINT;

const generationConfig = {
  max_length: 20,
  max_new_tokens: 128,
  min_new_tokens: 1,
  min_length: 0,
  early_stopping: true,
  do_sample: false,
  num_beams: 5,
  num_beam_groups: 1,
  top_k: 50,
  top_p: 0.95,
  temperature: 1.0,
  diversity_penalty: 0.0,
};

const authHeaders = {
  Authorization: `Bearer ${process.env.API_TOKEN}`,
};

app.use(express.text());

app.post("/coptic", async (req, res) => {
  await handleRequest(req.body, res, "coptic");
});

app.post("/english", async (req, res) => {
  await handleRequest(req.body, res, "english");
});

app.use((req, res) => {
  res.status(404).json({ message: "Not Found" });
});

async function handleRequest(req, res, api) {
  console.log("Request: ", req);
  if (typeof req !== "string") {
    res.status(500).json({ code: 500, message: "InternalServerError" });
    return;
  }
  let input = req;
  let endpoint = englishToCopticEndpoint;
  if (api === "english") {
    input = input.toLowerCase();
    input = greekify(input);
    endpoint = copticToEnglishEndpoint;
  }
  console.log("Input: ", input);
  const instance = {
    inputs: [input],
    parameters: generationConfig,
  };

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        ...authHeaders,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(instance),
    }).then((res) => res.json());
    console.log("Response: ", response);
    if (response.error && response.error.includes("loading")) {
      setTimeout(() => {
        console.log("Retrying request: ", req);
        handleRequest(req, res, api);
      }, 5000);
      return;
    }
    let output = response[0].generated_text;
    if (api === "coptic") {
      output = degreekify(output);
    }
    console.log("Output: ", output);
    res.status(200).json({ translation: output });
  } catch (error) {
    console.log("Error: ", error);
    res.status(500).json({ code: 500, message: "InternalServerError" });
  }
}

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}/`);
});
