const express = require("express");
const path = require("path");
const fs = require("fs");
const dotenv = require("dotenv");
const aiplatform = require("@google-cloud/aiplatform");
const { PredictionServiceClient } = aiplatform.v1;
const { helpers } = aiplatform;
const app = express();

// Check if the .env file exists
const envFilePath = path.resolve(__dirname, ".env");
if (fs.existsSync(envFilePath)) {
  // Load variables from the .env file
  dotenv.config({ path: envFilePath });
}

const port = parseInt(process.env.PORT) || 8080;
const projectId = process.env.PROJECT_ID;
const locationId = process.env.LOCATION;
const clientOptions = {
  apiEndpoint: `${locationId}-aiplatform.googleapis.com`,
};

const copticToEnglishEndpointId = process.env.COPTIC_TO_ENGLISH_ENDPOINT_ID;
const englishToCopticEndpointId = process.env.ENGLISH_TO_COPTIC_ENDPOINT_ID;
const copticToEnglishEndpoint = `projects/${projectId}/locations/${locationId}/endpoints/${copticToEnglishEndpointId}`;
const englishToCopticEndpoint = `projects/${projectId}/locations/${locationId}/endpoints/${englishToCopticEndpointId}`;

const predictionServiceClient = new PredictionServiceClient(clientOptions);

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
  const instance = helpers.toValue({
    data: { b64: Buffer.from(req, "utf-8").toString("base64") },
  });
  const instances = [instance];
  const request = {
    endpoint:
      api === "coptic" ? englishToCopticEndpoint : copticToEnglishEndpoint,
    instances,
  };

  try {
    const [response] = await predictionServiceClient.predict(request);
    const prediction = response.predictions[0];
    const val = prediction?.structValue?.fields?.translation?.stringValue || "";
    const statusMsg =
      prediction?.structValue?.fields?.message?.stringValue || "";
    const statusCode = prediction?.structValue?.fields?.code?.numberValue;

    const obj = {};
    if (statusMsg) {
      obj.message = statusMsg;
    }
    if (val) {
      obj.translation = val;
    }

    res.status(statusCode).json(obj);
  } catch (error) {
    res.status(500).json({ code: 500, message: "InternalServerError" });
  }
}

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}/`);
});
