const ort = require("onnxruntime-node");
const sharp = require("sharp");
const fs = require("fs").promises;
const ndarray = require("ndarray");
const ops = require("ndarray-ops");
// const jpeg = require("jpeg-js");
const { createCanvas, loadImage } = require("canvas");

const drawBoundingBox = async (
  imagePath,
  boxes,
  classes,
  scores,
  classList,
  threshold = 0.5
) => {
  // Load the image
  const image = await loadImage(imagePath);
  const canvas = createCanvas(image.width, image.height);
  const context = canvas.getContext("2d");

  context.drawImage(image, 0, 0);

  boxes.forEach((box, i) => {
    if (scores[i] > threshold) {
      const [y1, x1, y2, x2] = box;

      // Draw bounding box
      context.beginPath();
      context.rect(x1, y1, x2 - x1, y2 - y1);
      context.lineWidth = 2;
      context.strokeStyle = "red";
      context.fillStyle = "red";
      context.stroke();

      // Draw label
      context.fillText(
        classList[classes[i]] + " " + scores[i].toFixed(2),
        x1,
        y1 > 10 ? y1 - 5 : 10
      );
    }
  });

  const buffer = canvas.toBuffer("image/png");
  fs.writeFileSync("./out.png", buffer);
};

// drawBoundingBox("input.png", boxes, classes, scores, [
//   "class1",
//   "class2",
//   "class3",
// ]);

async function preprocessImage(imagePath) {
  // Load image and convert to RGB, resize to 640x640 and normalize the pixels values to be between 0 and 1
  let { data, info } = await sharp(imagePath)
    .resize(640, 640)
    .raw()
    .toBuffer({ resolveWithObject: true });

  // Initialize an empty ndarray with the correct shape (3, 640, 640)
  let imageNdArray = ndarray(new Float32Array(3 * info.width * info.height), [
    3,
    info.width,
    info.height,
  ]);

  // Fill the ndarray with normalized image pixel data
  for (let y = 0; y < info.height; ++y) {
    for (let x = 0; x < info.width; ++x) {
      for (let ch = 0; ch < 3; ++ch) {
        // Normalized pixel value and transfer to ndarray
        let pixelValue = data[(info.width * y + x) * 3 + ch] / 255.0;
        imageNdArray.set(ch, y, x, pixelValue);
      }
    }
  }

  // Transpose the image to the format the model requires (1, 3, 640, 640)
  let transposedImageNdArray = ndarray(
    new Float32Array(1 * 3 * info.width * info.height),
    [1, 3, info.width, info.height]
  );
  ops.assign(transposedImageNdArray.pick(0), imageNdArray);

  return transposedImageNdArray;
}

async function predict() {
  const processedImage = await preprocessImage("./1.png");
  // Load your model
  // console.log(processedImage.data);
  const session = await ort.InferenceSession.create("./best.onnx");

  // Load your image
  let inputTensor = new ort.Tensor(
    "float32",
    processedImage.data,
    [1, 3, 640, 640]
  );
  // let inputTensor = new ort.Tensor(
  //   Float32Array.from(processedImage),
  //   "float32",
  //   processedImage.shape
  // );

  //   // Feed the inputs
  let feeds = { images: inputTensor }; // Replace 'input1' with the actual name of your model's input node

  //   // Run the model
  const output = session
    .run(feeds)
    .then((output) => {
      return output;
    })
    .catch((error) => {
      console.log(error);
    });
  return output;
}

function convertOutputToBoundingBoxes(
  output80,
  output40,
  output20,
  threshold = 0.5
) {
  const allOutputs = [output80, output40, output20];
  const allGridSizes = [80, 40, 20];
  const numClasses = 5;
  const boxes = [];

  for (let k = 0; k < allOutputs.length; k++) {
    const output = allOutputs[k];
    const gridSize = allGridSizes[k];

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        for (let a = 0; a < 3; a++) {
          const objectness = output[0][a][i][j][4];
          if (objectness > threshold) {
            const box = {};
            box.classProbabilities = output[0][a][i][j].slice(
              5,
              5 + numClasses
            );
            box.classId = box.classProbabilities.indexOf(
              Math.max(...box.classProbabilities)
            );
            box.score = objectness * box.classProbabilities[box.classId];
            box.x = output[0][a][i][j][0];
            box.y = output[0][a][i][j][1];
            box.width = output[0][a][i][j][2];
            box.height = output[0][a][i][j][3];
            boxes.push(box);
          }
        }
      }
    }
  }

  return boxes;
}

predict().then((output) => {
  let boundingBoxes = convertOutputToBoundingBoxes(
    output["534"],
    output.output,
    output["552"],
    0.5
  );
  console.log(boundingBoxes);
});
// console.log(boxes);
