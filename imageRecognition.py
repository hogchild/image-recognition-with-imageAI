from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(execution_path, "inception_v3_google-1a9a5a14.pth"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(
    os.path.join(execution_path, "images/laurenz-krabisch-U0f_XcB1tFY-unsplash.jpg"), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
