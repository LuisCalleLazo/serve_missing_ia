
from generated import predict_service_pb2_grpc, predict_service_pb2
import PredictionFace.generate_model as generate_model
import PredictionFace.predict_face as predict
import PredictionFace.predict_face_multiple as predict_mult

class PredictService(predict_service_pb2_grpc.PredictServiceServicer):

  def GenerateModel(self, request, context):
    success = generate_model.generate_model(
      request.name_model, request.type_save, request.folder_path
    )
    return predict_service_pb2.GenerateModelResponse(success = success)
  
  def PredictFace(self, request, context):
    result = predict.predict_face(request.image_path, "model")
    return predict_service_pb2.PredictFaceResponse(result = result)
  
  def PredictFaceMultiple(self, request, context):
    result = predict_mult.predict_face_multiple(request.folder_path, "model")
    return result